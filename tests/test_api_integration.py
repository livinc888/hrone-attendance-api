"""Integration tests: real FastAPI app + a REAL MongoDB 6.0+ (no mocks, so aggregation behaviour,
unique indexes, $densify / $setWindowFields and explain plans are genuinely exercised).

    TEST_MONGO_URI="mongodb://localhost:27017" python -m pytest tests -v
    (local: docker run -d -p 27017:27017 mongo:7)

Each test class creates its own throw-away database (hrone_test_<random>) and drops it afterwards;
your real MONGO_DB is never touched. Without TEST_MONGO_URI these tests are skipped.
HRONE_LARGE=1 additionally runs the 100,000-record explain / speed check.
All timestamps are fixed (July 2026), never the current clock.
"""
import copy
import importlib.util
import json
import os
import pathlib
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from tests import reference as ref
from tests.reference import ms

TEST_URI = os.environ.get("TEST_MONGO_URI")
MAIN_PATH = pathlib.Path(__file__).resolve().parents[1] / "app" / "main.py"


@unittest.skipUnless(TEST_URI, "set TEST_MONGO_URI to run integration tests against a real MongoDB")
class IntegrationBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient  # imported lazily so unit tests need no httpx

        cls.db_name = f"hrone_test_{uuid.uuid4().hex[:10]}"
        os.environ["MONGO_URI"], os.environ["MONGO_DB"] = TEST_URI, cls.db_name
        spec = importlib.util.spec_from_file_location(f"app_main_{cls.db_name}", MAIN_PATH)
        cls.main = importlib.util.module_from_spec(spec)  # private module instance bound to the test DB
        spec.loader.exec_module(cls.main)
        cls.TestClient = TestClient
        cls.client = TestClient(cls.main.app)
        cls.addClassCleanup(cls._cleanup)  # registered before anything else can fail
        cls.client.__enter__()  # runs the lifespan: creates the indexes at startup
        cls.db = cls.main.db

    @classmethod
    def _cleanup(cls):
        """Registered with addClassCleanup, so it also runs when setUpClass fails half way.
        Order matters: drop the database FIRST (the app's lifespan shutdown closes its MongoClient,
        and a closed client cannot be used), then stop the app, then close the client."""
        main = getattr(cls, "main", None)
        try:
            if main is not None:
                main.client.drop_database(cls.db_name)
        finally:
            try:
                if getattr(cls, "client", None) is not None:
                    cls.client.__exit__(None, None, None)
            finally:
                if main is not None:
                    main.client.close()  # idempotent; also covers a lifespan that never ran

    # ---- helpers -------------------------------------------------------
    def create_emp(self, code, dept="Eng", start="09:30", end="18:30", joined="2026-01-05"):
        r = self.client.post("/employees", json={"emp_code": code, "name": f"Name {code}", "email": f"{code}@x.io",
                                                 "department": dept, "shift_start": start, "shift_end": end,
                                                 "joined_on": joined})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def punch_in(self, code, at_ms=None, **extra):
        body = {"emp_code": code, **extra}
        if at_ms is not None:
            body["punched_at"] = at_ms
        return self.client.post("/attendance/punch-in", json=body)

    def punch_out(self, code, at_ms=None):
        return self.client.post("/attendance/punch-out", json={"emp_code": code, "punched_at": at_ms})

    def race(self, n, call):
        """Run call(client, i) on n threads released at the same instant; returns the list of responses."""
        barrier = threading.Barrier(n)

        def worker(i):
            c = self.TestClient(self.main.app)  # own client per thread (no lifespan, indexes already exist)
            barrier.wait()
            return call(c, i)

        with ThreadPoolExecutor(n) as pool:
            return list(pool.map(worker, range(n)))


# =========================================================================== #
class EmployeeAndAttendanceTests(IntegrationBase):
    def test_health_and_startup_indexes(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        emp = self.db.employees.index_information()
        att = self.db.attendance_logs.index_information()
        self.assertTrue(emp["uq_employee_code"]["unique"])
        self.assertTrue(att["uq_attendance_emp_date"]["unique"])
        for name in ("department_joined", "joined_on_idx"):
            self.assertIn(name, emp)
        for name in ("attendance_date_emp", "attendance_status_date_emp", "attendance_emp_punch_in"):
            self.assertIn(name, att)
        self.main.ensure_indexes()  # idempotent: running it again must not raise

    # ---- employees -----------------------------------------------------
    def test_create_employee_shape_defaults_and_duplicates(self):
        r = self.client.post("/employees", json={"emp_code": "EMP0101", "name": "A", "email": "a@b.co",
                                                 "department": "Eng", "joined_on": "2026-01-05"})
        self.assertEqual(r.status_code, 201)
        body = r.json()
        self.assertNotIn("_id", body)
        self.assertEqual((body["shift_start"], body["shift_end"]), ("09:30", "18:30"))
        self.assertIsInstance(body["created_at"], int)
        self.assertGreaterEqual(body["created_at"], 100_000_000_000)
        dup = self.client.post("/employees", json={"emp_code": "EMP0101", "name": "B", "email": "b@b.co",
                                                   "department": "Ops", "joined_on": "2026-01-05"})
        self.assertEqual(dup.status_code, 409)

    def test_create_employee_validation(self):
        good = {"emp_code": "EMP0102", "name": "A", "email": "a@b.co", "department": "Eng", "joined_on": "2026-01-05"}
        bad_cases = [
            {**good, "emp_code": "E1"}, {**good, "emp_code": "EMP12"}, {**good, "emp_code": "emp0102"},
            {**good, "email": "nope"}, {**good, "name": ""}, {**good, "department": ""},
            {**good, "joined_on": "2026-02-30"}, {**good, "joined_on": "06/07/2026"},
            {**good, "shift_start": "9:30"}, {**good, "shift_start": "24:00"},
            {**good, "shift_start": "10:00", "shift_end": "10:00"},
            {k: v for k, v in good.items() if k != "name"}, {**good, "emp_code": 102},
        ]
        for body in bad_cases:
            self.assertEqual(self.client.post("/employees", json=body).status_code, 422, body)

    def test_concurrent_duplicate_employee_creates(self):
        body = {"emp_code": "EMP0103", "name": "A", "email": "a@b.co", "department": "Eng", "joined_on": "2026-01-05"}
        codes = sorted(r.status_code for r in self.race(8, lambda c, i: c.post("/employees", json=body)))
        self.assertEqual(codes, [201] + [409] * 7)
        self.assertEqual(self.db.employees.count_documents({"emp_code": "EMP0103"}), 1)

    def test_list_employees_filter_sort_pagination(self):
        for code, dept in (("EMP0203", "ListA"), ("EMP0201", "ListA"), ("EMP0202", "ListB"), ("EMP0204", "ListA")):
            self.create_emp(code, dept)
        r = self.client.get("/employees", params={"department": "ListA", "page_size": 2}).json()
        self.assertEqual([e["emp_code"] for e in r["items"]], ["EMP0201", "EMP0203"])
        self.assertEqual((r["total"], r["page"], r["page_size"]), (3, 1, 2))  # total reflects the filter
        r2 = self.client.get("/employees", params={"department": "ListA", "page_size": 2, "page": 2}).json()
        self.assertEqual([e["emp_code"] for e in r2["items"]], ["EMP0204"])
        self.assertEqual(self.client.get("/employees", params={"department": "listA"}).json()["total"], 0)  # case-sensitive
        for params in ({"page": 0}, {"page_size": 0}, {"page_size": 101}, {"page": "x"}):
            self.assertEqual(self.client.get("/employees", params=params).status_code, 422, params)

    # ---- punch-in ------------------------------------------------------
    def test_punch_in_late_boundaries_spec_examples(self):
        self.create_emp("EMP0301")
        for day, (h, mi, s), expected in ((6, (9, 40, 0), 0), (7, (9, 40, 1), 10), (8, (10, 15, 59), 45), (9, (9, 30, 0), 0)):
            r = self.punch_in("EMP0301", ms(2026, 7, day, h, mi, s))
            self.assertEqual(r.status_code, 201, r.text)
            body = r.json()
            self.assertEqual(body["late_minutes"], expected, (day, h, mi, s))
            self.assertEqual(body["date"], f"2026-07-{day:02d}")
            self.assertEqual((body["punch_out"], body["work_hours"], body["overtime_minutes"], body["half_day"], body["history"]),
                             (None, None, 0, False, []))
            self.assertEqual(body["status"], "PRESENT")

    def test_punch_in_truncates_ignores_derived_fields_and_accepts_statuses(self):
        self.create_emp("EMP0302")
        body = self.punch_in("EMP0302", ms(2026, 7, 6, 9, 40, 0) + 900, late_minutes=999, work_hours=99, status="WFH").json()
        self.assertEqual(body["punch_in"] % 1000, 0)  # truncated, and the truncated value is returned
        self.assertEqual(body["late_minutes"], 0)      # 09:40:00.900 counts as 09:40:00 -> not late
        self.assertEqual((body["status"], body["work_hours"]), ("WFH", None))
        self.create_emp("EMP0303")
        self.assertEqual(self.punch_in("EMP0303", ms(2026, 7, 6, 9, 0), status="ON_DUTY").json()["status"], "ON_DUTY")

    def test_punch_in_rejects_bad_timestamps_and_statuses(self):
        self.create_emp("EMP0304")
        valid = ms(2026, 7, 6, 9, 0)
        for bad in (valid // 1000, float(valid), str(valid), True, 99_999_999_999, 4_102_444_800_001, -5, [valid], {"a": 1}):
            r = self.client.post("/attendance/punch-in", json={"emp_code": "EMP0304", "punched_at": bad})
            self.assertEqual(r.status_code, 422, bad)
        for status in ("ABSENT", "LEAVE", "HOLIDAY", "", None):
            self.assertEqual(self.client.post("/attendance/punch-in", json={"emp_code": "EMP0304", "punched_at": valid, "status": status}).status_code, 422, status)
        self.assertEqual(self.client.post("/attendance/punch-in", json={"punched_at": valid}).status_code, 422)
        self.assertEqual(self.db.attendance_logs.count_documents({"emp_code": "EMP0304"}), 0)  # nothing was written
        # the exact bounds of the allowed range are accepted
        self.assertEqual(self.punch_in("EMP0304", 100_000_000_000).status_code, 201)

    def test_punch_in_unknown_employee_and_duplicate(self):
        self.assertEqual(self.punch_in("EMP9998", ms(2026, 7, 6, 9, 0)).status_code, 404)
        self.create_emp("EMP0305")
        self.assertEqual(self.punch_in("EMP0305", ms(2026, 7, 6, 9, 0)).status_code, 201)
        self.assertEqual(self.punch_in("EMP0305", ms(2026, 7, 6, 15, 0)).status_code, 409)   # same IST date
        self.assertEqual(self.punch_in("EMP0305", ms(2026, 7, 7, 9, 0)).status_code, 201)    # next date is fine

    def test_punch_in_uses_ist_date_not_utc(self):
        self.create_emp("EMP0306")
        # 2026-07-06 19:00 UTC == 2026-07-07 00:30 IST
        at = int(datetime(2026, 7, 6, 19, 0, tzinfo=timezone.utc).timestamp()) * 1000
        self.assertEqual(self.punch_in("EMP0306", at).json()["date"], "2026-07-07")

    def test_concurrent_punch_in_yields_exactly_one_201(self):
        self.create_emp("EMP0307")
        at = ms(2026, 7, 6, 9, 15)
        codes = sorted(r.status_code for r in self.race(10, lambda c, i: c.post("/attendance/punch-in", json={"emp_code": "EMP0307", "punched_at": at})))
        self.assertEqual(codes, [201] + [409] * 9)
        self.assertEqual(self.db.attendance_logs.count_documents({"emp_code": "EMP0307", "date": "2026-07-06"}), 1)

    # ---- punch-out -----------------------------------------------------
    def test_punch_out_sample_day(self):
        self.create_emp("EMP0401")
        self.punch_in("EMP0401", ms(2026, 7, 6, 9, 28))
        body = self.punch_out("EMP0401", ms(2026, 7, 6, 18, 35)).json()
        self.assertEqual((body["work_hours"], body["overtime_minutes"], body["half_day"], body["late_minutes"]), (9.12, 0, False, 0))
        self.assertEqual(body["history"], [])  # punch-in / punch-out never write history

    def test_overtime_threshold_and_half_day_boundary(self):
        self.create_emp("EMP0402")
        cases = [  # (day, out h:m:s) -> overtime, work_hours, half_day ; in 09:30:00 each day
            (6, (19, 0, 0), 30, 9.5, False),     # exactly 30 min past 18:30 counts
            (7, (18, 59, 59), 0, 9.5, False),    # 29:59 does not count; 9h29m59s = 9.4997 h -> 9.50
            (8, (13, 59, 41), 0, 4.49, True),    # 16181 s = 4.4947 h -> 4.49 -> half day
            (9, (13, 59, 42), 0, 4.5, False),    # 16182 s = 4.495 h -> half-up 4.50 -> not a half day
            (10, (13, 30, 0), 0, 4.0, True),
        ]
        for day, (h, mi, s), ot, hours, half in cases:
            self.assertEqual(self.punch_in("EMP0402", ms(2026, 7, day, 9, 30)).status_code, 201)
            body = self.punch_out("EMP0402", ms(2026, 7, day, h, mi, s)).json()
            self.assertEqual((body["overtime_minutes"], body["work_hours"], body["half_day"]), (ot, hours, half), day)

    def test_overnight_shift_across_midnight(self):
        self.create_emp("EMP0403", start="22:00", end="06:00")
        body = self.punch_in("EMP0403", ms(2026, 7, 6, 21, 55)).json()
        self.assertEqual((body["date"], body["late_minutes"]), ("2026-07-06", 0))
        out = self.punch_out("EMP0403", ms(2026, 7, 7, 6, 40)).json()
        self.assertEqual((out["date"], out["work_hours"], out["overtime_minutes"]), ("2026-07-06", 8.75, 40))
        # a punch-in after midnight but before shift_end belongs to the shift that started the day before
        early = self.punch_in("EMP0403", ms(2026, 7, 9, 0, 30)).json()
        self.assertEqual((early["date"], early["late_minutes"]), ("2026-07-08", 150))

    def test_punch_out_errors(self):
        self.assertEqual(self.punch_out("EMP9998", ms(2026, 7, 6, 18, 0)).status_code, 404)
        self.create_emp("EMP0404")
        self.assertEqual(self.punch_out("EMP0404", ms(2026, 7, 6, 18, 0)).status_code, 404)   # nothing to close
        self.punch_in("EMP0404", ms(2026, 7, 6, 9, 30))
        self.assertEqual(self.punch_out("EMP0404", ms(2026, 7, 6, 9, 30)).status_code, 422)   # equal to punch-in
        self.assertEqual(self.punch_out("EMP0404", ms(2026, 7, 6, 9, 0)).status_code, 404)    # earlier than any punch-in
        self.assertEqual(self.punch_out("EMP0404", ms(2026, 7, 7, 9, 30, 1)).status_code, 422)  # > 24 h
        for bad in (ms(2026, 7, 6, 18, 0) // 1000, float(ms(2026, 7, 6, 18, 0)), "x"):
            self.assertEqual(self.client.post("/attendance/punch-out", json={"emp_code": "EMP0404", "punched_at": bad}).status_code, 422)
        self.assertEqual(self.punch_out("EMP0404", ms(2026, 7, 7, 9, 30)).status_code, 200)  # exactly 24 h is allowed
        self.assertEqual(self.punch_out("EMP0404", ms(2026, 7, 7, 10, 0)).status_code, 409)   # already punched out

    def test_concurrent_punch_out_yields_exactly_one_200(self):
        self.create_emp("EMP0405")
        self.punch_in("EMP0405", ms(2026, 7, 6, 9, 30))
        results = self.race(8, lambda c, i: c.post("/attendance/punch-out", json={"emp_code": "EMP0405", "punched_at": ms(2026, 7, 6, 18, 0, i)}))
        self.assertEqual(sorted(r.status_code for r in results), [200] + [409] * 7)

    # ---- regularization ------------------------------------------------
    def patch(self, code, day, **body):
        body.setdefault("reason", "biometric glitch")
        body.setdefault("regularized_by", "hr.admin")
        return self.client.patch(f"/attendance/{code}/{day}", json=body)

    def test_regularize_recomputes_and_appends_history(self):
        self.create_emp("EMP0501")
        self.punch_in("EMP0501", ms(2026, 7, 7, 10, 5))   # 35 min after the 09:30 shift start (spec sample)
        self.punch_out("EMP0501", ms(2026, 7, 7, 19, 10))
        r = self.patch("EMP0501", "2026-07-07", punch_in=ms(2026, 7, 7, 9, 28))
        self.assertEqual(r.status_code, 200, r.text)
        rec = r.json()
        self.assertEqual((rec["late_minutes"], rec["work_hours"], rec["overtime_minutes"]), (0, 9.7, 40))
        self.assertEqual(len(rec["history"]), 1)
        entry = rec["history"][0]
        self.assertEqual((entry["by"], entry["reason"]), ("hr.admin", "biometric glitch"))
        self.assertEqual(set(entry["changes"]), {"punch_in", "late_minutes", "work_hours"})  # only what changed
        self.assertEqual(entry["changes"]["punch_in"], {"from": ms(2026, 7, 7, 10, 5), "to": ms(2026, 7, 7, 9, 28)})
        self.assertEqual(entry["changes"]["late_minutes"], {"from": 35, "to": 0})
        self.assertEqual(entry["changes"]["work_hours"], {"from": 9.08, "to": 9.7})  # 10:05->19:10 = 9h05m; 09:28->19:10 = 9h42m
        self.assertIsInstance(entry["at"], int)
        # second correction appends; the first entry is untouched
        rec2 = self.patch("EMP0501", "2026-07-07", status="WFH", reason="worked from home").json()
        self.assertEqual(len(rec2["history"]), 2)
        self.assertEqual(rec2["history"][0], entry)
        self.assertEqual(rec2["history"][1]["changes"], {"status": {"from": "PRESENT", "to": "WFH"}})
        # LEAVE clears the punch times and zeroes the derived fields
        rec3 = self.patch("EMP0501", "2026-07-07", status="LEAVE", reason="approved leave").json()
        self.assertEqual((rec3["punch_in"], rec3["punch_out"], rec3["work_hours"], rec3["overtime_minutes"], rec3["half_day"]), (None, None, None, 0, False))
        self.assertEqual(set(rec3["history"][2]["changes"]), {"status", "punch_in", "punch_out", "work_hours", "overtime_minutes"})
        # and back to a presence status needs a punch_in
        self.assertEqual(self.patch("EMP0501", "2026-07-07", status="PRESENT").status_code, 422)
        back = self.patch("EMP0501", "2026-07-07", status="PRESENT", punch_in=ms(2026, 7, 7, 9, 0))
        self.assertEqual((back.status_code, len(back.json()["history"])), (200, 4))
        # the stored derived values are what GET /attendance now returns
        listed = self.client.get("/attendance", params={"emp_code": "EMP0501"}).json()["items"][0]
        self.assertEqual(listed, back.json())

    def test_regularize_validation_and_404s(self):
        self.create_emp("EMP0502")
        self.punch_in("EMP0502", ms(2026, 7, 8, 9, 30))
        self.punch_out("EMP0502", ms(2026, 7, 8, 18, 30))
        day = "2026-07-08"
        self.assertEqual(self.patch("EMP0502", day).status_code, 422)                                   # changes nothing
        self.assertEqual(self.patch("EMP0502", day, punch_in=ms(2026, 7, 8, 9, 30)).status_code, 422)  # same value
        self.assertEqual(self.patch("EMP0502", day, status="ABSENT", punch_in=ms(2026, 7, 8, 9, 0)).status_code, 422)
        self.assertEqual(self.patch("EMP0502", day, status="LEAVE", punch_out=ms(2026, 7, 8, 9, 0)).status_code, 422)
        self.assertEqual(self.patch("EMP0502", day, punch_in=ms(2026, 7, 9, 9, 30)).status_code, 422)   # other date
        self.assertEqual(self.patch("EMP0502", day, punch_out=ms(2026, 7, 8, 9, 30)).status_code, 422)  # not after punch_in
        self.assertEqual(self.patch("EMP0502", day, punch_out=ms(2026, 7, 9, 9, 31)).status_code, 422)  # > 24 h
        self.assertEqual(self.patch("EMP0502", day, reason="abc").status_code, 422)
        self.assertEqual(self.patch("EMP0502", day, punch_in=float(ms(2026, 7, 8, 9, 0))).status_code, 422)
        self.assertEqual(self.patch("EMP0502", day, punch_in=ms(2026, 7, 8, 9, 0) // 1000).status_code, 422)
        self.assertEqual(self.patch("EMP0502", "2026-02-30", status="LEAVE").status_code, 422)
        self.assertEqual(self.client.patch(f"/attendance/EMP0502/{day}", json={"status": "LEAVE"}).status_code, 422)  # reason missing
        self.assertEqual(self.patch("EMP9998", day, status="LEAVE").status_code, 404)
        self.assertEqual(self.patch("EMP0502", "2026-07-20", status="LEAVE").status_code, 404)
        self.assertEqual(self.client.get("/attendance", params={"emp_code": "EMP0502"}).json()["items"][0]["history"], [])  # rejects leave no trace

    def test_regularize_derived_fields_in_body_are_ignored(self):
        self.create_emp("EMP0503")
        self.punch_in("EMP0503", ms(2026, 7, 8, 9, 30))
        rec = self.patch("EMP0503", "2026-07-08", status="WFH", late_minutes=77, half_day=True, work_hours=1).json()
        self.assertEqual((rec["late_minutes"], rec["half_day"], rec["work_hours"]), (0, False, None))

    def test_concurrent_regularizations_never_lose_history(self):
        self.create_emp("EMP0504")
        self.punch_in("EMP0504", ms(2026, 7, 8, 9, 30))
        self.punch_out("EMP0504", ms(2026, 7, 8, 18, 30))
        results = self.race(8, lambda c, i: c.patch("/attendance/EMP0504/2026-07-08", json={
            "punch_out": ms(2026, 7, 8, 18, 31 + i), "reason": f"edit number {i}", "regularized_by": f"hr{i}"}))
        codes = [r.status_code for r in results]
        self.assertTrue(set(codes) <= {200, 409}, codes)
        wins = codes.count(200)
        self.assertGreaterEqual(wins, 1)
        history = self.client.get("/attendance", params={"emp_code": "EMP0504"}).json()["items"][0]["history"]
        self.assertEqual(len(history), wins)  # every successful edit is recorded; none overwritten

    # ---- listing -------------------------------------------------------
    def test_attendance_listing(self):
        for code in ("EMP0601", "EMP0602"):
            self.create_emp(code)
        docs = [ref._log("EMP0602", "2026-09-07"), ref._log("EMP0601", "2026-09-07", late=15),
                ref._log("EMP0601", "2026-09-08", status="WFH"), ref._log("EMP0602", "2026-09-09", status="LEAVE"),
                ref._log("EMP0601", "2026-09-10", legacy=True)]
        self.db.attendance_logs.insert_many(copy.deepcopy(docs))

        def get(**params):
            r = self.client.get("/attendance", params=params)
            self.assertEqual(r.status_code, 200, r.text)
            return r.json()

        everything = get(date_from="2026-09-07", date_to="2026-09-10")
        self.assertEqual([(i["date"], i["emp_code"]) for i in everything["items"]],
                         [("2026-09-10", "EMP0601"), ("2026-09-09", "EMP0602"), ("2026-09-08", "EMP0601"),
                          ("2026-09-07", "EMP0601"), ("2026-09-07", "EMP0602")])  # date desc, emp_code asc
        self.assertEqual(everything["total"], 5)
        self.assertNotIn("_id", everything["items"][0])
        self.assertNotIn("id", everything["items"][0])
        legacy = everything["items"][0]
        self.assertEqual((legacy["history"], legacy["half_day"]), ([], False))  # missing fields default
        self.assertEqual(get(emp_code="EMP0601", date_from="2026-09-08", date_to="2026-09-08")["total"], 1)  # inclusive
        self.assertEqual(get(status="WFH", date_from="2026-09-07", date_to="2026-09-10")["total"], 1)
        page = get(date_from="2026-09-07", date_to="2026-09-10", page=2, page_size=2)
        self.assertEqual((page["total"], len(page["items"]), page["page"], page["page_size"]), (5, 2, 2, 2))
        self.assertEqual(get(date_from="2026-09-07", date_to="2026-09-10", page=9)["items"], [])
        for params in ({"date_from": "2026-09-10", "date_to": "2026-09-07"}, {"date_from": "2026-7-9"},
                       {"status": "HOLIDAY"}, {"page": 0}, {"page_size": 101}, {"date_to": "2026-02-30"}):
            self.assertEqual(self.client.get("/attendance", params=params).status_code, 422, params)


# =========================================================================== #
class AnalyticsTests(IntegrationBase):
    """Aggregation pipelines versus the pure-Python oracle on the deterministic dataset."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.employees, cls.logs = ref.build_dataset()
        cls.db.employees.insert_many(copy.deepcopy(cls.employees))
        cls.db.attendance_logs.insert_many(copy.deepcopy(cls.logs))

    def get(self, url, **params):
        r = self.client.get(url, params=params)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_employee_monthly_matches_oracle_for_everyone(self):
        for month in ("2026-06", "2026-07", "2026-08"):
            for emp in self.employees:
                got = self.get(f"/analytics/employees/{emp['emp_code']}/monthly", month=month)
                self.assertEqual(got, ref.ref_monthly(emp, self.logs, month), (emp["emp_code"], month))

    def test_employee_monthly_special_cases(self):
        mid = self.get("/analytics/employees/EMP1004/monthly", month="2026-07")  # joined 2026-07-15
        self.assertEqual(mid["working_days"], 13)
        none = self.get("/analytics/employees/EMP1006/monthly", month="2026-07")  # no logs at all
        self.assertEqual((none["present_days"], none["late_count"], none["attendance_pct"], none["working_days"]), (0.0, 0, 0.0, 23))
        future = self.get("/analytics/employees/EMP1005/monthly", month="2026-07")  # joins in August
        self.assertEqual((future["working_days"], future["attendance_pct"]), (0, None))
        self.assertEqual(self.client.get("/analytics/employees/EMP9999/monthly", params={"month": "2026-07"}).status_code, 404)
        for month in ("2026-13", "2026-7", "July"):
            self.assertEqual(self.client.get("/analytics/employees/EMP1001/monthly", params={"month": month}).status_code, 422)
        self.assertEqual(self.client.get("/analytics/employees/EMP1001/monthly").status_code, 422)

    def test_department_summary_matches_oracle(self):
        for month in ("2026-06", "2026-07", "2026-08", "2027-01"):
            self.assertEqual(self.get("/analytics/departments/summary", month=month)["items"],
                             ref.ref_summary(self.employees, self.logs, month), month)
        for dept in ("Engineering", "Support", "Ops", "Nope"):
            self.assertEqual(self.get("/analytics/departments/summary", month="2026-07", department=dept)["items"],
                             ref.ref_summary(self.employees, self.logs, "2026-07", dept), dept)

    def test_department_summary_headcount_rules(self):
        july = {i["department"]: i for i in self.get("/analytics/departments/summary", month="2026-07")["items"]}
        self.assertEqual(july["Engineering"]["headcount"], 5)  # includes EMP1006 (no logs), excludes EMP1005 (joins Aug)
        aug = {i["department"]: i for i in self.get("/analytics/departments/summary", month="2026-08")["items"]}
        self.assertEqual(aug["Engineering"]["headcount"], 6)
        self.assertEqual(self.get("/analytics/departments/summary", month="2025-12")["items"], [])  # nobody joined yet
        self.assertEqual(self.client.get("/analytics/departments/summary").status_code, 422)
        self.assertEqual(self.client.get("/analytics/departments/summary", params={"month": "2026-00"}).status_code, 422)

    def test_leaderboard_matches_oracle_including_ties(self):
        for limit in (1, 2, 3, 4, 10, 50):
            for dept in (None, "Engineering", "Support", "Ops"):
                params = {"month": "2026-07", "limit": limit, **({"department": dept} if dept else {})}
                self.assertEqual(self.get("/analytics/leaderboard/late", **params)["items"],
                                 ref.ref_leaderboard(self.employees, self.logs, "2026-07", limit, dept), params)

    def test_leaderboard_competition_ranking_shapes(self):
        support = self.get("/analytics/leaderboard/late", month="2026-07", department="Support", limit=50)["items"]
        self.assertEqual([(r["rank"], r["emp_code"], r["total_late_minutes"]) for r in support],
                         [(1, "EMP2001", 100), (2, "EMP2002", 50), (2, "EMP2003", 50), (4, "EMP2004", 40)])
        cut = self.get("/analytics/leaderboard/late", month="2026-07", department="Support", limit=2)["items"]
        self.assertEqual([r["emp_code"] for r in cut], ["EMP2001", "EMP2002", "EMP2003"])  # tie at the cutoff: 3 rows for limit 2
        ops = self.get("/analytics/leaderboard/late", month="2026-07", department="Ops", limit=1)["items"]
        self.assertEqual([(r["rank"], r["emp_code"]) for r in ops], [(1, "EMP3001"), (1, "EMP3002"), (1, "EMP3003")])
        everyone = self.get("/analytics/leaderboard/late", month="2026-07", limit=50)["items"]
        self.assertNotIn("EMP9999", [r["emp_code"] for r in everyone])  # orphan logs ignored
        self.assertTrue(all(r["total_late_minutes"] > 0 for r in everyone))
        for params in ({"month": "2026-07", "limit": 0}, {"month": "2026-07", "limit": 51}, {"limit": 5}, {"month": "bad"}):
            self.assertEqual(self.client.get("/analytics/leaderboard/late", params=params).status_code, 422, params)

    def test_trend_matches_oracle(self):
        ranges = [("Engineering", date(2026, 6, 22), date(2026, 7, 12)),
                  ("Engineering", date(2026, 7, 13), date(2026, 8, 9)),    # spans the August joiner
                  ("Engineering", date(2026, 1, 1), date(2026, 1, 12)),    # headcount 0 -> 3, rows with no records at all
                  ("Support", date(2026, 7, 1), date(2026, 7, 31)),
                  ("Ops", date(2026, 7, 4), date(2026, 7, 4)),             # a single weekend day
                  ("Engineering", date(2026, 5, 1), date(2026, 7, 31))]    # exactly 92 days
        for dept, start, end in ranges:
            got = self.get(f"/analytics/departments/{dept}/trend", **{"from": start.isoformat(), "to": end.isoformat()})
            self.assertEqual(got["department"], dept)
            self.assertEqual(got["items"], ref.ref_trend(self.employees, self.logs, dept, start, end), (dept, start, end))
            self.assertEqual(len(got["items"]), (end - start).days + 1)  # exactly one row per calendar day

    def test_trend_rows_for_days_without_data(self):
        rows = self.get("/analytics/departments/Ops/trend", **{"from": "2026-02-02", "to": "2026-02-08"})["items"]
        self.assertEqual([r["is_working_day"] for r in rows], [True] * 5 + [False] * 2)
        self.assertEqual([r["attendance_rate"] for r in rows], [0.0] * 5 + [None] * 2)  # no records: 0.0 on working days
        self.assertEqual([r["moving_avg_7d"] for r in rows][-1], 0.0)

    def test_trend_errors(self):
        url = "/analytics/departments/Engineering/trend"
        self.assertEqual(self.client.get("/analytics/departments/NoSuchDept/trend", params={"from": "2026-07-01", "to": "2026-07-02"}).status_code, 404)
        for params in ({"from": "2026-07-02", "to": "2026-07-01"}, {"from": "2026-05-01", "to": "2026-08-01"},  # 93 days
                       {"from": "2026-07-01"}, {"to": "2026-07-01"}, {"from": "x", "to": "2026-07-01"},
                       {"from": "2026-02-30", "to": "2026-03-01"}):
            self.assertEqual(self.client.get(url, params=params).status_code, 422, params)

    # ---- explain -------------------------------------------------------
    EXPLAIN_CASES = [
        ("attendance_list", {"date_from": "2026-07-01", "date_to": "2026-07-31", "page": 2, "page_size": 5}, "attendance_logs"),
        ("attendance_list", {"emp_code": "EMP1001"}, "attendance_logs"),
        ("attendance_list", {"status": "WFH"}, "attendance_logs"),
        ("attendance_list", {}, "attendance_logs"),
        ("employee_monthly", {"emp_code": "EMP1001", "month": "2026-07"}, "attendance_logs"),
        ("department_summary", {"month": "2026-07"}, "employees"),
        ("department_summary", {"month": "2026-07", "department": "Support"}, "employees"),
        ("late_leaderboard", {"month": "2026-07", "limit": 5, "department": "Ops"}, "attendance_logs"),
        ("late_leaderboard", {"month": "2026-07"}, "attendance_logs"),
        ("department_trend", {"department": "Engineering", "from": "2026-07-01", "to": "2026-07-31"}, "employees"),
    ]

    def test_explain_uses_indexes_and_no_collection_scan(self):
        for endpoint, params, collection in self.EXPLAIN_CASES:
            body = self.get(f"/admin/explain/{endpoint}", **params)
            self.assertEqual((body["endpoint"], body["collection"]), (endpoint, collection))
            text = json.dumps(body["explain"])
            self.assertIn("IXSCAN", text, (endpoint, params))
            self.assertNotIn("COLLSCAN", text, (endpoint, params))
            self.assertIn("executionStats", text)

    def test_explain_explains_the_real_pipeline_not_a_simplified_query(self):
        trend = json.dumps(self.get("/admin/explain/department_trend", department="Engineering", **{"from": "2026-07-01", "to": "2026-07-31"})["explain"])
        for stage in ("densify", "setwindowfields", "lookup"):  # explain shows internal names, e.g. $_internalDensify
            self.assertIn(stage, trend.lower())
        board = json.dumps(self.get("/admin/explain/late_leaderboard", month="2026-07")["explain"])
        self.assertIn("setwindowfields", board.lower())
        summary = json.dumps(self.get("/admin/explain/department_summary", month="2026-07")["explain"])
        self.assertIn("joined_on_idx", summary)

    def test_explain_validation(self):
        for endpoint, params in (("employee_monthly", {}), ("employee_monthly", {"emp_code": "EMP1001"}),
                                 ("department_summary", {}), ("late_leaderboard", {}),
                                 ("department_trend", {"department": "Engineering"}),
                                 ("department_trend", {"department": "Engineering", "from": "2026-07-02", "to": "2026-07-01"}),
                                 ("attendance_list", {"date_from": "2026-07-09", "date_to": "2026-07-01"}),
                                 ("not_an_endpoint", {})):
            self.assertEqual(self.client.get(f"/admin/explain/{endpoint}", params=params).status_code, 422, (endpoint, params))


# =========================================================================== #
@unittest.skipUnless(os.environ.get("HRONE_LARGE") == "1", "set HRONE_LARGE=1 to run the 100,000-record check")
class LargeDatasetTests(IntegrationBase):
    """Mirrors the grader: ~100k attendance records, plans must be IXSCAN-only and requests fast."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        depts = ["Engineering", "Support", "Ops", "Sales", "Finance"]
        emps = [ref._employee(f"EMP{10000 + i}", f"N{i}", depts[i % 5], "2025-01-01") for i in range(500)]
        cls.db.employees.insert_many(emps)
        start = date(2026, 1, 1)
        batch = []
        for i in range(500):
            for d in range(200):
                day = start + timedelta(days=d)
                batch.append(ref._log(f"EMP{10000 + i}", day.isoformat(), late=(i * 7 + d) % 40 if (i + d) % 3 == 0 else 0))
            if len(batch) >= 20000:
                cls.db.attendance_logs.insert_many(batch)
                batch = []
        if batch:
            cls.db.attendance_logs.insert_many(batch)

    def test_100k_plans_and_speed(self):
        self.assertEqual(self.db.attendance_logs.estimated_document_count(), 100_000)
        for endpoint, params, _ in AnalyticsTests.EXPLAIN_CASES:
            if "month" in params:
                params = {**params, "month": "2026-03"}
            if endpoint == "attendance_list" and "date_from" in params:
                params = {**params, "date_from": "2026-03-01", "date_to": "2026-03-31"}
            if endpoint == "department_trend":
                params = {"department": "Engineering", "from": "2026-03-01", "to": "2026-05-31"}
            if "emp_code" in params:
                params = {**params, "emp_code": "EMP10001"}
            text = json.dumps(self.client.get(f"/admin/explain/{endpoint}", params=params).json()["explain"])
            self.assertIn("IXSCAN", text, (endpoint, params))
            self.assertNotIn("COLLSCAN", text, (endpoint, params))
        timings = {}
        for name, url, params in (
                ("list", "/attendance", {"page": 3}),
                ("monthly", "/analytics/employees/EMP10001/monthly", {"month": "2026-03"}),
                ("summary", "/analytics/departments/summary", {"month": "2026-03"}),
                ("leaderboard", "/analytics/leaderboard/late", {"month": "2026-03"}),
                ("trend", "/analytics/departments/Engineering/trend", {"from": "2026-03-01", "to": "2026-05-31"})):
            t0 = time.perf_counter()
            self.assertEqual(self.client.get(url, params=params).status_code, 200, name)
            timings[name] = time.perf_counter() - t0
        print("\n100k timings (s):", {k: round(v, 3) for k, v in timings.items()})
        self.assertTrue(all(t < 5 for t in timings.values()), timings)


if __name__ == "__main__":
    unittest.main()
