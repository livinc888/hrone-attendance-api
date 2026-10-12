"""Unit tests for the business rules and helpers in app/main.py.

No MongoDB needed: these exercise pure functions with fixed timestamps (never the current clock).
They run with `python -m unittest` or `pytest`. They do NOT prove aggregation behaviour; that is
what tests/test_api_integration.py is for.
"""
import json
import math
import os
import random
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

# main.py reads these at import time; MongoClient is lazy, so a dummy value never connects.
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "unit_tests_never_connected")

from fastapi import HTTPException  # noqa: E402

from app import main as m  # noqa: E402
from tests import reference as ref  # noqa: E402

DAY = {"shift_start": "09:30", "shift_end": "18:30"}
NIGHT = {"shift_start": "22:00", "shift_end": "06:00"}


def ist(y, mo, d, h=0, mi=0, s=0):
    """IST wall-clock time as an aware datetime (stored/compared in UTC like the app does)."""
    return datetime(y, mo, d, h, mi, s, tzinfo=m.IST).astimezone(timezone.utc)


class TimestampTests(unittest.TestCase):
    def test_truncates_to_whole_seconds(self):
        dt = m.ms_to_datetime(1_783_312_500_999)
        self.assertEqual(dt.microsecond, 0)
        self.assertEqual(m.datetime_to_ms(dt), 1_783_312_500_000)

    def test_round_trip_keeps_exact_milliseconds_for_stored_values(self):
        stored = datetime(2026, 7, 6, 3, 58, 0, 123000, tzinfo=timezone.utc)  # seeded value with ms
        self.assertEqual(m.datetime_to_ms(stored) % 1000, 123)

    def test_naive_datetime_is_treated_as_utc(self):
        self.assertEqual(m.datetime_to_ms(datetime(1970, 1, 1, 0, 0, 1)), 1000)
        self.assertIsNone(m.datetime_to_ms(None))

    def test_allowed_range_constants_match_contract(self):
        self.assertEqual(m.MIN_EPOCH_MS, 100_000_000_000)
        self.assertEqual(m.MAX_EPOCH_MS, 4_102_444_800_000)


class DateParsingTests(unittest.TestCase):
    def test_parse_date(self):
        self.assertEqual(m.parse_date("2026-07-06"), date(2026, 7, 6))
        for bad in ("2026-02-30", "2026-7-6", "20260706", "2026-07-06T00:00", "", " 2026-07-06"):
            with self.assertRaises(ValueError, msg=bad):
                m.parse_date(bad)

    def test_month_range(self):
        self.assertEqual(m.month_range("2026-07"), (date(2026, 7, 1), date(2026, 7, 31)))
        self.assertEqual(m.month_range("2028-02")[1], date(2028, 2, 29))
        for bad in ("2026-13", "2026-00", "2026-7", "0000-05", "abcd-01", "2026-07-01"):
            with self.assertRaises(HTTPException, msg=bad) as ctx:
                m.month_range(bad)
            self.assertEqual(ctx.exception.status_code, 422)

    def test_date_param_raises_422(self):
        with self.assertRaises(HTTPException) as ctx:
            m.date_param("2026-02-30", "from")
        self.assertEqual(ctx.exception.status_code, 422)

    def test_trend_range_limits(self):
        start = date(2026, 1, 1)
        self.assertEqual(m.trend_range("2026-01-01", "2026-01-01"), (start, start))  # one day is fine
        end_92 = start + timedelta(days=91)  # exactly 92 days inclusive
        self.assertEqual(m.trend_range("2026-01-01", end_92.isoformat())[1], end_92)
        for args in (("2026-01-01", (start + timedelta(days=92)).isoformat()), ("2026-01-02", "2026-01-01")):
            with self.assertRaises(HTTPException) as ctx:
                m.trend_range(*args)
            self.assertEqual(ctx.exception.status_code, 422)


class RoundingTests(unittest.TestCase):
    def test_half_up_not_bankers_or_float(self):
        self.assertEqual(m.round_half_up(0.125, 2), 0.13)  # banker's would give 0.12
        self.assertEqual(m.round_half_up(2.675, 2), 2.68)  # float round() gives 2.67
        self.assertEqual(m.round_half_up(1.005, 2), 1.01)
        self.assertEqual(m.round_half_up(0.03125, 4), 0.0313)
        self.assertEqual(m.round_half_up(0.00005, 4), 0.0001)
        self.assertEqual(m.round_half_up(Decimal("4.495"), 2), 4.50)
        self.assertEqual(m.round_half_up(3, 2), 3.0)


class LateRuleTests(unittest.TestCase):
    """R2, shift 09:30: 09:40:00 -> 0; 09:40:01 -> 10; 10:15:59 -> 45 (from the spec)."""

    def late(self, h, mi, s, emp=DAY, day=date(2026, 7, 6)):
        return m.late_minutes(ist(2026, 7, 6, h, mi, s), day, emp)

    def test_spec_examples(self):
        self.assertEqual(self.late(9, 40, 0), 0)
        self.assertEqual(self.late(9, 40, 1), 10)
        self.assertEqual(self.late(10, 15, 59), 45)

    def test_edges(self):
        self.assertEqual(self.late(9, 30, 0), 0)
        self.assertEqual(self.late(9, 29, 59), 0)   # early is never negative
        self.assertEqual(self.late(8, 0, 0), 0)
        self.assertEqual(self.late(9, 41, 0), 11)   # measured from shift_start, not from the grace end
        self.assertEqual(self.late(9, 40, 59), 10)

    def test_no_punch_in(self):
        self.assertEqual(m.late_minutes(None, date(2026, 7, 6), DAY), 0)


class OvertimeRuleTests(unittest.TestCase):
    """R3: whole minutes past shift_end (18:30), counted only when >= 30."""

    def ot(self, h, mi, s=0, emp=DAY, day=date(2026, 7, 6), d=6):
        return m.overtime_minutes(ist(2026, 7, d, h, mi, s), day, emp)

    def test_threshold(self):
        self.assertEqual(self.ot(18, 59, 59), 0)    # 29:59 -> not counted
        self.assertEqual(self.ot(19, 0, 0), 30)     # exactly 30 -> counted
        self.assertEqual(self.ot(19, 0, 59), 30)    # floor to whole minutes
        self.assertEqual(self.ot(19, 9, 59), 39)
        self.assertEqual(self.ot(18, 30, 0), 0)
        self.assertEqual(self.ot(17, 0, 0), 0)      # left early

    def test_overnight_end_is_next_day(self):
        # sample 7: shift 22:00-06:00 started on the 6th, out 06:40 on the 7th -> 40
        self.assertEqual(m.overtime_minutes(ist(2026, 7, 7, 6, 40), date(2026, 7, 6), NIGHT), 40)
        self.assertEqual(m.overtime_minutes(ist(2026, 7, 7, 6, 29, 59), date(2026, 7, 6), NIGHT), 0)
        self.assertEqual(m.overtime_minutes(ist(2026, 7, 7, 6, 30), date(2026, 7, 6), NIGHT), 30)

    def test_no_punch_out(self):
        self.assertEqual(m.overtime_minutes(None, date(2026, 7, 6), DAY), 0)


class WorkHoursAndHalfDayTests(unittest.TestCase):
    base = ist(2026, 7, 6, 9, 30)

    def wh(self, seconds):
        return m.work_hours(self.base, self.base + timedelta(seconds=seconds))

    def fields(self, seconds):
        return m.derive_fields("PRESENT", self.base, self.base + timedelta(seconds=seconds), date(2026, 7, 6), DAY)

    def test_half_up_rounding(self):
        self.assertEqual(self.wh(16182), 4.50)   # 4.495 h -> 4.50 (half-up)
        self.assertEqual(self.wh(16181), 4.49)   # 4.49472 h
        self.assertEqual(self.wh(54), 0.02)      # 0.015 h: float round() would give 0.01
        self.assertEqual(self.wh(18), 0.01)      # 0.005 h

    def test_half_day_boundary_uses_rounded_value(self):
        self.assertTrue(self.fields(16181)["half_day"])        # 4.49 -> half day
        self.assertFalse(self.fields(16182)["half_day"])       # 4.50 -> not
        self.assertFalse(self.fields(16200)["half_day"])       # exactly 4.50 h
        self.assertTrue(self.fields(int(3.5 * 3600))["half_day"])  # sample 4

    def test_sample_documents(self):
        def hours(a, b):
            return m.work_hours(a, b)
        self.assertEqual(hours(ist(2026, 7, 6, 9, 28), ist(2026, 7, 6, 18, 35)), 9.12)   # sample 1
        self.assertEqual(hours(ist(2026, 7, 6, 10, 5), ist(2026, 7, 6, 19, 10)), 9.08)   # sample 2
        self.assertEqual(hours(ist(2026, 7, 6, 21, 55), ist(2026, 7, 7, 6, 40)), 8.75)   # sample 7

    def test_open_record(self):
        out = m.derive_fields("WFH", self.base, None, date(2026, 7, 6), DAY)
        self.assertEqual(out, {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False})

    def test_absent_and_leave_are_all_zero(self):
        for status in ("ABSENT", "LEAVE"):
            out = m.derive_fields(status, None, None, date(2026, 7, 6), DAY)
            self.assertEqual(out, {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False})

    def test_full_late_day_with_overtime(self):  # sample 2
        out = m.derive_fields("PRESENT", ist(2026, 7, 6, 10, 5), ist(2026, 7, 6, 19, 10), date(2026, 7, 6), DAY)
        self.assertEqual(out, {"work_hours": 9.08, "late_minutes": 35, "overtime_minutes": 40, "half_day": False})


class AttendanceDateTests(unittest.TestCase):
    def test_day_shift_uses_ist_calendar_date_not_utc(self):
        # 2026-07-06 19:00 UTC is already 2026-07-07 00:30 IST
        punch = datetime(2026, 7, 6, 19, 0, tzinfo=timezone.utc)
        self.assertEqual(m.attendance_date(punch, DAY), date(2026, 7, 7))

    def test_day_shift_early_morning_stays_on_same_day(self):
        self.assertEqual(m.attendance_date(ist(2026, 7, 7, 3, 0), DAY), date(2026, 7, 7))

    def test_overnight_boundaries(self):
        self.assertEqual(m.attendance_date(ist(2026, 7, 6, 21, 55), NIGHT), date(2026, 7, 6))
        self.assertEqual(m.attendance_date(ist(2026, 7, 7, 0, 30), NIGHT), date(2026, 7, 6))
        self.assertEqual(m.attendance_date(ist(2026, 7, 7, 5, 59, 59), NIGHT), date(2026, 7, 6))
        self.assertEqual(m.attendance_date(ist(2026, 7, 7, 6, 0, 0), NIGHT), date(2026, 7, 7))  # not earlier than end

    def test_overnight_late_minutes_follow_the_started_shift(self):
        d = date(2026, 7, 6)
        self.assertEqual(m.late_minutes(ist(2026, 7, 6, 22, 10), d, NIGHT), 0)
        self.assertEqual(m.late_minutes(ist(2026, 7, 6, 22, 11), d, NIGHT), 11)
        self.assertEqual(m.late_minutes(ist(2026, 7, 7, 0, 30), d, NIGHT), 150)

    def test_shift_end_instants(self):
        self.assertEqual(m.shift_end_at(date(2026, 7, 6), DAY), ist(2026, 7, 6, 18, 30))
        self.assertEqual(m.shift_end_at(date(2026, 7, 6), NIGHT), ist(2026, 7, 7, 6, 0))


class WorkingDaysTests(unittest.TestCase):
    JULY = (date(2026, 7, 1), date(2026, 7, 31))

    def test_full_month(self):
        self.assertEqual(date(2026, 7, 1).weekday(), 2)  # Wednesday
        self.assertEqual(m.working_days(date(2026, 1, 5), *self.JULY), 23)
        self.assertEqual(m.working_days(date(2026, 2, 1), date(2026, 2, 1), date(2026, 2, 28)), 20)

    def test_join_date_boundaries(self):
        self.assertEqual(m.working_days(date(2026, 7, 1), *self.JULY), 23)    # joined on the first
        self.assertEqual(m.working_days(date(2026, 7, 15), *self.JULY), 13)   # mid-month (Wed)
        self.assertEqual(m.working_days(date(2026, 7, 18), *self.JULY), 10)   # joined on a Saturday
        self.assertEqual(m.working_days(date(2026, 7, 31), *self.JULY), 1)    # last day (Friday)
        self.assertEqual(m.working_days(date(2026, 8, 1), *self.JULY), 0)     # joins after the month


class SerialisationTests(unittest.TestCase):
    def test_legacy_record_defaults(self):
        doc = {"emp_code": "EMP0002", "date": "2026-07-14", "status": "PRESENT",
               "punch_in": datetime(2026, 7, 14, 4, 2, tzinfo=timezone.utc),
               "punch_out": datetime(2026, 7, 14, 13, 1, tzinfo=timezone.utc),
               "work_hours": 8.98, "late_minutes": 0, "overtime_minutes": 0}
        out = m.public_attendance(doc)
        self.assertEqual(out["history"], [])
        self.assertIs(out["half_day"], False)
        self.assertEqual(out["punch_in"], 1_784_001_720_000)
        self.assertNotIn("_id", out)
        self.assertNotIn("id", out)

    def test_history_instants_become_epoch_ms(self):
        t_old = datetime(2026, 7, 7, 4, 35, tzinfo=timezone.utc)
        t_new = datetime(2026, 7, 7, 3, 58, tzinfo=timezone.utc)
        doc = {"emp_code": "EMP0001", "date": "2026-07-07", "status": "PRESENT", "punch_in": t_new,
               "punch_out": None, "work_hours": None, "history": [
                   {"at": datetime(2026, 7, 8, 5, 0, tzinfo=timezone.utc), "by": "hr", "reason": "glitch",
                    "changes": {"punch_in": {"from": t_old, "to": t_new}, "late_minutes": {"from": 35, "to": 0}}}]}
        entry = m.public_attendance(doc)["history"][0]
        self.assertEqual(entry["changes"]["punch_in"], {"from": m.datetime_to_ms(t_old), "to": m.datetime_to_ms(t_new)})
        self.assertEqual(entry["changes"]["late_minutes"], {"from": 35, "to": 0})
        self.assertIsInstance(entry["at"], int)

    def test_employee_hides_internal_id(self):
        out = m.public_employee({"_id": "x", "emp_code": "EMP0001", "name": "A", "email": "a@b.co",
                                 "department": "Eng", "shift_start": "09:30", "shift_end": "18:30",
                                 "joined_on": "2026-01-05", "created_at": datetime(2026, 1, 5, 4, 30, tzinfo=timezone.utc)})
        self.assertNotIn("_id", out)
        self.assertIsInstance(out["created_at"], int)


class AttendanceFilterTests(unittest.TestCase):
    def test_builds_expected_query(self):
        q = m.attendance_filter("EMP0001", "2026-07-01", "2026-07-31", "WFH")
        self.assertEqual(q, {"emp_code": "EMP0001", "date": {"$gte": "2026-07-01", "$lte": "2026-07-31"}, "status": "WFH"})
        self.assertEqual(m.attendance_filter(None, None, None, None), {})
        self.assertEqual(m.attendance_filter(None, "2026-07-01", None, None), {"date": {"$gte": "2026-07-01"}})

    def test_rejects_bad_input_with_422(self):
        for args in (("a", "2026-07-31", "2026-07-01", None),   # date_from > date_to
                     (None, "2026-7-1", None, None),
                     (None, None, "2026-02-30", None),
                     (None, None, None, "HOLIDAY")):
            with self.assertRaises(HTTPException, msg=str(args)) as ctx:
                m.attendance_filter(*args)
            self.assertEqual(ctx.exception.status_code, 422)

    def test_equal_dates_are_allowed_and_inclusive(self):
        q = m.attendance_filter(None, "2026-07-06", "2026-07-06", None)
        self.assertEqual(q["date"], {"$gte": "2026-07-06", "$lte": "2026-07-06"})


class PipelineShapeTests(unittest.TestCase):
    """Structure only (no database): guards the design decisions the assignment asks for."""

    first, last = date(2026, 7, 1), date(2026, 7, 31)

    def stages(self, pipeline):
        return [next(iter(s)) for s in pipeline]

    def test_pipelines_are_json_serialisable_apart_from_dates(self):
        for p in (m.monthly_pipeline("EMP0001", self.first, self.last),
                  m.summary_pipeline(self.first, self.last, None),
                  m.leaderboard_pipeline(self.first, self.last, 10, "Eng"),
                  m.trend_pipeline("Eng", self.first, self.last)):
            json.dumps(p, default=str)

    def test_trend_runs_in_the_database(self):
        p = m.trend_pipeline("Eng", self.first, self.last)
        names = self.stages(p)
        self.assertIn("$densify", names)
        self.assertIn("$setWindowFields", names)
        self.assertLess(names.index("$densify"), names.index("$setWindowFields"))
        self.assertEqual(p[0], {"$match": {"department": "Eng"}})  # starts on an indexed employees filter
        step = next(s for s in p if "$densify" in s)["$densify"]["range"]
        self.assertEqual((step["step"], step["unit"]), (1, "day"))
        # upper bound is exclusive in $densify, so it must be the midnight AFTER the last requested day
        self.assertEqual(step["bounds"], [datetime(2026, 7, 1), datetime(2026, 8, 1)])
        one_day = next(s for s in m.trend_pipeline("Eng", date(2026, 2, 28), date(2026, 2, 28)) if "$densify" in s)
        self.assertEqual(one_day["$densify"]["range"]["bounds"], [datetime(2026, 2, 28), datetime(2026, 3, 1)])
        window = next(s for s in p if "$setWindowFields" in s)["$setWindowFields"]["output"]["ma_units"]["window"]
        self.assertEqual(window, {"documents": [-6, 0]})

    def test_summary_starts_from_employees_and_keeps_empty_ones(self):
        p = m.summary_pipeline(self.first, self.last, "Eng")
        self.assertEqual(p[0], {"$match": {"joined_on": {"$lte": "2026-07-31"}, "department": "Eng"}})
        unwind = next(s for s in p if "$unwind" in s)["$unwind"]
        self.assertTrue(unwind["preserveNullAndEmptyArrays"])
        self.assertEqual(m.summary_pipeline(self.first, self.last, None)[0],
                         {"$match": {"joined_on": {"$lte": "2026-07-31"}}})

    def test_leaderboard_ranks_then_limits(self):
        p = m.leaderboard_pipeline(self.first, self.last, 3, None)
        names = self.stages(p)
        rank_at = names.index("$setWindowFields")
        self.assertEqual(p[rank_at]["$setWindowFields"]["output"]["rank"], {"$rank": {}})
        limit_at = next(i for i, s in enumerate(p) if s.get("$match") == {"rank": {"$lte": 3}})
        self.assertGreater(limit_at, rank_at)  # limit applied after ranking, never $limit

    def test_leaderboard_department_filter_precedes_ranking(self):
        p = m.leaderboard_pipeline(self.first, self.last, 3, "Eng")
        dept_at = next(i for i, s in enumerate(p) if s.get("$match") == {"employee.department": "Eng"})
        self.assertLess(dept_at, self.stages(p).index("$setWindowFields"))

    def test_monthly_matches_on_indexed_prefix(self):
        p = m.monthly_pipeline("EMP0001", self.first, self.last)
        self.assertEqual(p[0]["$match"]["emp_code"], "EMP0001")
        self.assertEqual(p[0]["$match"]["date"], {"$gte": "2026-07-01", "$lte": "2026-07-31"})

    def test_every_index_is_declared_once_with_unique_names(self):
        names = [opts["name"] for _, _, opts in m.INDEXES]
        self.assertEqual(len(names), len(set(names)))
        unique = {opts["name"] for _, _, opts in m.INDEXES if opts.get("unique")}
        self.assertEqual(unique, {"uq_employee_code", "uq_attendance_emp_date"})


class RateArithmeticTests(unittest.TestCase):
    """The trend pipeline rounds with floor(x*10000 + 0.5) on IEEE doubles. This replays exactly that
    arithmetic in Python and checks it equals Decimal half-up for every realistic input."""

    def test_rate_units_match_decimal_half_up(self):
        for headcount in range(1, 301):
            for halves in range(0, 2 * headcount + 1):  # present_count in steps of 0.5
                present = halves / 2
                units = math.floor(present * 10000 / headcount + 0.5)
                exact = Decimal(halves) / 2 / headcount
                expected = int((exact * 10000).quantize(Decimal(1), rounding="ROUND_HALF_UP"))
                self.assertEqual(units, expected, (present, headcount))

    def test_moving_average_units_match_decimal_half_up(self):
        rng = random.Random(7)
        for _ in range(20000):
            window = [rng.randint(0, 10000) for _ in range(rng.randint(1, 7))]
            got = math.floor(sum(window) / len(window) + 0.5)
            expected = int((Decimal(sum(window)) / len(window)).quantize(Decimal(1), rounding="ROUND_HALF_UP"))
            self.assertEqual(got, expected, window)

    def test_a_known_tie(self):
        self.assertEqual(math.floor(1 * 10000 / 32 + 0.5) / 10000, 0.0313)  # 1/32 = 0.03125 -> 0.0313


class ReferenceOracleTests(unittest.TestCase):
    """The pure-Python oracle used by the integration tests must itself be right, so it is checked here
    against hand-computed answers."""

    def test_leaderboard_competition_ranking_and_limit_after_ranking(self):
        emps = [{"emp_code": c, "name": c, "department": "D", "joined_on": "2026-01-01"} for c in ("EMP0001", "EMP0002", "EMP0003", "EMP0004", "EMP0005")]
        late = {"EMP0001": 100, "EMP0002": 50, "EMP0003": 50, "EMP0004": 40, "EMP0005": 0}
        logs = [{"emp_code": c, "date": "2026-07-01", "status": "PRESENT", "late_minutes": v} for c, v in late.items()]
        logs.append({"emp_code": "EMP9999", "date": "2026-07-01", "status": "PRESENT", "late_minutes": 999})  # orphan
        full = ref.ref_leaderboard(emps, logs, "2026-07", 50)
        self.assertEqual([(r["rank"], r["emp_code"]) for r in full],
                         [(1, "EMP0001"), (2, "EMP0002"), (2, "EMP0003"), (4, "EMP0004")])
        # a tie at the cutoff returns BOTH tied rows, i.e. more rows than the limit
        self.assertEqual([r["emp_code"] for r in ref.ref_leaderboard(emps, logs, "2026-07", 2)],
                         ["EMP0001", "EMP0002", "EMP0003"])
        self.assertEqual([r["emp_code"] for r in ref.ref_leaderboard(emps, logs, "2026-07", 3)],
                         ["EMP0001", "EMP0002", "EMP0003"])  # rank 3 does not exist, rank 4 > limit
        self.assertEqual([r["emp_code"] for r in ref.ref_leaderboard(emps, logs, "2026-07", 4)][-1], "EMP0004")

    def test_trend_gap_filling_weekends_and_window(self):
        emps = [{"emp_code": "EMP0001", "department": "D", "joined_on": "2026-06-01"},
                {"emp_code": "EMP0002", "department": "D", "joined_on": "2026-07-08"}]
        logs = [{"emp_code": "EMP0001", "date": "2026-07-06", "status": "PRESENT", "late_minutes": 20},
                {"emp_code": "EMP0001", "date": "2026-07-07", "status": "WFH", "half_day": True},
                {"emp_code": "EMP0001", "date": "2026-07-11", "status": "PRESENT"}]  # Saturday record
        rows = ref.ref_trend(emps, logs, "D", date(2026, 7, 4), date(2026, 7, 13))
        self.assertEqual([r["date"] for r in rows], [f"2026-07-{d:02d}" for d in range(4, 14)])  # every day
        by = {r["date"]: r for r in rows}
        self.assertFalse(by["2026-07-04"]["is_working_day"])           # Saturday
        self.assertIsNone(by["2026-07-04"]["attendance_rate"])
        self.assertEqual(by["2026-07-06"]["attendance_rate"], 1.0)     # 1 of 1
        self.assertEqual(by["2026-07-06"]["late_count"], 1)
        self.assertEqual(by["2026-07-07"]["attendance_rate"], 0.5)     # half day
        self.assertEqual(by["2026-07-08"]["headcount"], 2)             # second employee joins
        self.assertEqual(by["2026-07-08"]["attendance_rate"], 0.0)     # working day, nobody recorded
        self.assertEqual(by["2026-07-11"]["present_count"], 1.0)       # weekend records still counted
        self.assertIsNone(by["2026-07-11"]["attendance_rate"])
        self.assertEqual(by["2026-07-04"]["moving_avg_7d"], None)      # no non-null rate yet
        self.assertEqual(by["2026-07-06"]["moving_avg_7d"], 1.0)
        self.assertEqual(by["2026-07-07"]["moving_avg_7d"], 0.75)      # mean(1.0, 0.5)
        self.assertEqual(by["2026-07-08"]["moving_avg_7d"], 0.5)       # mean(1.0, 0.5, 0.0)

    def test_monthly_mid_month_joiner_and_weekend_records(self):
        emp = {"emp_code": "EMP0001", "joined_on": "2026-07-15"}
        logs = [{"emp_code": "EMP0001", "date": "2026-07-15", "status": "PRESENT", "late_minutes": 15, "overtime_minutes": 40},
                {"emp_code": "EMP0001", "date": "2026-07-16", "status": "WFH", "half_day": True},
                {"emp_code": "EMP0001", "date": "2026-07-18", "status": "PRESENT", "late_minutes": 30, "overtime_minutes": 35},  # Saturday
                {"emp_code": "EMP0001", "date": "2026-07-17", "status": "LEAVE"}]
        out = ref.ref_monthly(emp, logs, "2026-07")
        self.assertEqual(out["working_days"], 13)
        self.assertEqual(out["present_days"], 1.5)               # 1 + 0.5, Saturday excluded
        self.assertEqual(out["leave_days"], 1)
        self.assertEqual((out["late_count"], out["total_late_minutes"], out["total_overtime_minutes"]), (2, 45, 75))
        self.assertEqual(out["attendance_pct"], 11.54)           # 1.5 / 13 * 100 = 11.538...

    def test_summary_counts_people_without_logs_and_excludes_future_joiners(self):
        emps = [{"emp_code": "EMP0001", "department": "D", "joined_on": "2026-01-01"},
                {"emp_code": "EMP0002", "department": "D", "joined_on": "2026-01-01"},   # never has logs
                {"emp_code": "EMP0003", "department": "D", "joined_on": "2026-08-01"},   # joins after July
                {"emp_code": "EMP0004", "department": "Z", "joined_on": "2026-09-01"}]   # whole dept not yet joined
        logs = [{"emp_code": "EMP0001", "date": "2026-07-01", "status": "PRESENT", "work_hours": 8.0},
                {"emp_code": "EMP0001", "date": "2026-07-02", "status": "PRESENT", "work_hours": 9.0, "half_day": False},
                {"emp_code": "EMP0001", "date": "2026-07-03", "status": "ON_DUTY", "work_hours": 7.0},
                {"emp_code": "EMP0001", "date": "2026-07-04", "status": "LEAVE"}]
        out = ref.ref_summary(emps, logs, "2026-07")
        self.assertEqual(len(out), 1)  # department Z omitted (headcount 0)
        self.assertEqual(out[0]["headcount"], 2)
        self.assertEqual(out[0]["avg_work_hours"], 8.0)  # mean over records, not people
        self.assertEqual((out[0]["leave_count"], out[0]["on_duty_count"], out[0]["present_days"]), (1, 1, 3.0))


if __name__ == "__main__":
    unittest.main()
