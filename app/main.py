"""Employee Attendance & Analytics API (FastAPI + MongoDB).

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (real environment variables win over a local .env)

Layout of this single file
  1. configuration            5. request models
  2. time / number helpers    6. endpoints: system, employees, attendance
  3. business rules R1-R5     7. aggregation builders + analytics endpoints
  4. indexes and startup      8. admin explain endpoint
"""

import calendar
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
load_dotenv()  # override=False: variables already in the environment take priority

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
if not MONGO_URI or not MONGO_DB:
    raise RuntimeError("MONGO_URI and MONGO_DB must be set (environment or .env)")

# tz_aware=True: PyMongo returns aware UTC datetimes instead of naive ones.
client = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
db = client[MONGO_DB]
logger = logging.getLogger("attendance")

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

MIN_EPOCH_MS = 100_000_000_000  # anything smaller looks like seconds
MAX_EPOCH_MS = 4_102_444_800_000

PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")  # R6
NON_PRESENCE_STATUSES = ("ABSENT", "LEAVE")
ALL_STATUSES = PRESENCE_STATUSES + NON_PRESENCE_STATUSES

LATE_GRACE = timedelta(minutes=10)  # R2
MIN_OVERTIME = timedelta(minutes=30)  # R3
MAX_SHIFT_LENGTH = timedelta(hours=24)
HALF_DAY_BELOW_HOURS = 4.50  # R5
MAX_TREND_DAYS = 92
MONGO_WEEKDAYS = [2, 3, 4, 5, 6]  # $dayOfWeek: 1 = Sunday ... 7 = Saturday

ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]


# --------------------------------------------------------------------------- #
# 2. Time and number helpers (pure functions)
# --------------------------------------------------------------------------- #
def round_half_up(value: Any, places: int = 2) -> float:
    """Half-up rounding (R8). Goes through Decimal so 4.495 -> 4.50, never banker's rounding."""
    quantum = Decimal(1).scaleb(-places)
    exact = value if isinstance(value, Decimal) else Decimal(str(value))
    return float(exact.quantize(quantum, rounding=ROUND_HALF_UP))


def ms_to_datetime(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime, truncated to whole seconds (R1)."""
    return datetime.fromtimestamp(ms // 1000, tz=UTC)


def datetime_to_ms(value: Optional[datetime]) -> Optional[int]:
    """Datetime -> integer epoch milliseconds. Naive values are treated as UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return (value - EPOCH) // timedelta(milliseconds=1)


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_MONTH_RE = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])")


def parse_date(value: str) -> Date:
    """Strict YYYY-MM-DD. Raises ValueError (also for impossible dates like 2026-02-30)."""
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise ValueError("must be YYYY-MM-DD")
    return Date.fromisoformat(value)


def date_param(value: str, label: str) -> Date:
    try:
        return parse_date(value)
    except ValueError:
        raise HTTPException(422, f"{label} must be a valid YYYY-MM-DD date")


def month_range(month: str) -> tuple[Date, Date]:
    """'2026-07' -> (2026-07-01, 2026-07-31)."""
    try:
        if not _MONTH_RE.fullmatch(month):
            raise ValueError
        year, mon = int(month[:4]), int(month[5:])
        return Date(year, mon, 1), Date(year, mon, calendar.monthrange(year, mon)[1])
    except ValueError:
        raise HTTPException(422, "month must be YYYY-MM")


def working_days(joined_on: Date, first: Date, last: Date) -> int:
    """Mon-Fri days in [first, last], counted from joined_on if that is later (R7)."""
    day, total = max(joined_on, first), 0
    while day <= last:
        total += day.weekday() < 5
        day += timedelta(days=1)
    return total


# --------------------------------------------------------------------------- #
# 3. Business rules R1-R5 (pure functions; `emp` is an employees document)
# --------------------------------------------------------------------------- #
def at_ist(day: Date, hhmm: str) -> datetime:
    hour, minute = hhmm.split(":")
    return datetime(day.year, day.month, day.day, int(hour), int(minute), tzinfo=IST)


def is_overnight(emp: dict) -> bool:
    return emp["shift_end"] <= emp["shift_start"]


def attendance_date(punch_in: datetime, emp: dict) -> Date:
    """R1: IST calendar date of the punch-in; overnight shifts started the day before
    when the punch-in is earlier than shift_end."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(emp) and local.strftime("%H:%M") < emp["shift_end"]:
        day -= timedelta(days=1)
    return day


def shift_start_at(att_date: Date, emp: dict) -> datetime:
    return at_ist(att_date, emp["shift_start"])


def shift_end_at(att_date: Date, emp: dict) -> datetime:
    end = at_ist(att_date, emp["shift_end"])
    return end + timedelta(days=1) if is_overnight(emp) else end


def late_minutes(punch_in: Optional[datetime], att_date: Date, emp: dict) -> int:
    """R2: late only when strictly more than 10:00 after shift_start; then whole minutes
    since shift_start (not since the end of the grace period)."""
    if punch_in is None:
        return 0
    delta = punch_in - shift_start_at(att_date, emp)
    return delta // timedelta(minutes=1) if delta > LATE_GRACE else 0


def overtime_minutes(punch_out: Optional[datetime], att_date: Date, emp: dict) -> int:
    """R3: whole minutes past shift_end, counted only when that is at least 30."""
    if punch_out is None:
        return 0
    delta = punch_out - shift_end_at(att_date, emp)
    return delta // timedelta(minutes=1) if delta >= MIN_OVERTIME else 0


def work_hours(punch_in: Optional[datetime], punch_out: Optional[datetime]) -> Optional[float]:
    """R4: exact Decimal division, rounded half-up to 2 places. None until punch-out."""
    if punch_in is None or punch_out is None:
        return None
    microseconds = (punch_out - punch_in) // timedelta(microseconds=1)
    return round_half_up(Decimal(microseconds) / Decimal(3_600_000_000), 2)


def derive_fields(status: str, punch_in: Optional[datetime], punch_out: Optional[datetime],
                  att_date: Date, emp: dict) -> dict:
    """The four stored derived fields (R2-R5) for a record in its final state."""
    if status not in PRESENCE_STATUSES:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    hours = work_hours(punch_in, punch_out)
    return {
        "work_hours": hours,
        "late_minutes": late_minutes(punch_in, att_date, emp),
        "overtime_minutes": overtime_minutes(punch_out, att_date, emp),
        "half_day": hours is not None and hours < HALF_DAY_BELOW_HOURS,  # compares the rounded value
    }


# --------------------------------------------------------------------------- #
# 4. Serialisation, indexes and startup
# --------------------------------------------------------------------------- #
def public_employee(doc: dict) -> dict:
    out = {k: doc.get(k) for k in ("emp_code", "name", "email", "department", "shift_start", "shift_end", "joined_on")}
    out["created_at"] = datetime_to_ms(doc.get("created_at"))
    return out


def public_attendance(doc: dict) -> dict:
    """Stored document -> API shape. Fills defaults for legacy records (no history / half_day)."""
    history = []
    for entry in doc.get("history") or []:
        changes = {}
        for field, change in (entry.get("changes") or {}).items():
            convert = datetime_to_ms if field in ("punch_in", "punch_out") else (lambda v: v)
            changes[field] = {"from": convert(change.get("from")), "to": convert(change.get("to"))}
        history.append({"at": datetime_to_ms(entry.get("at")), "by": entry.get("by"),
                        "reason": entry.get("reason"), "changes": changes})
    return {
        "emp_code": doc.get("emp_code"),
        "date": doc.get("date"),
        "status": doc.get("status"),
        "punch_in": datetime_to_ms(doc.get("punch_in")),
        "punch_out": datetime_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": bool(doc.get("half_day", False)),
        "history": history,
    }


# (collection, keys, options). Which query each one serves is explained in DECISIONS.md.
INDEXES = [
    ("employees", [("emp_code", ASCENDING)], {"unique": True, "name": "uq_employee_code"}),
    ("employees", [("department", ASCENDING), ("joined_on", ASCENDING)], {"name": "department_joined"}),
    ("employees", [("joined_on", ASCENDING)], {"name": "joined_on_idx"}),
    ("attendance_logs", [("emp_code", ASCENDING), ("date", ASCENDING)],
     {"unique": True, "name": "uq_attendance_emp_date"}),
    ("attendance_logs", [("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "attendance_date_emp"}),
    ("attendance_logs", [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)],
     {"name": "attendance_status_date_emp"}),
    ("attendance_logs", [("emp_code", ASCENDING), ("punch_in", DESCENDING)], {"name": "attendance_emp_punch_in"}),
]
_indexes_ready = False


def ensure_indexes() -> None:
    """Idempotent: create_index is a no-op when the same index already exists."""
    global _indexes_ready
    ok = True
    for collection, keys, options in INDEXES:
        try:
            db[collection].create_index(keys, **options)
        except OperationFailure:  # e.g. an index with this name but other keys already exists
            ok = False
            logger.exception("could not create index %s on %s", options["name"], collection)
    _indexes_ready = ok


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError:  # MongoDB unreachable: start anyway, /health answers 503 until it is back
        logger.exception("MongoDB unavailable at startup; indexes will be created by /health when it returns")
    yield
    client.close()


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_: Request, exc: PyMongoError):
    logger.error("database error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --------------------------------------------------------------------------- #
# 5. Request models
# --------------------------------------------------------------------------- #
# strict=True rejects floats, strings and bools; the bounds reject seconds-looking and absurd values.
EpochMs = Annotated[int, Field(strict=True, ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]
HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"


class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM)
    shift_end: str = Field(default="18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def joined_on_is_a_date(cls, value: str) -> str:
        parse_date(value)
        return value

    @model_validator(mode="after")
    def shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")  # derived fields sent by a client are ignored

    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# 6. Endpoints: system, employees, attendance
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "MongoDB unavailable")
    if not _indexes_ready:
        ensure_indexes()
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)  # the unique index on emp_code is the guarantee, not a prior find
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return public_employee(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: int = Query(1, ge=1),
                   page_size: int = Query(20, ge=1, le=100)):
    query = {} if department is None else {"department": department}
    total = db.employees.count_documents(query)
    cursor = (db.employees.find(query, {"_id": 0}).sort("emp_code", ASCENDING)
              .skip((page - 1) * page_size).limit(page_size))
    return {"items": [public_employee(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    punched_at = ms_to_datetime(body.punched_at) if body.punched_at is not None else now_utc()
    att_date = attendance_date(punched_at, emp)
    doc = {
        "emp_code": body.emp_code, "date": att_date.isoformat(), "status": body.status,
        "punch_in": punched_at, "punch_out": None, "work_hours": None,
        "late_minutes": late_minutes(punched_at, att_date, emp),
        "overtime_minutes": 0, "half_day": False, "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date): exactly one concurrent caller wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return public_attendance(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    punched_at = ms_to_datetime(body.punched_at) if body.punched_at is not None else now_utc()

    # Most recent record whose punch_in is at or before punched_at (works across midnight).
    doc = db.attendance_logs.find_one({"emp_code": body.emp_code, "punch_in": {"$lte": punched_at}},
                                      sort=[("punch_in", DESCENDING)])
    if doc is None:
        raise HTTPException(404, "no punch-in found")
    if doc.get("punch_out") is not None:
        raise HTTPException(409, "record already punched out")
    if punched_at <= doc["punch_in"] or punched_at - doc["punch_in"] > MAX_SHIFT_LENGTH:
        raise HTTPException(422, "punched_at must be after punch-in and within 24 hours of it")

    fields = derive_fields(doc["status"], doc["punch_in"], punched_at, parse_date(doc["date"]), emp)
    fields.pop("late_minutes")  # fixed at punch-in; a punch-out must not rewrite the stored value
    fields["punch_out"] = punched_at
    # Conditional update: only succeeds while punch_out is still empty, so of two simultaneous
    # punch-outs exactly one matches and the other gets 409.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": doc["_id"], "punch_out": None}, {"$set": fields}, return_document=ReturnDocument.AFTER)
    if updated is None:
        raise HTTPException(409, "record already punched out")
    return public_attendance(updated)


def attendance_filter(emp_code: Optional[str], date_from: Optional[str], date_to: Optional[str],
                      status: Optional[str]) -> dict:
    """Shared by GET /attendance and its explain so both run the identical query."""
    query: dict = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    start = date_param(date_from, "date_from") if date_from is not None else None
    end = date_param(date_to, "date_to") if date_to is not None else None
    if start and end and start > end:
        raise HTTPException(422, "date_from must be on or before date_to")
    if start or end:
        query["date"] = {}
        if start:
            query["date"]["$gte"] = start.isoformat()
        if end:
            query["date"]["$lte"] = end.isoformat()
    if status is not None:
        if status not in ALL_STATUSES:
            raise HTTPException(422, f"status must be one of {', '.join(ALL_STATUSES)}")
        query["status"] = status
    return query


@app.get("/attendance")
def list_attendance(emp_code: Optional[str] = None, date_from: Optional[str] = None,
                    date_to: Optional[str] = None, status: Optional[str] = None,
                    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    query = attendance_filter(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(query)
    cursor = (db.attendance_logs.find(query, {"_id": 0}).sort(ATTENDANCE_SORT)
              .skip((page - 1) * page_size).limit(page_size))  # filter, sort and paginate in MongoDB
    return {"items": [public_attendance(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


TRACKED_FIELDS = ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day")


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    att_date = date_param(date, "date")
    emp = db.employees.find_one({"emp_code": emp_code})
    doc = db.attendance_logs.find_one({"emp_code": emp_code, "date": att_date.isoformat()}) if emp else None
    if emp is None or doc is None:
        raise HTTPException(404, "employee or attendance record not found")

    old = {"status": doc.get("status"), "punch_in": doc.get("punch_in"), "punch_out": doc.get("punch_out"),
           "work_hours": doc.get("work_hours"), "late_minutes": doc.get("late_minutes", 0),
           "overtime_minutes": doc.get("overtime_minutes", 0), "half_day": doc.get("half_day", False)}
    new = dict(old)
    if body.status is not None:
        new["status"] = body.status
    if body.punch_in is not None:
        new["punch_in"] = ms_to_datetime(body.punch_in)
    if body.punch_out is not None:
        new["punch_out"] = ms_to_datetime(body.punch_out)

    if new["status"] in NON_PRESENCE_STATUSES:
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "ABSENT / LEAVE cannot carry punch times")
        new["punch_in"] = new["punch_out"] = None
    else:
        if new["punch_in"] is None:
            raise HTTPException(422, "a presence status requires a punch_in")
        if attendance_date(new["punch_in"], emp) != att_date:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if new["punch_out"] is not None and not (
                timedelta(0) < new["punch_out"] - new["punch_in"] <= MAX_SHIFT_LENGTH):
            raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")

    new.update(derive_fields(new["status"], new["punch_in"], new["punch_out"], att_date, emp))
    changes = {k: {"from": old[k], "to": new[k]} for k in TRACKED_FIELDS if old[k] != new[k]}
    if not changes:
        raise HTTPException(422, "request does not change the record")

    entry = {"at": now_utc(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # Optimistic concurrency: the update only matches while every tracked field still has the value
    # we read. A concurrent edit changes at least one of them, so the loser gets 409 instead of
    # silently overwriting the other edit's history entry.
    guard = {"_id": doc["_id"]}
    for key in TRACKED_FIELDS:
        guard[key] = doc[key] if key in doc else {"$exists": False}
    updated = db.attendance_logs.find_one_and_update(
        guard, {"$set": {k: new[k] for k in TRACKED_FIELDS}, "$push": {"history": entry}},
        return_document=ReturnDocument.AFTER)
    if updated is None:
        raise HTTPException(409, "record was changed concurrently; retry")
    return public_attendance(updated)


# --------------------------------------------------------------------------- #
# 7. Aggregation builders and analytics endpoints
#    Each builder is used by both the endpoint and /admin/explain, so the explained
#    pipeline is exactly the executed one.
# --------------------------------------------------------------------------- #
def ref(field: str, prefix: Optional[str] = None) -> str:
    return f"${prefix}.{field}" if prefix else f"${field}"


def late_flag(prefix: Optional[str] = None) -> dict:
    """1 when late_minutes > 0 (missing counts as 0)."""
    return {"$cond": [{"$gt": [{"$ifNull": [ref("late_minutes", prefix), 0]}, 0]}, 1, 0]}


def late_value(prefix: Optional[str] = None) -> dict:
    return {"$ifNull": [ref("late_minutes", prefix), 0]}


def present_value(prefix: Optional[str] = None, weekdays_only: bool = False) -> dict:
    """1 / 0.5 / 0 per log: presence status (R6), half day = 0.5 (R5), optionally Mon-Fri only (R7)."""
    conditions: list = [{"$in": [ref("status", prefix), list(PRESENCE_STATUSES)]}]
    if weekdays_only:
        weekday = {"$dayOfWeek": {"$dateFromString": {"dateString": ref("date", prefix), "format": "%Y-%m-%d"}}}
        conditions.append({"$in": [weekday, MONGO_WEEKDAYS]})
    half = {"$eq": [{"$ifNull": [ref("half_day", prefix), False]}, True]}
    return {"$cond": [{"$and": conditions}, {"$cond": [half, 0.5, 1]}, 0]}


def logs_in_range(first: str, last: str) -> list:
    """Sub-pipeline for $lookup into attendance_logs (joined on emp_code via uq_attendance_emp_date)."""
    return [
        {"$match": {"date": {"$gte": first, "$lte": last}}},
        {"$project": {"_id": 0, "date": 1, "status": 1, "half_day": 1, "work_hours": 1, "late_minutes": 1}},
    ]


def monthly_pipeline(emp_code: str, first: Date, last: Date) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {"$group": {
            "_id": None,
            "present_days": {"$sum": present_value(weekdays_only=True)},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": late_flag()},
            "total_late_minutes": {"$sum": late_value()},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


def summary_pipeline(first: Date, last: Date, department: Optional[str]) -> list:
    """Starts from employees so people with no logs still count (R9)."""
    match: dict = {"joined_on": {"$lte": last.isoformat()}}
    if department is not None:
        match["department"] = department
    return [
        {"$match": match},
        {"$lookup": {"from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": logs_in_range(first.isoformat(), last.isoformat()), "as": "logs"}},
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},  # keep employees with no logs
        {"$group": {
            "_id": "$department",
            "emp_codes": {"$addToSet": "$emp_code"},
            "present_days": {"$sum": present_value("logs", weekdays_only=True)},
            # Mean over records (not per-employee means). Decimal128 keeps the average exact.
            "avg_work_hours": {"$avg": {"$cond": [{"$in": ["$logs.status", list(PRESENCE_STATUSES)]},
                                                  {"$toDecimal": "$logs.work_hours"}, None]}},
            "late_count": {"$sum": late_flag("logs")},
            "total_late_minutes": {"$sum": late_value("logs")},
            "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
            "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}},
        }},
        {"$project": {"_id": 0, "department": "$_id", "headcount": {"$size": "$emp_codes"},
                      "present_days": 1, "avg_work_hours": 1, "late_count": 1, "total_late_minutes": 1,
                      "leave_count": 1, "on_duty_count": 1}},
        {"$sort": {"department": 1}},
    ]


def leaderboard_pipeline(first: Date, last: Date, limit: int, department: Optional[str]) -> list:
    pipeline: list = [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"},
                    "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "employee"}},
        {"$unwind": "$employee"},  # logs whose employee does not exist are dropped here
    ]
    if department is not None:
        pipeline.append({"$match": {"employee.department": department}})  # rank inside the department only
    pipeline += [
        {"$set": {"emp_code": "$_id", "name": "$employee.name", "department": "$employee.department"}},
        # $rank is standard competition ranking: 1, 2, 2, 4 ...
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit is applied after ranking, so ties at the cutoff stay
        {"$project": {"_id": 0, "rank": 1, "emp_code": 1, "name": 1, "department": 1,
                      "total_late_minutes": 1, "late_count": 1}},
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}},
    ]
    return pipeline


def trend_pipeline(department: str, start: Date, end: Date) -> list:
    """One row per day in [start, end]. Gap filling ($densify) and the 7-day moving average
    ($setWindowFields) both run in MongoDB.

    Rates are carried as integer 'units' of 1/10000 so half-up rounding is exact: the stored
    attendance_rate is the 4-decimal value and moving_avg_7d is the mean of those reported values."""
    first, last = start.isoformat(), end.isoformat()
    # $densify bounds are [lower, upper): lower inclusive, upper EXCLUSIVE. To generate `end` itself
    # the upper bound must be the following midnight. (Naive datetimes are encoded as UTC.)
    bounds = [datetime(start.year, start.month, start.day),
              datetime(end.year, end.month, end.day) + timedelta(days=1)]
    return [
        {"$match": {"department": department}},
        {"$lookup": {"from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": logs_in_range(first, last), "as": "logs"}},
        # preserve employees without logs so the pipeline never goes empty (also seeds $densify)
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {"$project": {"_id": 0, "day": {"$ifNull": ["$logs.date", first]},
                      "present": present_value("logs"), "late": late_flag("logs")}},
        {"$group": {"_id": "$day", "present_count": {"$sum": "$present"}, "late_count": {"$sum": "$late"}}},
        {"$project": {"_id": 0, "present_count": 1, "late_count": 1,
                      "date": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}}},
        {"$densify": {"field": "date", "range": {"step": 1, "unit": "day", "bounds": bounds}}},
        {"$set": {"present_count": {"$ifNull": ["$present_count", 0]},
                  "late_count": {"$ifNull": ["$late_count", 0]},
                  "day": {"$dateToString": {"date": "$date", "format": "%Y-%m-%d"}}}},
        {"$lookup": {"from": "employees", "let": {"day": "$day"}, "as": "hc", "pipeline": [
            {"$match": {"department": department, "$expr": {"$lte": ["$joined_on", "$$day"]}}},
            {"$count": "n"}]}},
        {"$set": {"headcount": {"$ifNull": [{"$arrayElemAt": ["$hc.n", 0]}, 0]},
                  "is_working_day": {"$in": [{"$dayOfWeek": "$date"}, MONGO_WEEKDAYS]}}},
        {"$set": {"rate_units": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            {"$floor": {"$add": [{"$divide": [{"$multiply": ["$present_count", 10000]}, "$headcount"]}, 0.5]}},
            None]}}},
        {"$setWindowFields": {"sortBy": {"date": 1}, "output": {
            "ma_units": {"$avg": "$rate_units", "window": {"documents": [-6, 0]}}}}},
        {"$project": {
            "_id": 0, "date": "$day", "is_working_day": 1, "headcount": 1, "present_count": 1, "late_count": 1,
            "attendance_rate": {"$cond": [{"$eq": ["$rate_units", None]}, None, {"$divide": ["$rate_units", 10000]}]},
            "moving_avg_7d": {"$cond": [{"$eq": ["$ma_units", None]}, None,
                                        {"$divide": [{"$floor": {"$add": ["$ma_units", 0.5]}}, 10000]}]},
        }},
        {"$sort": {"date": 1}},
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(...)):
    first, last = month_range(month)
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    rows = list(db.attendance_logs.aggregate(monthly_pipeline(emp_code, first, last)))
    agg = rows[0] if rows else {}

    days = working_days(parse_date(emp["joined_on"]), first, last)
    present = Decimal(str(agg.get("present_days", 0)))
    return {
        "emp_code": emp_code, "month": month, "working_days": days,
        "present_days": round_half_up(present, 2),
        "leave_days": int(agg.get("leave_days", 0)),
        "late_count": int(agg.get("late_count", 0)),
        "total_late_minutes": int(agg.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(agg.get("total_overtime_minutes", 0)),
        # attendance_pct is a percentage, a reported number -> 2 decimals (R8); exact Decimal maths
        "attendance_pct": round_half_up(present * 100 / Decimal(days), 2) if days else None,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(...), department: Optional[str] = None):
    first, last = month_range(month)
    items = list(db.employees.aggregate(summary_pipeline(first, last, department)))
    for item in items:
        item["present_days"] = round_half_up(item["present_days"], 2)
        avg = item.get("avg_work_hours")  # Decimal128 (exact) or None when no record qualifies
        if avg is not None:
            avg = avg.to_decimal() if hasattr(avg, "to_decimal") else avg
        item["avg_work_hours"] = None if avg is None else round_half_up(avg, 2)
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: str = Query(...), limit: int = Query(10, ge=1, le=50),
                     department: Optional[str] = None):
    first, last = month_range(month)
    return {"month": month, "items": list(db.attendance_logs.aggregate(
        leaderboard_pipeline(first, last, limit, department)))}


def trend_range(from_: str, to: str) -> tuple[Date, Date]:
    start, end = date_param(from_, "from"), date_param(to, "to")
    if end < start:
        raise HTTPException(422, "to must be on or after from")
    if (end - start).days + 1 > MAX_TREND_DAYS:
        raise HTTPException(422, f"range cannot exceed {MAX_TREND_DAYS} days")
    return start, end


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    start, end = trend_range(from_, to)
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "department not found")
    items = list(db.employees.aggregate(trend_pipeline(department, start, end)))
    for item in items:
        item["present_count"] = round_half_up(item["present_count"], 2)
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 8. Admin: explain
# --------------------------------------------------------------------------- #
ExplainTarget = Literal["attendance_list", "employee_monthly", "department_summary",
                        "late_leaderboard", "department_trend"]


def _require(**values: Any) -> None:
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise HTTPException(422, f"missing required query parameter(s): {', '.join(missing)}")


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(endpoint: ExplainTarget, emp_code: Optional[str] = None, month: Optional[str] = None,
                     department: Optional[str] = None, limit: int = Query(10, ge=1, le=50),
                     date_from: Optional[str] = None, date_to: Optional[str] = None,
                     status: Optional[str] = None, page: int = Query(1, ge=1),
                     page_size: int = Query(20, ge=1, le=100),
                     from_: Optional[str] = Query(None, alias="from"), to: Optional[str] = None):
    """executionStats explain of the endpoint's real query / pipeline (built by the same functions)."""
    if endpoint == "attendance_list":
        collection = "attendance_logs"
        command: dict = {"find": collection, "projection": {"_id": 0}, "sort": dict(ATTENDANCE_SORT),
                         "filter": attendance_filter(emp_code, date_from, date_to, status),
                         "skip": (page - 1) * page_size, "limit": page_size}
    else:
        if endpoint == "employee_monthly":
            _require(emp_code=emp_code, month=month)
            collection, pipeline = "attendance_logs", monthly_pipeline(emp_code, *month_range(month))
        elif endpoint == "department_summary":
            _require(month=month)
            collection, pipeline = "employees", summary_pipeline(*month_range(month), department)
        elif endpoint == "late_leaderboard":
            _require(month=month)
            collection, pipeline = "attendance_logs", leaderboard_pipeline(*month_range(month), limit, department)
        else:  # department_trend
            _require(department=department, **{"from": from_, "to": to})
            collection, pipeline = "employees", trend_pipeline(department, *trend_range(from_, to))
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}

    explain = db.command({"explain": command, "verbosity": "executionStats"})
    return json.loads(json_util.dumps({"endpoint": endpoint, "collection": collection, "explain": explain}))
