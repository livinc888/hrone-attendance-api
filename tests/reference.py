"""Pure-Python reference ("oracle") for the analytics rules, plus a deterministic dataset builder.

Deliberately naive: plain loops over lists, Decimal for rounding, no MongoDB. The integration tests
compare the real aggregation pipelines against these functions. It reads the *stored* derived fields
(late_minutes, half_day, work_hours ...) exactly like the API must.
"""
import calendar
import random
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

PRESENCE = {"PRESENT", "WFH", "ON_DUTY"}
IST = timezone(timedelta(hours=5, minutes=30))


def ms(y, mo, d, h=0, mi=0, s=0):
    """Epoch milliseconds of an IST wall-clock time."""
    return int(datetime(y, mo, d, h, mi, s, tzinfo=IST).timestamp()) * 1000


def q(value, places):
    value = value if isinstance(value, Decimal) else Decimal(str(value))
    return float(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def present_value(log):
    if log.get("status") not in PRESENCE:
        return Decimal(0)
    return Decimal("0.5") if log.get("half_day") else Decimal(1)


def is_weekday(iso):
    return date.fromisoformat(iso).weekday() < 5


def month_bounds(month):
    y, m = map(int, month.split("-"))
    return date(y, m, 1).isoformat(), date(y, m, calendar.monthrange(y, m)[1]).isoformat()


def ref_monthly(emp, logs, month):
    first, last = month_bounds(month)
    mine = [l for l in logs if l["emp_code"] == emp["emp_code"] and first <= l["date"] <= last]
    start = max(date.fromisoformat(emp["joined_on"]), date.fromisoformat(first))
    end = date.fromisoformat(last)
    days = 0
    d = start
    while d <= end:
        days += d.weekday() < 5
        d += timedelta(days=1)
    present = sum((present_value(l) for l in mine if is_weekday(l["date"])), Decimal(0))
    return {
        "emp_code": emp["emp_code"], "month": month, "working_days": days,
        "present_days": q(present, 2),
        "leave_days": sum(1 for l in mine if l["status"] == "LEAVE"),
        "late_count": sum(1 for l in mine if l.get("late_minutes", 0) > 0),
        "total_late_minutes": sum(l.get("late_minutes", 0) for l in mine),
        "total_overtime_minutes": sum(l.get("overtime_minutes", 0) for l in mine),
        "attendance_pct": q(present * 100 / days, 2) if days else None,
    }


def ref_summary(employees, logs, month, department=None):
    first, last = month_bounds(month)
    out = []
    for dept in sorted({e["department"] for e in employees}):
        if department is not None and dept != department:
            continue
        members = [e for e in employees if e["department"] == dept and e["joined_on"] <= last]
        if not members:
            continue
        codes = {e["emp_code"] for e in members}
        recs = [l for l in logs if l["emp_code"] in codes and first <= l["date"] <= last]
        hours = [Decimal(str(l["work_hours"])) for l in recs
                 if l["status"] in PRESENCE and l.get("work_hours") is not None]
        out.append({
            "department": dept, "headcount": len(members),
            "present_days": q(sum((present_value(l) for l in recs if is_weekday(l["date"])), Decimal(0)), 2),
            "avg_work_hours": q(sum(hours) / len(hours), 2) if hours else None,
            "late_count": sum(1 for l in recs if l.get("late_minutes", 0) > 0),
            "total_late_minutes": sum(l.get("late_minutes", 0) for l in recs),
            "leave_count": sum(1 for l in recs if l["status"] == "LEAVE"),
            "on_duty_count": sum(1 for l in recs if l["status"] == "ON_DUTY"),
        })
    return out


def ref_leaderboard(employees, logs, month, limit, department=None):
    first, last = month_bounds(month)
    by_code = {e["emp_code"]: e for e in employees}
    totals = {}
    for l in logs:
        if l["emp_code"] in by_code and first <= l["date"] <= last and l.get("late_minutes", 0) > 0:
            t = totals.setdefault(l["emp_code"], [0, 0])
            t[0] += l["late_minutes"]
            t[1] += 1
    rows = [(c, t[0], t[1]) for c, t in totals.items()
            if department is None or by_code[c]["department"] == department]
    rows.sort(key=lambda r: (-r[1], r[0]))
    out = []
    for code, total, count in rows:
        rank = 1 + sum(1 for r in rows if r[1] > total)  # standard competition ranking: 1, 2, 2, 4
        if rank <= limit:
            out.append({"rank": rank, "emp_code": code, "name": by_code[code]["name"],
                        "department": by_code[code]["department"],
                        "total_late_minutes": total, "late_count": count})
    return out


def ref_trend(employees, logs, department, start, end):
    members = [e for e in employees if e["department"] == department]
    codes = {e["emp_code"] for e in members}
    rows = []
    day = start
    while day <= end:
        iso = day.isoformat()
        headcount = sum(1 for e in members if e["joined_on"] <= iso)
        todays = [l for l in logs if l["emp_code"] in codes and l["date"] == iso]
        present = sum((present_value(l) for l in todays), Decimal(0))
        working = day.weekday() < 5
        rows.append({"date": iso, "is_working_day": working, "headcount": headcount,
                     "present_count": q(present, 2),
                     "late_count": sum(1 for l in todays if l.get("late_minutes", 0) > 0),
                     "attendance_rate": q(present / headcount, 4) if working and headcount > 0 else None})
        day += timedelta(days=1)
    for i, row in enumerate(rows):  # mean of the reported (4-decimal) rates inside the requested range
        window = [Decimal(str(r["attendance_rate"])) for r in rows[max(0, i - 6):i + 1]
                  if r["attendance_rate"] is not None]
        row["moving_avg_7d"] = q(sum(window) / len(window), 4) if window else None
    return rows


# --------------------------------------------------------------------------- #
# Deterministic dataset with the awkward cases from the assignment
# --------------------------------------------------------------------------- #
def _employee(code, name, dept, joined, start="09:30", end="18:30"):
    return {"emp_code": code, "name": name, "email": f"{code.lower()}@example.com", "department": dept,
            "shift_start": start, "shift_end": end, "joined_on": joined,
            "created_at": datetime(2026, 1, 5, 4, 30, tzinfo=timezone.utc)}


def _log(code, day, status="PRESENT", late=0, overtime=0, hours=8.0, half_day=False, legacy=False):
    on_floor = status in PRESENCE
    pin = datetime.fromisoformat(day).replace(hour=4, tzinfo=timezone.utc) if on_floor else None
    doc = {"emp_code": code, "date": day, "status": status, "punch_in": pin,
           "punch_out": pin + timedelta(hours=hours) if pin and hours is not None else None,
           "work_hours": hours if on_floor else None, "late_minutes": late if on_floor else 0,
           "overtime_minutes": overtime if on_floor else 0, "half_day": half_day, "history": []}
    if legacy:  # older records omit these two fields entirely
        del doc["half_day"], doc["history"]
    return doc


def build_dataset(seed=20261011):
    """Employees + logs for 2026-06-20 .. 2026-07-31. Contains: an employee with no logs, a mid-month
    joiner, a joiner after July, an overnight-shift department, weekend records, missing weekdays,
    legacy records, orphan logs, and late-minute ties (1,2,2,4 in Support; 1,1,1 in Ops)."""
    rng = random.Random(seed)
    employees = [
        _employee("EMP1001", "Asha Rao", "Engineering", "2026-01-05"),
        _employee("EMP1002", "Bala K", "Engineering", "2026-01-05"),
        _employee("EMP1003", "Chitra S", "Engineering", "2026-03-02"),
        _employee("EMP1004", "Dev P", "Engineering", "2026-07-15"),   # mid-month joiner
        _employee("EMP1005", "Esha M", "Engineering", "2026-08-03"),  # joins after July
        _employee("EMP1006", "Farid A", "Engineering", "2026-01-05"),  # never has any log
        _employee("EMP2001", "Gita N", "Support", "2026-01-05", "22:00", "06:00"),
        _employee("EMP2002", "Hari V", "Support", "2026-01-05", "22:00", "06:00"),
        _employee("EMP2003", "Indu R", "Support", "2026-01-05"),
        _employee("EMP2004", "Jay T", "Support", "2026-01-05"),
        _employee("EMP3001", "Kavi L", "Ops", "2026-01-05"),
        _employee("EMP3002", "Lata D", "Ops", "2026-01-05"),
        _employee("EMP3003", "Manu B", "Ops", "2026-01-05"),
    ]
    scripted = {"EMP2001", "EMP2002", "EMP2003", "EMP2004", "EMP3001", "EMP3002", "EMP3003", "EMP1006"}
    logs = []
    start, end = date(2026, 6, 20), date(2026, 7, 31)
    for emp in employees:
        if emp["emp_code"] in scripted:
            continue
        day = start
        while day <= end:
            iso = day.isoformat()
            roll = rng.random()
            weekend = day.weekday() >= 5
            if iso >= emp["joined_on"] and roll < (0.08 if weekend else 0.9):  # else: no record at all
                status = rng.choices(["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"], [70, 12, 6, 6, 6])[0]
                half = rng.random() < 0.1
                logs.append(_log(emp["emp_code"], iso, status,
                                 late=rng.choice([0, 0, 0, 11, 20, 35, 45]),
                                 overtime=rng.choice([0, 0, 0, 30, 45]),
                                 hours=round(rng.uniform(3.0, 10.0), 2) if not half else round(rng.uniform(2.0, 4.4), 2),
                                 half_day=half, legacy=rng.random() < 0.1))
            day += timedelta(days=1)
    # Support: totals 100, 50, 50, 40  -> ranks 1, 2, 2, 4
    logs += [_log("EMP2001", "2026-07-01", late=60), _log("EMP2001", "2026-07-02", late=40),
             _log("EMP2002", "2026-07-01", late=50), _log("EMP2003", "2026-07-01", late=20),
             _log("EMP2003", "2026-07-02", late=30), _log("EMP2004", "2026-07-03", late=40),
             _log("EMP2004", "2026-06-30", late=500)]  # outside July: must not count in July
    # Ops: three-way tie at 30 -> ranks 1, 1, 1
    logs += [_log("EMP3001", "2026-07-01", late=30), _log("EMP3002", "2026-07-02", late=30),
             _log("EMP3003", "2026-07-03", late=10), _log("EMP3003", "2026-07-06", late=20)]
    # weekend records contribute to late/overtime but not to present_days
    logs += [_log("EMP3001", "2026-07-04", late=0, overtime=60), _log("EMP3002", "2026-07-05", status="WFH")]
    # logs of an employee that does not exist must be ignored everywhere
    logs += [_log("EMP9999", "2026-07-01", late=999), _log("EMP9999", "2026-07-02", late=999)]
    return employees, logs
