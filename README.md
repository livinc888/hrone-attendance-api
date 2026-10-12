# Employee Attendance & Analytics API

FastAPI + MongoDB service for the HROne engineering assignment: employees punch in and out, managers regularize records, and analytics endpoints (monthly summary, department summary, late leaderboard, department trend) are MongoDB aggregation pipelines. All application code is in `app/main.py`.

## Prerequisites

- Python 3.11 or newer
- MongoDB 6.0 or newer (local, Docker, or an Atlas cluster). Needed for `$densify`, `$setWindowFields` and `$lookup` with `localField` plus `pipeline`.

Local MongoDB with Docker: `docker run -d --name mongo -p 27017:27017 mongo:7`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Two environment variables, read at startup. Real environment variables take priority over a local `.env` file (which is git-ignored and must never be committed).

```bash
# .env  (example with placeholders only)
MONGO_URI=mongodb://localhost:27017            # or mongodb+srv://<user>:<password>@<cluster-host>/
MONGO_DB=<your_database_name>
```

Nothing else is configurable and no connection details are hard-coded. The app refuses to start if either variable is missing.

## Run

```bash
uvicorn app.main:app --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Readiness: `GET /health` returns 200 only when MongoDB answers a ping, otherwise 503.

**Indexes** are created automatically at startup (idempotent, safe to restart; no manual step). They are listed in `INDEXES` in `app/main.py` and explained in `DECISIONS.md`. Collections are `employees` and `attendance_logs`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_rules_unit.py -v          # no database needed
TEST_MONGO_URI="mongodb://localhost:27017" python -m pytest tests -v      # adds the integration tests
HRONE_LARGE=1 TEST_MONGO_URI="..." python -m pytest tests -v -k 100k     # optional 100,000-record check
```

- `tests/test_rules_unit.py`: business rules R1-R5 with fixed timestamps (late exactly 10:00 vs 10:01, overtime 29:59 vs 30:00, work hours 4.49 vs 4.50, overnight shifts, join-date and weekend working days), rounding, serialisation, and the shape of each pipeline. It does not touch a database.
- `tests/test_api_integration.py`: every endpoint against a **real** MongoDB: validation and status codes, duplicate and simultaneous requests, regularization and history, pagination and sorting, analytics compared with a pure-Python reference (`tests/reference.py`) on a deterministic dataset with ties, no-log employees, mid-month and future joiners, weekend records and orphan logs, and explain plans (IXSCAN, no COLLSCAN). Each run uses a throw-away database and drops it afterwards; it never touches `MONGO_DB`. Skipped when `TEST_MONGO_URI` is unset.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | 200 `{"status":"ok"}` or 503 |
| POST | `/employees` | 201 / 409 duplicate `emp_code` / 422 |
| GET | `/employees` | `department`, `page`, `page_size`; sorted by `emp_code` |
| POST | `/attendance/punch-in` | creates the day's record (R1, R2); race-safe via a unique index |
| POST | `/attendance/punch-out` | work hours, overtime, half day (R3-R5); works across midnight |
| GET | `/attendance` | filters `emp_code`, `date_from`, `date_to`, `status`; sorted by date desc, `emp_code` asc |
| PATCH | `/attendance/{emp_code}/{date}` | regularize: recompute derived fields, append to `history`; 409 for the loser of a concurrent edit |
| GET | `/analytics/employees/{emp_code}/monthly` | `month=YYYY-MM` |
| GET | `/analytics/departments/summary` | `month`, optional `department` |
| GET | `/analytics/leaderboard/late` | `month`, `limit` (1-50), optional `department`; competition ranking |
| GET | `/analytics/departments/{department}/trend` | `from`, `to` (max 92 days); one row per day |
| GET | `/admin/explain/{endpoint}` | `executionStats` explain of the endpoint's real query |

## Design notes and assumptions

- **Concurrency:** duplicate punch-ins and duplicate employees are stopped by unique indexes, not by an existence check. Punch-out is a conditional update (`punch_out` must still be empty). Regularization is optimistic: the update only matches if the record still has the values that were read, so a concurrent edit gets 409 and no history entry is lost.
- **Analytics read the stored derived fields** (`late_minutes`, `half_day`, `work_hours`, ...); nothing is recomputed from punch times.
- **Interpretations of the spec where it is not explicit** (worth a second look):
  - `attendance_pct` (a percentage) is rounded to 2 decimals per R8; `attendance_rate` and `moving_avg_7d` use 4 decimals.
  - `moving_avg_7d` is the mean of the *reported* (4-decimal) `attendance_rate` values in the window, as the contract words it. Rounding is done with exact integer-unit arithmetic inside the pipeline.
  - `late_minutes` for an overnight shift is measured from the start of the shift that began on the record's date, so a 00:30 punch-in on a 22:00 shift is 150 minutes late.
  - A `null` field in a PATCH body is treated as "not supplied".
- **Limitations:** no authentication (the contract has none); if MongoDB is unreachable at boot the app still starts and `/health` returns 503 until it returns (indexes are then created); aggregation behaviour (pipelines, indexes, explain plans, races) is only exercised by the integration tests, which need a real MongoDB; run them before relying on those parts.

## AI assistance

An AI assistant (Claude) was used as a reference while reviewing the starter, drafting tests and writing documentation. I reviewed the code and can explain every decision.
