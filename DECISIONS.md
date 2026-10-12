# DECISIONS.md  (about 150-200 words in total - plain language, your own words)

1. **Indexes.** Which indexes did you create, and which query or guarantee does each one serve? Which one did you consider and reject?

   `employees`: unique `emp_code` (duplicate guard, join key); `(department, joined_on)` for trend and filtered summary; `joined_on` for the all-department summary. `attendance_logs`: unique `(emp_code, date)` (one record per day, lookups); `(date desc, emp_code)` for the listing sort and month scans; `(status, date desc, emp_code)` for status filters; `(emp_code, punch_in desc)` for punch-out. Rejected: copying `department` onto each log with a `(department, date)` index; it goes stale when someone changes department.

2. **Punch-in race.** Two identical punch-in requests arrive at the same instant. Walk through exactly what happens in your code.

   Both compute the same IST date and call `insert_one`. The unique `(emp_code, date)` index lets MongoDB admit only one; the other gets `DuplicateKeyError`, returned as 409. There is no find-then-insert, so no gap to race through.

3. **Ties.** Two employees are tied on late minutes at the cutoff of `limit`. What does your leaderboard return, and why?

   Both. `$setWindowFields` with `$rank` gives 1, 2, 2, 4, and I filter `rank <= limit` afterwards, so `limit=2` can return three rows. Cutting a tie would pick a winner by `emp_code` alone.

4. **Headcount.** How does your department summary make sure employees with zero logs are counted?

   The pipeline starts from `employees` (joined on or before month end) and `$lookup`s the month's logs. `$unwind` keeps empty results, so a person with no logs still produces a row. Headcount is distinct `emp_code`s, not log rows.

5. **One thing you would change** if this had to serve 100x the data.

   Daily roll-ups per `(department, date)`, updated on punch and correction, so analytics read a few hundred rows instead of scanning logs. Cost: a harder write path and keeping roll-ups consistent with corrections.
