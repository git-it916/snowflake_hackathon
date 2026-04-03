"""
Step 3: Deep-profile all 11 telecom views.
Captures exact column names, types, date ranges, ALL distinct values
for categorical columns, numeric statistics, NULL counts, and sample rows.

Output: exploration/output/telecom_profiles.json
"""
import os
import sys
import json
import traceback
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from snowflake.snowpark import Session

# ─── Connection ──────────────────────────────────────────────────────
session = Session.builder.configs(
    {
        "account": os.getenv("SF_ACCOUNT", "wjvhgmf-pv80016"),
        "user": os.getenv("SF_USER"),
        "password": os.getenv("SF_PASSWORD"),
        "role": "ACCOUNTADMIN",
        "warehouse": "COMPUTE_WH",
    }
).create()

TELECOM_DB = (
    "SOUTH_KOREA_TELECOM_SUBSCRIPTION_ANALYTICS__"
    "CONTRACTS_MARKETING_AND_CALL_CENTER_INSIGHTS_BY_REGION"
)
SCHEMA = "TELECOM_INSIGHTS"

VIEW_NAMES = [
    "V01_MONTHLY_REGIONAL_CONTRACT_STATS",
    "V02_SERVICE_BUNDLE_PATTERNS",
    "V03_CONTRACT_FUNNEL_CONVERSION",
    "V04_CHANNEL_CONTRACT_PERFORMANCE",
    "V05_REGIONAL_NEW_INSTALL",
    "V06_RENTAL_CATEGORY_TRENDS",
    "V07_GA4_MARKETING_ATTRIBUTION",
    "V08_GA4_DEVICE_STATS",
    "V09_MONTHLY_CALL_STATS",
    "V10_HOURLY_CALL_DISTRIBUTION",
    "V11_CALL_TO_CONTRACT_CONVERSION",
]


def fqn(view: str) -> str:
    """Return fully qualified name with double-quotes."""
    return f'"{TELECOM_DB}"."{SCHEMA}"."{view}"'


def run_sql(sql: str) -> list[dict[str, Any]]:
    """Execute SQL via Snowpark and return list of dicts.
    Uses Row.as_dict() for reliable key access.
    """
    df = session.sql(sql)
    rows = df.collect()
    if not rows:
        return []
    result = []
    for row in rows:
        # as_dict() returns unquoted keys
        record = row.as_dict()
        result.append(record)
    return result


def safe_str(val: Any) -> str | None:
    """Convert value to string, handling None."""
    if val is None:
        return None
    return str(val)


def profile_view(view_name: str) -> dict[str, Any]:
    """Run all profiling queries for a single view."""
    fq = fqn(view_name)
    profile: dict[str, Any] = {"view_name": view_name, "fqn": f"{TELECOM_DB}.{SCHEMA}.{view_name}"}

    print(f"\n{'=' * 70}")
    print(f"PROFILING: {view_name}")
    print(f"{'=' * 70}")

    # ── Step 1: Schema (DESCRIBE) ────────────────────────────────────
    try:
        desc_rows = run_sql(f"DESCRIBE VIEW {fq}")
        columns = []
        for r in desc_rows:
            col_info = {
                "name": r.get("name", ""),
                "type": r.get("type", ""),
                "kind": r.get("kind", ""),
                "null?": r.get("null?", ""),
                "comment": r.get("comment", ""),
            }
            columns.append(col_info)
            print(f"  COL: {col_info['name']:45s} {col_info['type']}")
        profile["columns"] = columns
    except Exception as e:
        print(f"  ERROR describing: {e}")
        traceback.print_exc()
        profile["columns"] = []
        profile["error_describe"] = str(e)
        return profile

    col_names = [c["name"] for c in columns]
    col_types = {c["name"]: c["type"] for c in columns}

    # ── Step 2: Row count ────────────────────────────────────────────
    try:
        cnt_rows = run_sql(f"SELECT COUNT(*) AS CNT FROM {fq}")
        row_count = cnt_rows[0]["CNT"]
        profile["row_count"] = row_count
        print(f"  ROWS: {row_count:,}")
    except Exception as e:
        print(f"  ERROR counting: {e}")
        profile["row_count"] = None

    # ── Step 3: Sample rows (5 rows) ─────────────────────────────────
    try:
        sample_rows = run_sql(f"SELECT * FROM {fq} LIMIT 5")
        serializable_samples = []
        for sr in sample_rows:
            row_dict = {}
            for k, v in sr.items():
                row_dict[k] = safe_str(v)
            serializable_samples.append(row_dict)
        profile["sample_rows"] = serializable_samples
        if serializable_samples:
            print(f"  SAMPLE[0]: {serializable_samples[0]}")
    except Exception as e:
        print(f"  ERROR sampling: {e}")
        profile["sample_rows"] = []

    # ── Step 4: NULL counts per column ───────────────────────────────
    try:
        null_exprs = []
        for cn in col_names:
            null_exprs.append(
                f'SUM(CASE WHEN "{cn}" IS NULL THEN 1 ELSE 0 END) AS "{cn}_NULLS"'
            )
        null_sql = f"SELECT COUNT(*) AS TOTAL_ROWS, {', '.join(null_exprs)} FROM {fq}"
        null_rows = run_sql(null_sql)
        null_info = {}
        if null_rows:
            r = null_rows[0]
            total = r["TOTAL_ROWS"]
            for cn in col_names:
                null_key = f"{cn}_NULLS"
                null_count = r.get(null_key, 0)
                null_info[cn] = {
                    "null_count": int(null_count) if null_count is not None else 0,
                    "null_pct": round(
                        float(null_count) / float(total) * 100, 2
                    )
                    if total and null_count
                    else 0.0,
                }
            print(f"  NULL COUNTS: {null_info}")
        profile["null_info"] = null_info
    except Exception as e:
        print(f"  ERROR null counts: {e}")
        traceback.print_exc()
        profile["null_info"] = {}

    # ── Step 5: Per-column deep stats ────────────────────────────────
    col_stats: dict[str, Any] = {}

    for cn in col_names:
        ct = col_types[cn].upper()
        stats: dict[str, Any] = {}

        is_date_like = any(
            x in ct for x in ["DATE", "TIMESTAMP", "TIME"]
        ) or any(
            x in cn.upper()
            for x in ["STD_YM", "YEAR_MONTH", "BASE_MONTH", "YM", "_DT", "_DATE"]
        )
        is_numeric = any(
            x in ct for x in ["NUMBER", "INT", "FLOAT", "DOUBLE", "DECIMAL"]
        )
        is_string = any(x in ct for x in ["VARCHAR", "STRING", "TEXT", "CHAR"])

        # ── Date / time-like columns ─────────────────────────────────
        if is_date_like:
            try:
                q = f"""
                    SELECT
                        MIN("{cn}") AS MIN_VAL,
                        MAX("{cn}") AS MAX_VAL,
                        COUNT(DISTINCT "{cn}") AS DISTINCT_CNT
                    FROM {fq}
                """
                r = run_sql(q)[0]
                stats["min"] = safe_str(r["MIN_VAL"])
                stats["max"] = safe_str(r["MAX_VAL"])
                stats["distinct_count"] = r["DISTINCT_CNT"]
                print(f"    {cn} range: {stats['min']} ~ {stats['max']} ({stats['distinct_count']} distinct)")

                # Get all distinct values for date columns (usually <= 70)
                if r["DISTINCT_CNT"] and r["DISTINCT_CNT"] <= 200:
                    q2 = f'SELECT DISTINCT "{cn}" AS VAL FROM {fq} ORDER BY "{cn}"'
                    vals = run_sql(q2)
                    stats["all_values"] = [safe_str(v["VAL"]) for v in vals]
            except Exception as e:
                stats["error"] = str(e)

        # ── Numeric columns ──────────────────────────────────────────
        elif is_numeric:
            try:
                q = f"""
                    SELECT
                        MIN("{cn}") AS MIN_VAL,
                        MAX("{cn}") AS MAX_VAL,
                        ROUND(AVG("{cn}"), 4) AS AVG_VAL,
                        ROUND(MEDIAN("{cn}"), 4) AS MED_VAL,
                        ROUND(STDDEV("{cn}"), 4) AS STD_VAL,
                        COUNT(DISTINCT "{cn}") AS DISTINCT_CNT,
                        ROUND(
                            SUM(CASE WHEN "{cn}" IS NULL THEN 1 ELSE 0 END)::FLOAT
                            / NULLIF(COUNT(*), 0) * 100,
                            2
                        ) AS NULL_PCT
                    FROM {fq}
                """
                r = run_sql(q)[0]
                stats["min"] = safe_str(r["MIN_VAL"])
                stats["max"] = safe_str(r["MAX_VAL"])
                stats["avg"] = safe_str(r["AVG_VAL"])
                stats["median"] = safe_str(r["MED_VAL"])
                stats["stddev"] = safe_str(r["STD_VAL"])
                stats["distinct_count"] = r["DISTINCT_CNT"]
                stats["null_pct"] = float(r["NULL_PCT"]) if r["NULL_PCT"] else 0.0
                print(
                    f"    {cn}: min={stats['min']}, max={stats['max']}, "
                    f"avg={stats['avg']}, std={stats['stddev']}, "
                    f"distinct={stats['distinct_count']}, null%={stats['null_pct']}"
                )

                # For small-cardinality numerics, get all values
                if r["DISTINCT_CNT"] and r["DISTINCT_CNT"] <= 50:
                    q2 = f"""
                        SELECT "{cn}" AS VAL, COUNT(*) AS CNT
                        FROM {fq}
                        GROUP BY "{cn}"
                        ORDER BY CNT DESC
                    """
                    vals = run_sql(q2)
                    stats["value_distribution"] = [
                        {"value": safe_str(v["VAL"]), "count": v["CNT"]}
                        for v in vals
                    ]
            except Exception as e:
                stats["error"] = str(e)

        # ── String / categorical columns ─────────────────────────────
        elif is_string:
            try:
                q = f'SELECT COUNT(DISTINCT "{cn}") AS DC FROM {fq}'
                dc = run_sql(q)[0]["DC"]
                stats["distinct_count"] = dc

                # For ALL categorical columns, get full value list up to 500
                if dc and dc <= 500:
                    q2 = f"""
                        SELECT "{cn}" AS VAL, COUNT(*) AS CNT
                        FROM {fq}
                        GROUP BY "{cn}"
                        ORDER BY CNT DESC
                    """
                    vals = run_sql(q2)
                    stats["value_distribution"] = [
                        {"value": safe_str(v["VAL"]), "count": v["CNT"]}
                        for v in vals
                    ]
                    print(
                        f"    {cn}: {dc} distinct → "
                        f"{[v['value'] for v in stats['value_distribution'][:8]]}"
                    )
                else:
                    q2 = f"""
                        SELECT "{cn}" AS VAL, COUNT(*) AS CNT
                        FROM {fq}
                        GROUP BY "{cn}"
                        ORDER BY CNT DESC
                        LIMIT 20
                    """
                    vals = run_sql(q2)
                    stats["top_values"] = [
                        {"value": safe_str(v["VAL"]), "count": v["CNT"]}
                        for v in vals
                    ]
                    print(
                        f"    {cn}: {dc} distinct (top 20) → "
                        f"{[v['value'] for v in stats['top_values'][:8]]}"
                    )
            except Exception as e:
                stats["error"] = str(e)

        if stats:
            col_stats[cn] = stats

    profile["column_stats"] = col_stats

    # ── Step 6: Date range summary ───────────────────────────────────
    date_cols = [
        cn
        for cn in col_names
        if any(
            x in cn.upper()
            for x in ["YEAR_MONTH", "STD_YM", "BASE_MONTH", "YM", "DATE"]
        )
        or any(x in col_types[cn].upper() for x in ["DATE", "TIMESTAMP"])
    ]
    if date_cols:
        profile["date_columns"] = date_cols
        for dc in date_cols:
            if dc in col_stats and "min" in col_stats[dc]:
                print(
                    f"  DATE COL [{dc}]: {col_stats[dc]['min']} ~ {col_stats[dc]['max']}"
                )

    return profile


# ─── Main ────────────────────────────────────────────────────────────
all_profiles: dict[str, Any] = {}

for vn in VIEW_NAMES:
    try:
        p = profile_view(vn)
        all_profiles[vn] = p
    except Exception as e:
        print(f"\nFATAL ERROR profiling {vn}: {e}")
        traceback.print_exc()
        all_profiles[vn] = {"view_name": vn, "error": str(e)}

session.close()

# ─── Save output ─────────────────────────────────────────────────────
os.makedirs("exploration/output", exist_ok=True)
out_path = "exploration/output/telecom_profiles.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_profiles, f, indent=2, ensure_ascii=False, default=str)

print(f"\n\n{'=' * 70}")
print(f"All telecom profiles saved to {out_path}")
print(f"Total views profiled: {len(all_profiles)}")
print(f"{'=' * 70}")
