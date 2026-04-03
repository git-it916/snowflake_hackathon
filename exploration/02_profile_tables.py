"""
Step 2: Profile every table - columns, row counts, date ranges,
distinct values, numeric stats, sample rows.
"""
import os
import json
import sys
import traceback

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

conn = snowflake.connector.connect(
    account=os.getenv("SF_ACCOUNT"),
    user=os.getenv("SF_USER"),
    password=os.getenv("SF_PASSWORD"),
    warehouse=os.getenv("SF_WAREHOUSE"),
    role=os.getenv("SF_ROLE"),
)
cur = conn.cursor()

with open("exploration/output/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

profiles = {}

for db, schemas in manifest.items():
    for schema, objects in schemas.items():
        for obj in objects:
            tbl_name = obj["name"]
            tbl_type = obj["type"]
            fqn = f'"{db}"."{schema}"."{tbl_name}"'
            key = f"{db}.{schema}.{tbl_name}"
            print(f"\n{'='*70}")
            print(f"PROFILING: {key} ({tbl_type})")
            print(f"{'='*70}")

            profile = {"fqn": key, "type": tbl_type, "columns": [], "row_count": None}

            # 1. Get columns
            try:
                cur.execute(f"DESC {tbl_type} {fqn}")
                columns = []
                for c in cur.fetchall():
                    col_info = {
                        "name": c[0],
                        "type": c[1],
                        "nullable": c[3] if len(c) > 3 else None,
                    }
                    columns.append(col_info)
                    print(f"  COL: {c[0]:45s} {c[1]:20s}")
                profile["columns"] = columns
            except Exception as e:
                print(f"  ERROR describing: {e}")
                profiles[key] = profile
                continue

            # 2. Row count
            try:
                cur.execute(f"SELECT COUNT(*) FROM {fqn}")
                row_count = cur.fetchone()[0]
                profile["row_count"] = row_count
                print(f"  ROWS: {row_count:,}")
            except Exception as e:
                print(f"  ERROR counting: {e}")

            # 3. Sample rows (2 rows)
            try:
                cur.execute(f"SELECT * FROM {fqn} LIMIT 3")
                col_names = [d[0] for d in cur.description]
                samples = []
                for row in cur.fetchall():
                    sample = {}
                    for cn, val in zip(col_names, row):
                        sample[cn] = str(val) if val is not None else None
                    samples.append(sample)
                profile["sample_rows"] = samples
                print(f"  SAMPLE[0]: {samples[0] if samples else 'empty'}")
            except Exception as e:
                print(f"  ERROR sampling: {e}")

            # 4. Per-column stats
            use_sample = row_count and row_count > 100_000_000
            from_clause = f"{fqn} SAMPLE (1000000 ROWS)" if use_sample else fqn

            col_stats = {}
            for col in columns:
                cn = col["name"]
                ct = col["type"].upper()
                stats = {}

                # Date/time columns: MIN, MAX, distinct count
                if any(x in ct for x in ["DATE", "TIMESTAMP", "TIME"]) or \
                   any(x in cn.upper() for x in ["STD_YM", "BASE_MONTH", "DWDD", "YM", "_DT", "_DATE"]):
                    try:
                        cur.execute(f'SELECT MIN("{cn}"), MAX("{cn}"), COUNT(DISTINCT "{cn}") FROM {from_clause}')
                        r = cur.fetchone()
                        stats["min"] = str(r[0]) if r[0] else None
                        stats["max"] = str(r[1]) if r[1] else None
                        stats["distinct_count"] = r[2]
                        print(f"    {cn} range: {stats['min']} ~ {stats['max']} ({stats['distinct_count']} distinct)")
                    except Exception as e:
                        stats["error"] = str(e)

                # Numeric columns: min, max, avg, null ratio
                elif any(x in ct for x in ["NUMBER", "INT", "FLOAT", "DOUBLE", "DECIMAL"]):
                    try:
                        cur.execute(f'''
                            SELECT
                                MIN("{cn}"), MAX("{cn}"),
                                ROUND(AVG("{cn}"), 4),
                                ROUND(MEDIAN("{cn}"), 4),
                                ROUND(SUM(CASE WHEN "{cn}" IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100, 2),
                                COUNT(DISTINCT "{cn}")
                            FROM {from_clause}
                        ''')
                        r = cur.fetchone()
                        stats["min"] = str(r[0]) if r[0] is not None else None
                        stats["max"] = str(r[1]) if r[1] is not None else None
                        stats["avg"] = str(r[2]) if r[2] is not None else None
                        stats["median"] = str(r[3]) if r[3] is not None else None
                        stats["null_pct"] = float(r[4]) if r[4] is not None else None
                        stats["distinct_count"] = r[5]
                        print(f"    {cn}: min={stats['min']}, max={stats['max']}, avg={stats['avg']}, null%={stats['null_pct']}, distinct={stats['distinct_count']}")
                    except Exception as e:
                        stats["error"] = str(e)

                # String/varchar columns: distinct count + top values
                elif any(x in ct for x in ["VARCHAR", "STRING", "TEXT", "CHAR"]):
                    try:
                        cur.execute(f'SELECT COUNT(DISTINCT "{cn}") FROM {from_clause}')
                        dc = cur.fetchone()[0]
                        stats["distinct_count"] = dc

                        if dc <= 30:
                            cur.execute(f'SELECT DISTINCT "{cn}" FROM {from_clause} ORDER BY "{cn}" LIMIT 30')
                            stats["distinct_values"] = [str(r[0]) for r in cur.fetchall()]
                        else:
                            cur.execute(f'SELECT "{cn}", COUNT(*) as cnt FROM {from_clause} GROUP BY "{cn}" ORDER BY cnt DESC LIMIT 10')
                            stats["top_values"] = [{"value": str(r[0]), "count": r[1]} for r in cur.fetchall()]

                        print(f"    {cn}: {dc} distinct values")
                        if "distinct_values" in stats:
                            print(f"      values: {stats['distinct_values'][:10]}")
                        elif "top_values" in stats:
                            print(f"      top: {[v['value'] for v in stats['top_values'][:5]]}")
                    except Exception as e:
                        stats["error"] = str(e)

                if stats:
                    col_stats[cn] = stats

            profile["column_stats"] = col_stats
            profiles[key] = profile

cur.close()
conn.close()

# Save
with open("exploration/output/profiles.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=2, ensure_ascii=False, default=str)

print(f"\n\nProfiles saved to exploration/output/profiles.json")
print(f"Total tables profiled: {len(profiles)}")
