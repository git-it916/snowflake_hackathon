"""
Step 1: Discover all databases, schemas, tables in the Snowflake account.
Skip system databases. Output a JSON manifest for subsequent scripts.
"""
import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

SKIP_DBS = {"SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"}

conn = snowflake.connector.connect(
    account=os.getenv("SF_ACCOUNT"),
    user=os.getenv("SF_USER"),
    password=os.getenv("SF_PASSWORD"),
    warehouse=os.getenv("SF_WAREHOUSE"),
    role=os.getenv("SF_ROLE"),
)
cur = conn.cursor()

# 1. List all databases
cur.execute("SHOW DATABASES")
databases = [row[1] for row in cur.fetchall() if row[1] not in SKIP_DBS]
print(f"Found {len(databases)} user databases: {databases}")

manifest = {}

for db in databases:
    manifest[db] = {}
    try:
        cur.execute(f"SHOW SCHEMAS IN DATABASE \"{db}\"")
        schemas = [row[1] for row in cur.fetchall() if row[1] not in ("INFORMATION_SCHEMA",)]
    except Exception as e:
        print(f"  ERROR listing schemas in {db}: {e}")
        continue

    for schema in schemas:
        manifest[db][schema] = []
        try:
            cur.execute(f"SHOW TABLES IN \"{db}\".\"{schema}\"")
            tables = [(row[1], "TABLE") for row in cur.fetchall()]
            cur.execute(f"SHOW VIEWS IN \"{db}\".\"{schema}\"")
            views = [(row[1], "VIEW") for row in cur.fetchall()]
            all_objects = tables + views
            manifest[db][schema] = [{"name": name, "type": typ} for name, typ in all_objects]
            if all_objects:
                print(f"  {db}.{schema}: {len(tables)} tables, {len(views)} views")
        except Exception as e:
            print(f"  ERROR listing tables in {db}.{schema}: {e}")

cur.close()
conn.close()

# Save manifest
os.makedirs("exploration/output", exist_ok=True)
with open("exploration/output/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nManifest saved to exploration/output/manifest.json")
print(f"\nSummary:")
total_tables = 0
for db, schemas in manifest.items():
    for schema, objs in schemas.items():
        if objs:
            total_tables += len(objs)
            for obj in objs:
                print(f"  {db}.{schema}.{obj['name']} ({obj['type']})")
print(f"\nTotal objects: {total_tables}")
