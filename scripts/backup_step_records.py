"""Back up every step test record, then check it against a candidate schema.

Run this before changing the step test schema in the warehouse. It is
read-only: it downloads, it never writes to the warehouse.

    python scripts/get_token.py                  # log in once, in another terminal
    python scripts/backup_step_records.py

It writes, under backups/ (gitignored -- these are athlete records):

    step_records_<timestamp>.json   every record exactly as the API returned it
    step_schema_live_<timestamp>.json   the schema the warehouse has right now

and then reports how many existing records would fail the new schema
(JSON Schema/vo2_step_test_record_schema.txt by default), grouped by reason.
Records that fail are the ones at risk if the warehouse ever re-validates old
data against the head schema; fix those (or loosen the schema) first.
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from jsonschema import Draft202012Validator  # noqa: E402

from settings import SITE_URL, VO2_STEP_SOURCE_UUID  # noqa: E402
from warehouse import WarehouseAPIConfig, WarehouseClient, WarehouseClientError  # noqa: E402
from migrate_erg_records import resolve_token  # noqa: E402

DEFAULT_SCHEMA = os.path.join(REPO_ROOT, "JSON Schema", "vo2_step_test_record_schema.txt")
BACKUP_DIR = os.path.join(REPO_ROOT, "backups")


def record_data(rec):
    data = rec.get("data") if isinstance(rec, dict) else None
    return data if isinstance(data, dict) else rec


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--token", help="Warehouse access token (or set WAREHOUSE_TOKEN).")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Candidate schema to check against.")
    parser.add_argument("--show", type=int, default=5, help="Example failing records to print per reason.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    token = resolve_token(args.token)
    if not token:
        print("No token. Run scripts/get_token.py first.", file=sys.stderr)
        return 2

    client = WarehouseClient(WarehouseAPIConfig(base_url=SITE_URL), token_getter=lambda: token)

    try:
        records = client.list_records(source_uuid=VO2_STEP_SOURCE_UUID, role="primary", page_size=500)
        source = client.get_datasource(source_uuid=VO2_STEP_SOURCE_UUID)
    except WarehouseClientError as exc:
        print(f"Download failed: {exc}\n(An expired token gives 401 -- rerun scripts/get_token.py.)", file=sys.stderr)
        return 1

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_path = os.path.join(BACKUP_DIR, f"step_records_{stamp}.json")
    schema_path = os.path.join(BACKUP_DIR, f"step_schema_live_{stamp}.json")
    with open(records_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1, default=str)
    with open(schema_path, "w", encoding="utf-8") as handle:
        json.dump(source, handle, indent=1, default=str)

    sessions = {record_data(r).get("session_id") for r in records}
    print(f"Backed up {len(records)} record(s) across {len(sessions)} test(s) -> {records_path}")
    print(f"Live data source + schema -> {schema_path}")

    with open(args.schema, encoding="utf-8") as handle:
        candidate = json.load(handle)
    validator = Draft202012Validator(candidate)

    reasons = Counter()
    examples = {}
    failing = 0
    for rec in records:
        data = record_data(rec)
        errors = list(validator.iter_errors(data))
        if not errors:
            continue
        failing += 1
        for err in errors:
            field = ".".join(str(p) for p in err.absolute_path) or "(record)"
            reason = f"{field}: {err.message}"
            reasons[reason] += 1
            examples.setdefault(reason, [])
            if len(examples[reason]) < args.show:
                examples[reason].append(f"{rec.get('uuid')} ({data.get('test_date')}, step {data.get('step_no')})")

    print(f"\nAgainst {os.path.relpath(args.schema, REPO_ROOT)}: "
          f"{len(records) - failing} pass, {failing} fail.")
    for reason, count in reasons.most_common():
        print(f"  {count:5d} x {reason}")
        for example in examples[reason]:
            print(f"          e.g. {example}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
