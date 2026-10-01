"""Copy erg results into the new data source, tagging each with its protocol.

The erg schema gained a `protocol` field and free-form distances, and that went
into a freshly registered data source rather than onto the one holding four
years of 2k and 6k results. This carries that history across: every record is
read from the old source, given the protocol its distance implies, and ingested
into the new one.

    # look first -- writes nothing
    python scripts/migrate_erg_records.py --token "$TOKEN"

    # one record, end to end
    python scripts/migrate_erg_records.py --token "$TOKEN" --limit 1 --apply

    # the rest
    python scripts/migrate_erg_records.py --token "$TOKEN" --apply

    # confirm afterwards without writing
    python scripts/migrate_erg_records.py --token "$TOKEN" --verify

Sources default to ERG_TEST_LEGACY_SOURCE_UUID and ERG_TEST_SOURCE_UUID from
settings.py, so once the new UUID is pasted there this needs no arguments.

Three properties worth knowing, because they are what make this safe to run:

* **It never deletes, and never writes to the old source.** The original
  records are untouched and stay readable. If anything about the result looks
  wrong, the old source is still the complete record of what was there.
* **It is re-runnable.** Before copying anything it reads what the new source
  already holds and matches it against the old record for record, so a run that
  stops halfway -- a timeout, a dropped connection, a Ctrl-C -- is resumed by
  running it again. Records already copied are counted and skipped rather than
  duplicated.
* **Nothing is sent that would be rejected.** Each record is validated against
  the schema in "JSON Schema/" first; one that cannot pass is listed for you to
  look at, and the rest of the migration still runs.

A record is matched by athlete, date, protocol, distance and elapsed time --
everything except `row_no`, which is a display artifact of the entry table and
says nothing about the result. Genuine duplicates in the old data are preserved
as duplicates: matching counts copies rather than mere presence.

The token: pass --token, or set WAREHOUSE_TOKEN.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jsonschema import ValidationError, validate  # noqa: E402

from erg_protocols import ERG_PROTOCOL_VALUES, infer_protocol  # noqa: E402
from settings import (  # noqa: E402
    ERG_TEST_LEGACY_SOURCE_UUID,
    ERG_TEST_SOURCE_UUID,
    SITE_URL,
)
from warehouse import (  # noqa: E402
    WarehouseAPIConfig,
    WarehouseClient,
    WarehouseClientError,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(REPO_ROOT, "JSON Schema", "erg_test_record_schema.txt")

# Entry stores 0.1 where it has no value, including the seconds part of a time
# landing exactly on a minute, so a 30:00.0 piece reads as 30 min + 0.1 s.
MISSING_SENTINEL = 0.1

# Ingested in batches so one stalled request cannot cost the whole run, and so a
# resumed run has little to redo.
BATCH_SIZE = 100


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def check_uuid(value, label):
    """A mistyped UUID reaches the API as a 404, which reads like a permissions
    problem. Saying which part is wrong is faster than guessing."""
    if UUID_RE.match(value or ""):
        return None
    groups = (value or "").split("-")
    detail = ""
    if len(groups) == 5:
        expected = (8, 4, 4, 4, 12)
        wrong = [
            f"group {i + 1} has {len(g)} characters, expected {n}"
            for i, (g, n) in enumerate(zip(groups, expected))
            if len(g) != n
        ]
        detail = " -- " + "; ".join(wrong) if wrong else ""
    return f"{label} is not a valid UUID: {value!r}{detail}"


def check_target_schema(client, source_uuid):
    """Confirm the destination really is the erg source with the new schema.

    Pointing this at the wrong source is the mistake worth catching early: the
    records would be read, validated and sent before anything complained, and
    what came back would be a wall of failed batches rather than a reason.
    """
    try:
        client.get_datasource(source_uuid=source_uuid)
    except WarehouseClientError as exc:
        return (
            f"the destination UUID does not resolve to a data source: {exc}\n"
            "  Copy it again from the warehouse -- a UUID that is merely the "
            "right shape is not necessarily a real one."
        )

    try:
        schema = client.get_head_schema(source_uuid=source_uuid)
    except Exception as exc:
        return f"could not read the schema of the destination source: {exc}"

    if not schema:
        # Optional on some deployments -- a warning, not a reason to stop.
        return None

    properties = schema.get("properties") or {}
    missing = [f for f in ("distance_m", "time_min", "protocol") if f not in properties]
    if missing:
        return (
            "the destination source's schema is missing "
            + ", ".join(missing)
            + " -- this looks like the old erg source, or a different data set "
            "entirely. Check ERG_TEST_SOURCE_UUID in settings.py."
        )

    distance = properties.get("distance_m") or {}
    if distance.get("enum"):
        return (
            "the destination source still restricts distance_m to "
            f"{distance['enum']} -- the updated schema has not been registered "
            "on it, and every 30-minute result would be rejected."
        )

    return None


def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def record_payload(rec):
    """The ingested payload out of a warehouse record wrapper."""
    if not isinstance(rec, dict):
        return None
    for key in ("data", "record", "raw"):
        if isinstance(rec.get(key), dict):
            return rec[key]
    return None


def _number(value, drop_sentinel=True):
    try:
        if value in (None, ""):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if drop_sentinel and abs(number - MISSING_SENTINEL) < 1e-9:
        return None
    return number


def total_seconds(payload):
    minutes = _number(payload.get("time_min")) or 0.0
    seconds = _number(payload.get("time_s")) or 0.0
    total = minutes * 60 + seconds
    return total or None


def record_key(payload):
    """What makes two erg records the same result.

    row_no is left out on purpose: it numbers rows in the entry table, not
    tests, and the same result re-entered on a different line is not a different
    result. Times are rounded to a hundredth so a float that survived a JSON
    round trip as 12.400000000000001 still matches itself.
    """
    def rounded(field):
        value = _number(payload.get(field), drop_sentinel=False)
        return None if value is None else round(value, 2)

    return (
        _number(payload.get("profile_id"), drop_sentinel=False),
        str(payload.get("test_date") or ""),
        payload.get("protocol") or infer_protocol(
            payload.get("distance_m"), total_seconds(payload)
        ),
        rounded("distance_m"),
        rounded("time_min"),
        rounded("time_s"),
    )


def resolve_token(explicit):
    if explicit:
        return explicit

    # `--token "$TOKEN"` with an unset variable arrives as an empty string, and
    # looks from here exactly like no flag at all. Saying so beats sending the
    # reader to look for a token they thought they had passed.
    if explicit is not None:
        print(
            "  (--token was passed but is empty -- is the variable set in this shell?)",
            file=sys.stderr,
        )

    env_token = os.environ.get("WAREHOUSE_TOKEN")
    if env_token:
        return env_token

    # Written by scripts/get_token.py. Read before falling back to the OAuth
    # grant so a second terminal needs nothing passed to it.
    token_file = os.path.join(REPO_ROOT, ".warehouse_token")
    if os.path.exists(token_file):
        with open(token_file, encoding="utf-8") as handle:
            file_token = handle.read().strip()
        if file_token:
            return file_token

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from auth_cli import get_cli_token
    except ImportError:
        return None

    try:
        return get_cli_token()
    except Exception as exc:  # the CLI grant is not enabled on every deployment
        print(f"  (client-credentials token unavailable: {exc})", file=sys.stderr)
        return None


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--token", help="Warehouse access token (or set WAREHOUSE_TOKEN).")
    parser.add_argument(
        "--apply", action="store_true", help="Actually ingest. Without it nothing is written."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only compare the two sources and report. Writes nothing, even with --apply.",
    )
    parser.add_argument(
        "--limit", type=int, help="Stop after this many records. Use --limit 1 first."
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print the full contents of every record that could not be copied, "
             "with the dates of the records ingested alongside it.",
    )
    parser.add_argument(
        "--from-source",
        default=ERG_TEST_LEGACY_SOURCE_UUID,
        help="Source to read. Defaults to ERG_TEST_LEGACY_SOURCE_UUID.",
    )
    parser.add_argument(
        "--to-source",
        default=ERG_TEST_SOURCE_UUID,
        help="Source to write. Defaults to ERG_TEST_SOURCE_UUID.",
    )
    return parser.parse_args(argv)


def read_source(client, source_uuid, label):
    try:
        return client.list_records(source_uuid=source_uuid, role="primary", page_size=500)
    except WarehouseClientError as exc:
        print(f"Could not read the {label} source ({source_uuid}): {exc}", file=sys.stderr)
        return None


def summarise_keys(records):
    """Counted record keys, so duplicates are preserved rather than collapsed."""
    counts = Counter()
    for rec in records:
        payload = record_payload(rec)
        if payload is not None:
            counts[record_key(payload)] += 1
    return counts


def main(argv=None, client=None):
    args = parse_args(argv)

    if not args.from_source or not args.to_source:
        print("Both --from-source and --to-source are required.", file=sys.stderr)
        return 2

    uuid_problems = [
        problem
        for problem in (
            check_uuid(args.from_source, "--from-source"),
            check_uuid(args.to_source, "--to-source"),
        )
        if problem
    ]
    if uuid_problems:
        for problem in uuid_problems:
            print(problem, file=sys.stderr)
        return 2

    if args.from_source == args.to_source:
        print(
            "The two sources are the same UUID. Set the new one in settings.py "
            "(ERG_TEST_SOURCE_UUID) or pass --to-source.",
            file=sys.stderr,
        )
        return 2

    if client is None:
        token = resolve_token(args.token)
        if not token:
            print(
                "No warehouse token.\n"
                "(Checked WAREHOUSE_TOKEN and .warehouse_token.)\n"
                "This deployment's OAuth client cannot issue one from the command "
                "line, so get one by logging in:\n"
                "    python scripts/get_token.py\n"
                "It prints an `export WAREHOUSE_TOKEN=...` line to paste here.",
                file=sys.stderr,
            )
            return 2
        client = WarehouseClient(
            WarehouseAPIConfig(base_url=SITE_URL), token_getter=lambda: token
        )

    schema = load_schema()
    writing = args.apply and not args.verify

    if args.verify:
        mode = "VERIFY -- comparing the two sources, nothing will be written"
    elif args.apply:
        mode = "APPLY -- records will be copied"
    else:
        mode = "DRY RUN -- nothing will be written"
    print(f"{mode}\nFrom: {args.from_source}\nTo:   {args.to_source}\n")

    schema_problem = check_target_schema(client, args.to_source)
    if schema_problem:
        print(f"Destination check failed: {schema_problem}", file=sys.stderr)
        return 2

    source_records = read_source(client, args.from_source, "old")
    if source_records is None:
        return 1
    target_records = read_source(client, args.to_source, "new")
    if target_records is None:
        return 1

    already = summarise_keys(target_records)
    remaining = Counter(already)

    to_copy = []
    unreadable = []
    invalid = []
    not_copied = Counter()
    skipped_detail = []
    skipped_present = 0

    for rec in source_records:
        payload = record_payload(rec)
        if payload is None:
            unreadable.append(rec.get("uuid") if isinstance(rec, dict) else "<no uuid>")
            if isinstance(rec, dict):
                skipped_detail.append((rec.get("uuid"), "no readable payload", rec))
            continue

        updated = dict(payload)
        if updated.get("protocol") not in ERG_PROTOCOL_VALUES:
            updated["protocol"] = infer_protocol(
                updated.get("distance_m"), total_seconds(updated)
            )

        key = record_key(updated)
        if remaining.get(key):
            remaining[key] -= 1
            skipped_present += 1
            continue

        try:
            validate(updated, schema)
        except ValidationError as exc:
            invalid.append((rec.get("uuid"), exc.message))
            skipped_detail.append((rec.get("uuid"), exc.message, rec))
            not_copied[key] += 1
            continue

        to_copy.append(updated)
        if args.limit and len(to_copy) >= args.limit:
            break

    by_protocol = Counter(record["protocol"] for record in to_copy)

    print(f"Old source records:        {len(source_records)}")
    print(f"New source records:        {len(target_records)}")
    print(f"Already copied, skipped:   {skipped_present}")
    print(f"To copy:                   {len(to_copy)}")
    for protocol, count in sorted(by_protocol.items()):
        print(f"    {protocol:<8} {count}")

    if invalid:
        print(f"\nWould not validate, not copied ({len(invalid)}):")
        for record_uuid, message in invalid[:20]:
            print(f"    {record_uuid}  {message}")
        if len(invalid) > 20:
            print(f"    ...and {len(invalid) - 20} more")

    if unreadable:
        print(f"\nNo readable payload, not copied ({len(unreadable)}):")
        for record_uuid in unreadable[:20]:
            print(f"    {record_uuid}")
        if len(unreadable) > 20:
            print(f"    ...and {len(unreadable) - 20} more")

    if args.show_skipped and skipped_detail:
        # The dates of the records ingested in the same batch are the likeliest
        # way to recover a missing one: a batch is normally one session.
        siblings = {}
        for other in source_records:
            if not isinstance(other, dict):
                continue
            other_payload = record_payload(other) or {}
            siblings.setdefault(other.get("dataset_uuid"), []).append(
                (other.get("uuid"), other_payload.get("test_date"))
            )

        print("\n" + "=" * 68)
        print("Records that could not be copied")
        print("=" * 68)
        for record_uuid, reason, rec in skipped_detail:
            print(f"\n{record_uuid}\n  why: {reason}")
            meta = {k: v for k, v in rec.items() if k not in ("data", "record", "raw")}
            print("  record: " + json.dumps(meta, default=str))
            print("  payload: " + json.dumps(record_payload(rec), indent=4, default=str))

            dataset = rec.get("dataset_uuid")
            dates = sorted(
                {date for uuid, date in siblings.get(dataset, []) if date and uuid != record_uuid}
            )
            print(f"  ingested alongside ({dataset}):")
            if dates:
                for date in dates[:10]:
                    print(f"    test_date = {date}")
                print(f"    ({len(siblings.get(dataset, [])) - 1} other record(s) in that batch)")
            else:
                print("    nothing else in that batch to compare against")

    ingested = 0
    batch_errors = []

    if writing and to_copy:
        for start in range(0, len(to_copy), BATCH_SIZE):
            batch = to_copy[start:start + BATCH_SIZE]
            try:
                _, created = client.ingest_raw(
                    source_uuid=args.to_source,
                    records=batch,
                    subject_field="profile_id",
                    validate_client_side=False,
                )
                ingested += created
                print(f"  copied {ingested}/{len(to_copy)}")
            except WarehouseClientError as exc:
                # Stop rather than push on: a failing batch usually means the
                # schema or the token is wrong, and the next 3000 records would
                # fail the same way. What is already copied stays copied, and
                # re-running picks up from there.
                batch_errors.append(str(exc))
                print(f"\nStopped: batch starting at {start} failed: {exc}", file=sys.stderr)
                break

        print(f"\nCopied: {ingested}")

    verifying = args.verify or (writing and not args.limit and not batch_errors)

    if writing and not verifying:
        if batch_errors:
            print(
                "\nSkipping verification: the run stopped early, so a comparison "
                "would just report everything it never reached."
            )
        else:
            print(
                f"\nSkipping verification: --limit {args.limit} copied a sample on "
                "purpose. Run without --limit, then --verify."
            )

    if verifying:
        after = read_source(client, args.to_source, "new")
        if after is None:
            return 1
        source_keys = Counter()
        for rec in source_records:
            payload = record_payload(rec)
            if payload is None:
                continue
            updated = dict(payload)
            if updated.get("protocol") not in ERG_PROTOCOL_VALUES:
                updated["protocol"] = infer_protocol(
                    updated.get("distance_m"), total_seconds(updated)
                )
            source_keys[record_key(updated)] += 1

        target_keys = summarise_keys(after)
        missing = source_keys - target_keys
        extra = target_keys - source_keys
        # Records listed above as unvalidatable are missing for a reason you
        # have already seen. Separating them leaves one number that matters:
        # how many records went missing with no explanation at all.
        unexplained = missing - not_copied
        explained = sum(missing.values()) - sum(unexplained.values())

        print("\nVerification")
        print(f"  old source:  {sum(source_keys.values())} record(s)")
        print(f"  new source:  {sum(target_keys.values())} record(s)")
        print(f"  not copied, listed above: {explained}")
        print(f"  MISSING, unexplained:     {sum(unexplained.values())}")
        print(f"  present only in new:      {sum(extra.values())}  (results entered since the move)")

        for key, count in list(unexplained.items())[:10]:
            print(f"    unexplained x{count}: {key}")
        if len(unexplained) > 10:
            print(f"    ...and {len(unexplained) - 10} more distinct")

        if not unexplained:
            print("\n  Every old record is either in the new source or listed above.")
            print("  The old source is untouched -- keep it until you are satisfied.")

    if not writing and not args.verify:
        print("\nNothing was written. Re-run with --apply (try --limit 1 first).")

    return 1 if (invalid or unreadable or batch_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
