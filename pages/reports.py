# pages/reporting.py

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go

from auth_setup import auth
from utils import fetch_profiles
from settings import SITE_URL, VO2_STEP_SOURCE_UUID
from warehouse import WarehouseAPIConfig, WarehouseClient, WarehouseClientError
from lactate_thresholds import compute_hr_zones, lactate_thresholds
from step_protocols import (
    is_submax_test,
    MANUAL_MAX_FIELDS,
    MANUAL_MAX_RANGES,
    rate_out_of_range,
    rate_range_message,
    step_profile,
)

import erg_editor  # registers the Erg Scores tab and its callbacks


cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=auth.get_token)

dash.register_page(__name__, path="/reports", name="Reporting")


# =========================================================
# CONSTANTS / DEFAULTS
# =========================================================
REPORT_TABLE_COLUMNS = [
    {"name": "Record UUID", "id": "__record_uuid", "editable": False},
    {"name": "Row Issues", "id": "__row_issues", "editable": False},
    {"name": "Test Date", "id": "test_date", "editable": True},
    {"name": "Athlete", "id": "athlete_name", "editable": False},
    {"name": "Athlete ID", "id": "profile_id", "editable": False},
    {"name": "Session ID", "id": "session_id", "editable": False},
    {"name": "Step", "id": "step_no", "editable": False},
    {"name": "Test intensity", "id": "mode", "editable": True, "presentation": "dropdown"},
    {"name": "Equipment", "id": "test_type", "editable": True, "presentation": "dropdown"},
    {"name": "Target PO", "id": "target_po_w", "editable": True},
    {"name": "Actual PO", "id": "actual_po_w", "editable": True},
    {"name": "HR", "id": "hr_bpm", "editable": True},
    {"name": "La", "id": "lactate_mmol", "editable": True},
    {"name": "VO2", "id": "vo2", "editable": True},
    {"name": "Rate", "id": "rate_spm", "editable": True},
    {"name": "Split (s/500)", "id": "split_sec_per_500", "editable": False},
    {"name": "RPE", "id": "rpe", "editable": True},
    {"name": "Time (s)", "id": "time_s", "editable": True},
    {"name": "MAP (W)", "id": "map_w", "editable": True},
    {"name": "VO2max (L/min)", "id": "vo2max", "editable": True},
    {"name": "Body Mass", "id": "body_mass_kg", "editable": True},
    {"name": "Notes", "id": "notes", "editable": True},
]

REPORT_DROPDOWN_OPTIONS = {
    "mode": [
        {"label": "Max", "value": "Max"},
        {"label": "Submax", "value": "Submax"},
    ],
    "test_type": [
        {"label": "Erg C2", "value": "erg_C2"},
        {"label": "Erg RP3", "value": "erg_RP3"},
        {"label": "On-Water", "value": "row"},
        {"label": "Bike", "value": "bike"},
        {"label": "Other", "value": "other"},
    ],
}

# Spellings older records were saved with. They are no longer offered in the
# dropdown, but a loaded row still holding one must not block saving the table.
REPORT_LEGACY_DROPDOWN_VALUES = {
    "mode": {"Sub Max", "max", "submax"},
}

REPORT_EDITABLE_COLUMNS = {
    col["id"]
    for col in REPORT_TABLE_COLUMNS
    if col.get("editable") and not col["id"].startswith("__")
}

REPORT_NUMERIC_COLUMNS = {
    "profile_id",
    "body_mass_kg",
    "step_no",
    "target_po_w",
    "actual_po_w",
    "hr_bpm",
    "lactate_mmol",
    "vo2",
    "rate_spm",
    "split_sec_per_500",
    "rpe",
    "time_s",
    "map_w",
    "vo2max",
}

REPORT_INTEGER_COLUMNS = {
    "profile_id",
    "step_no",
}

REPORT_STRING_DEFAULTS = {
    "notes": "",
}

REPORT_DATA_COLUMNS = [
    "profile_id",
    "session_id",
    "session_ts",
    "test_date",
    "body_mass_kg",
    "test_type",
    "mode",
    "notes",
    "step_no",
    "step_type",
    "target_po_w",
    "actual_po_w",
    "hr_bpm",
    "lactate_mmol",
    "vo2",
    "rate_spm",
    "split_sec_per_500",
    "rpe",
    "time_s",
    "map_w",
    "vo2max",
]

REPORT_NON_PAYLOAD_COLUMNS = {"athlete_name", "__row_issues"}

ZONES_DEFAULT_ROWS = [
    {"Zone": "Z1", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z2", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z3", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z4", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z5", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
]

ZONES_COLUMNS = [
    {"name": "Zone", "id": "Zone", "type": "text"},
    {"name": "HR Low", "id": "HR_low", "type": "numeric"},
    {"name": "HR High", "id": "HR_high", "type": "numeric"},
    {"name": "PO Low (W)", "id": "PO_low", "type": "numeric"},
    {"name": "PO High (W)", "id": "PO_high", "type": "numeric"},
    {"name": "Split Low (s/500)", "id": "Split_low", "type": "text"},
    {"name": "Split High (s/500)", "id": "Split_high", "type": "text"},
    {"name": "Rate Low (spm)", "id": "Rate_low", "type": "numeric"},
    {"name": "Rate High (spm)", "id": "Rate_high", "type": "numeric"},
    {"name": "Notes", "id": "Notes", "type": "text"},
]


# =========================================================
# HELPERS
# =========================================================


def make_session_label(df):
    """
    Short display label for legend, while preserving separate sessions internally.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["test_date_dt", "session_label", "session_key"]) if df is not None else pd.DataFrame(columns=["test_date_dt", "session_label", "session_key"])

    dff = df.copy()

    for col, default in [("test_date", pd.NaT), ("session_id", None), ("mode", None)]:
        if col not in dff.columns:
            dff[col] = default

    dff["test_date_dt"] = pd.to_datetime(dff["test_date"], errors="coerce")

    def _label(row):
        dt = row["test_date_dt"]
        date_txt = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "Unknown date"
        mode_txt = str(row.get("mode", "Unknown")).strip() if pd.notna(row.get("mode", None)) else "Unknown"
        return f"{date_txt} | {mode_txt}"

    def _key(row):
        dt = row["test_date_dt"]
        date_txt = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "Unknown date"
        mode_txt = str(row.get("mode", "Unknown")).strip() if pd.notna(row.get("mode", None)) else "Unknown"
        sid = row.get("session_id", "")
        return f"{date_txt} | {mode_txt} | {sid}"

    dff["session_label"] = dff.apply(_label, axis=1)   # short legend text
    dff["session_key"] = dff.apply(_key, axis=1)       # unique grouping key
    return dff

def estimate_power_at_lactate_thresholds(session_df, thresholds=(2, 4, 6)):
    """Power (W) at each lactate threshold for one session; see lactate_thresholds.py."""
    return lactate_thresholds(session_df["actual_po_w"], session_df["lactate_mmol"], targets=thresholds)


def build_lactate_threshold_trend_df(df):
    """
    One row per session with estimated LT2 / LT4 / LT6 powers.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    sess = df.copy()
    sess["test_date_dt"] = pd.to_datetime(sess["test_date"], errors="coerce")

    rows = []
    group_cols = ["session_id", "profile_id", "athlete_name", "test_date_dt"]

    for keys, g in sess.groupby(group_cols, dropna=False):
        session_id, profile_id, athlete_name, test_date_dt = keys

        est = estimate_power_at_lactate_thresholds(g, thresholds=(2, 4, 6))

        rows.append({
            "session_id": session_id,
            "profile_id": profile_id,
            "athlete_name": athlete_name,
            "test_date_dt": test_date_dt,
            "lt_2_w": est["lt_2_w"],
            "lt_4_w": est["lt_4_w"],
            "lt_6_w": est["lt_6_w"],
            "fit_degree": est["fit_degree"],
            "n_points": est["n_points"],
            "max_po": g["actual_po_w"].max(skipna=True),
            "max_la": g["lactate_mmol"].max(skipna=True),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["test_date_dt", "athlete_name", "session_id"])







def make_card(title, body):
    return dbc.Card(
        [dbc.CardHeader(html.B(title)), dbc.CardBody(body)],
        className="shadow-sm",
    )


def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def format_split_mmss(split_seconds):
    if split_seconds is None or pd.isna(split_seconds):
        return "—"
    try:
        total_seconds = float(split_seconds)
    except Exception:
        return "—"
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:05.2f}"


def safe_date_str(x):
    if x is None or x == "":
        return None
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return None


def estimate_split_seconds(power_w):
    p = to_float(power_w)
    if p is None or p <= 0:
        return None
    pace = 500.0 * ((2.8 / p) ** (1.0 / 3.0))
    return round(pace, 2)


def _interp_y_at_x(df, x_col, y_col, x_target):
    if df is None or df.empty or x_target is None:
        return None

    d = df.copy()
    d[x_col] = pd.to_numeric(d.get(x_col), errors="coerce")
    d[y_col] = pd.to_numeric(d.get(y_col), errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if len(d) < 2:
        return None

    d = d.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)
    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    if x_target <= x.min():
        return float(y[0])
    if x_target >= x.max():
        return float(y[-1])

    return float(np.interp(float(x_target), x, y))


def clean_cell_value(value, column_id):
    if column_id in REPORT_STRING_DEFAULTS and (value is None or value == "" or pd.isna(value)):
        return REPORT_STRING_DEFAULTS[column_id]
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if column_id in REPORT_NUMERIC_COLUMNS:
        numeric_value = to_float(value)
        if numeric_value is None:
            return None
        if column_id in REPORT_INTEGER_COLUMNS:
            return int(numeric_value)
        return numeric_value
    if column_id == "test_date":
        return safe_date_str(value)
    if column_id == "session_ts":
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
    return value


def row_to_warehouse_payload(row):
    payload = {}
    for col in REPORT_DATA_COLUMNS:
        value = clean_cell_value(row.get(col), col)
        # MAP / VO2max are optional and newer than most records; leave them
        # out rather than write nulls into every record that is edited.
        if col in MANUAL_MAX_FIELDS and value is None:
            continue
        payload[col] = value
    return payload


def values_equal(a, b, column_id):
    a_clean = clean_cell_value(a, column_id)
    b_clean = clean_cell_value(b, column_id)

    if a_clean is None and b_clean is None:
        return True

    if column_id in REPORT_NUMERIC_COLUMNS:
        try:
            return np.isclose(float(a_clean), float(b_clean), equal_nan=True)
        except Exception:
            return a_clean == b_clean

    return a_clean == b_clean


def display_value(value, column_id=None):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if column_id == "split_sec_per_500":
        return format_split_mmss(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def row_label(row):
    athlete = row.get("athlete_name") or row.get("profile_id") or "Athlete"
    test_date = display_value(row.get("test_date"))
    session = row.get("session_id") or "No session"
    step = display_value(row.get("step_no"))
    return f"{athlete} | {test_date} | {session} | Step {step}"


def get_changed_cells(original_records, edited_rows):
    original_df = normalize_records_to_df(original_records)
    edited_df = pd.DataFrame(edited_rows or [])

    if original_df.empty or edited_df.empty or "__record_uuid" not in edited_df.columns:
        return []

    original_by_uuid = {
        str(row["__record_uuid"]): row
        for row in original_df.to_dict("records")
        if row.get("__record_uuid")
    }

    changes = []
    for edited_row in edited_df.to_dict("records"):
        record_uuid = edited_row.get("__record_uuid")
        if not record_uuid:
            continue

        original_row = original_by_uuid.get(str(record_uuid))
        if not original_row:
            continue

        for col in REPORT_EDITABLE_COLUMNS:
            if not values_equal(edited_row.get(col), original_row.get(col), col):
                changes.append(
                    {
                        "record_uuid": str(record_uuid),
                        "column_id": col,
                        "column_name": next(
                            (c["name"] for c in REPORT_TABLE_COLUMNS if c["id"] == col),
                            col,
                        ),
                        "old": clean_cell_value(original_row.get(col), col),
                        "new": clean_cell_value(edited_row.get(col), col),
                        "row_label": row_label(edited_row),
                    }
                )

    return changes


# Steps added in the editor have no warehouse record yet. They carry a
# placeholder uuid so the table's per-row styling and change tracking still
# have something to key on; it is swapped for the real uuid once created.
NEW_ROW_PREFIX = "new:"

# A new step with none of these filled in has nothing worth saving.
STEP_MEASUREMENT_COLUMNS = [
    "target_po_w", "actual_po_w", "hr_bpm", "lactate_mmol", "vo2", "rate_spm", "rpe", "time_s",
]


def is_new_row(row):
    return str(row.get("__record_uuid") or "").startswith(NEW_ROW_PREFIX)


def get_new_rows(edited_rows):
    return [row for row in edited_rows or [] if is_new_row(row)]


def _is_blank(value):
    if value is None or value == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def session_key(row):
    """Which test a step belongs to.

    `session_id` is minted once per submit, so it is the test. Older records
    without one fall back to athlete + date + type + mode, which is the closest
    thing they have to an identity.
    """
    sid = row.get("session_id")
    if not _is_blank(sid):
        return str(sid)
    profile = to_float(row.get("profile_id"))
    return "nosession:{}:{}:{}:{}".format(
        int(profile) if profile is not None else "",
        safe_date_str(row.get("test_date")) or "",
        row.get("test_type") or "",
        row.get("mode") or "",
    )


def build_session_options(records, athlete_options):
    """One dropdown entry per test currently loaded from the warehouse."""
    df = normalize_records_to_df(records)
    if df.empty:
        return []
    df = add_athlete_names(df, athlete_options)

    sessions = {}
    for row in df.to_dict("records"):
        key = session_key(row)
        if key not in sessions:
            sessions[key] = {"row": row, "steps": 0}
        sessions[key]["steps"] += 1

    def sort_key(item):
        return safe_date_str(item[1]["row"].get("test_date")) or ""

    options = []
    for key, info in sorted(sessions.items(), key=sort_key, reverse=True):
        row = info["row"]
        label = " | ".join(
            [
                str(row.get("athlete_name") or row.get("profile_id") or "Athlete"),
                safe_date_str(row.get("test_date")) or "Unknown date",
                str(row.get("test_type") or "—"),
                str(row.get("mode") or "—"),
                f"{info['steps']} step(s)",
            ]
        )
        options.append({"label": label, "value": key})
    return options


def build_new_step_row(table_rows, original_records, target_session):
    """A blank step at the end of `target_session`, carrying its test details.

    Returns (row, insert_at) or (None, None) when the session is not in the table.
    """
    session_rows = [
        (idx, row) for idx, row in enumerate(table_rows or [])
        if session_key(row) == target_session
    ]
    if not session_rows:
        return None, None

    template = next((row for _, row in session_rows if not is_new_row(row)), session_rows[0][1])
    step_numbers = [to_float(row.get("step_no")) for _, row in session_rows]
    step_numbers = [n for n in step_numbers if n is not None]
    next_step = int(max(step_numbers)) + 1 if step_numbers else 1

    new_row = dict(template)
    for col in STEP_MEASUREMENT_COLUMNS + ["split_sec_per_500"]:
        new_row[col] = None
    new_row["step_no"] = next_step
    new_row["step_type"] = template.get("step_type") or template.get("mode")
    new_row["__row_issues"] = ""
    new_row["__record_uuid"] = f"{NEW_ROW_PREFIX}{datetime.now().strftime('%H%M%S%f')}"

    # The table shows display-formatted timestamps; save the stored ones.
    original = original_records_by_uuid(original_records).get(str(template.get("__record_uuid")))
    if original:
        new_row["session_ts"] = original.get("session_ts")
        new_row["__dataset_uuid"] = original.get("__dataset_uuid")

    return new_row, session_rows[-1][0] + 1


def validate_reporting_rows(edited_rows):
    issues = []
    for idx, row in enumerate(edited_rows or []):
        label = row_label(row)
        record_uuid = row.get("__record_uuid")

        def add_issue(column_id, message):
            issues.append(
                {
                    "row_index": idx,
                    "record_uuid": str(record_uuid) if record_uuid else "",
                    "column_id": column_id,
                    "row_label": label,
                    "message": message,
                }
            )

        numeric_ranges = {
            "actual_po_w": (0, 1200, "Actual PO should be between 0 and 1200 W."),
            "target_po_w": (0, 1200, "Target PO should be between 0 and 1200 W."),
            "hr_bpm": (30, 240, "HR should be between 30 and 240 bpm."),
            "lactate_mmol": (0, 30, "Lactate should be between 0 and 30 mmol/L."),
            "vo2": (0, 100, "VO2 should be between 0 and 100."),
            "rpe": (0, 20, "RPE should be between 0 and 20."),
            "time_s": (0, 7200, "Time should be between 0 and 7200 seconds."),
            "body_mass_kg": (20, 250, "Body mass should be between 20 and 250 kg."),
            **MANUAL_MAX_RANGES,
        }

        for col, (low, high, message) in numeric_ranges.items():
            raw_value = row.get(col)
            if raw_value in (None, ""):
                continue
            value = to_float(raw_value)
            if value is None:
                add_issue(col, f"{col} must be numeric.")
            elif value < low or value > high:
                add_issue(col, message)

        # Rate bounds depend on the test: stroke rate on a rower, cadence on a bike.
        rate_value = row.get("rate_spm")
        if rate_value not in (None, ""):
            value = to_float(rate_value)
            if value is None:
                add_issue("rate_spm", "rate_spm must be numeric.")
            elif rate_out_of_range(row.get("test_type"), value):
                add_issue("rate_spm", rate_range_message(row.get("test_type")))

        raw_date = row.get("test_date")
        if not _is_blank(raw_date) and raw_date != "NaT" and safe_date_str(raw_date) is None:
            add_issue("test_date", "Test date should be a date like 2026-10-22.")

        for col, options in REPORT_DROPDOWN_OPTIONS.items():
            value = row.get(col)
            valid_values = {opt["value"] for opt in options} | REPORT_LEGACY_DROPDOWN_VALUES.get(col, set())
            if value not in (None, "") and value not in valid_values:
                add_issue(col, f"{col} should use one of the known options.")

        if is_new_row(row) and all(_is_blank(row.get(col)) for col in STEP_MEASUREMENT_COLUMNS):
            add_issue("actual_po_w", "New step has no data - fill it in or revert.")

    return issues


def propagate_session_dates(rows, previous_rows):
    """Copy an edited test date onto every other step of the same test.

    The date belongs to the test rather than the step, so editing it on one
    step moves the whole test. Steps are matched on their key from before the
    edit, since steps without a session_id are keyed partly by date.
    Returns None when no date was edited.
    """
    previous_by_uuid = {
        str(row.get("__record_uuid")): row for row in previous_rows or [] if row.get("__record_uuid")
    }

    new_dates = {}
    for row in rows or []:
        previous = previous_by_uuid.get(str(row.get("__record_uuid")))
        if previous is None or row.get("test_date") == previous.get("test_date"):
            continue
        new_date = safe_date_str(row.get("test_date"))
        # Leave an unreadable date on its own row for validation to flag.
        if new_date is not None:
            new_dates.setdefault(session_key(previous), new_date)

    if not new_dates:
        return None

    updated = []
    for row in rows:
        previous = previous_by_uuid.get(str(row.get("__record_uuid")), row)
        key = session_key(previous)
        if key in new_dates:
            row = {**row, "test_date": new_dates[key]}
        updated.append(row)
    return updated


def build_change_summary(changes, issues, new_rows=None):
    new_rows = new_rows or []
    if issues:
        issue_rows = [
            html.Li(f"{issue['row_label']} - {issue['message']}")
            for issue in issues[:12]
        ]
        if len(issues) > 12:
            issue_rows.append(html.Li(f"...and {len(issues) - 12} more issue(s)."))
        return html.Div(
            [
                html.P("Resolve these issues before updating the warehouse."),
                html.Ul(issue_rows, className="mb-0"),
            ]
        )

    if not changes and not new_rows:
        return html.Div("No editable changes detected.")

    new_row_blocks = []
    if new_rows:
        new_row_blocks = [
            html.P(f"New step(s) to add: {len(new_rows)}", className="mb-1 mt-2"),
            html.Ul(
                [
                    html.Li(
                        f"{row_label(row)} - PO {display_value(to_float(row.get('actual_po_w')))} W, "
                        f"HR {display_value(to_float(row.get('hr_bpm')))}, "
                        f"La {display_value(to_float(row.get('lactate_mmol')))}"
                    )
                    for row in new_rows
                ],
                className="mb-2",
            ),
        ]

    grouped = {}
    for change in changes:
        grouped.setdefault(change["row_label"], []).append(change)

    blocks = []
    for label, row_changes in list(grouped.items())[:8]:
        blocks.append(html.H6(label, className="mt-2 mb-1"))
        blocks.append(
            dbc.Table(
                [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(change["column_name"]),
                                    html.Td(display_value(change["old"], change["column_id"])),
                                    html.Td(display_value(change["new"], change["column_id"])),
                                ]
                            )
                            for change in row_changes
                        ]
                    )
                ],
                bordered=True,
                size="sm",
                className="mb-2",
            )
        )

    if len(grouped) > 8:
        blocks.append(html.Div(f"...and {len(grouped) - 8} more changed row(s)."))

    if not changes:
        return html.Div(new_row_blocks)

    return html.Div(
        [
            *new_row_blocks,
            html.P(f"Ready to update {len(grouped)} row(s), {len(changes)} field change(s)."),
            dbc.Table(
                html.Thead(html.Tr([html.Th("Field"), html.Th("Current"), html.Th("New")])),
                bordered=True,
                size="sm",
                className="mb-1",
            ),
            *blocks,
        ]
    )


def build_reporting_table_styles(changes, issues):
    styles = [
        {
            "if": {"column_id": col},
            "backgroundColor": "#f3f9ff",
        }
        for col in REPORT_EDITABLE_COLUMNS
    ]

    # Steps that only exist in the editor so far.
    styles.append(
        {
            "if": {"filter_query": f'{{__record_uuid}} contains "{NEW_ROW_PREFIX}"'},
            "backgroundColor": "#e6f4ea",
        }
    )

    for change in changes:
        styles.append(
            {
                "if": {
                    "filter_query": f'{{__record_uuid}} = "{change["record_uuid"]}"',
                    "column_id": change["column_id"],
                },
                "backgroundColor": "#fff3cd",
                "border": "1px solid #d39e00",
            }
        )

    for issue in issues:
        if not issue.get("record_uuid"):
            continue
        styles.append(
            {
                "if": {
                    "filter_query": f'{{__record_uuid}} = "{issue["record_uuid"]}"',
                    "column_id": issue["column_id"],
                },
                "backgroundColor": "#f8d7da",
                "border": "1px solid #dc3545",
            }
        )

    return styles


def build_reporting_table_columns(edit_mode=False):
    columns = []
    for col in REPORT_TABLE_COLUMNS:
        next_col = col.copy()
        if next_col["id"] in REPORT_EDITABLE_COLUMNS:
            next_col["editable"] = bool(edit_mode)
        columns.append(next_col)
    return columns


def latest_session_rows(df):
    """Steps of the most recent test in `df` (by test date, then submit time)."""
    if df is None or df.empty:
        return df
    dff = df.copy()
    dff["__key"] = dff.apply(session_key, axis=1)
    dff["__date"] = pd.to_datetime(dff.get("test_date"), errors="coerce")
    dff["__ts"] = pd.to_datetime(dff.get("session_ts"), errors="coerce")
    order = dff.sort_values(["__date", "__ts"], na_position="first")
    latest = order["__key"].iloc[-1]
    return dff[dff["__key"] == latest].drop(columns=["__key", "__date", "__ts"])


def submax_zone_rows(df):
    """Only whole sessions with a known sub-max intensity can set zones."""
    if df is None or df.empty or "mode" not in df.columns:
        return pd.DataFrame()
    eligible = df["mode"].map(is_submax_test)
    keys = df.apply(session_key, axis=1)
    return df[eligible.groupby(keys).transform("all")]


def compute_report_zones_from_df(df, max_hr_input=None):
    """HR zones for one test, by the shared method in lactate_thresholds.py.

    Zones only mean something for a single test, so with several tests loaded
    the most recent eligible sub-max test is used.
    """
    if df is None or df.empty:
        return ZONES_DEFAULT_ROWS

    zones_df = latest_session_rows(submax_zone_rows(df))
    if zones_df.empty:
        return ZONES_DEFAULT_ROWS
    for c in ["hr_bpm", "lactate_mmol", "actual_po_w", "rate_spm"]:
        if c not in zones_df.columns:
            zones_df[c] = None
        zones_df[c] = pd.to_numeric(zones_df[c], errors="coerce")

    zones = compute_hr_zones(
        zones_df["actual_po_w"], zones_df["hr_bpm"], zones_df["lactate_mmol"], max_hr=max_hr_input
    )
    if zones is None:
        return ZONES_DEFAULT_ROWS

    test_type = zones_df["test_type"].dropna().iloc[0] if zones_df["test_type"].notna().any() else None
    has_split = step_profile(test_type)["has_split"]

    def rounded(value, digits):
        return round(value, digits) if value is not None else None

    def split(po):
        return format_split_mmss(estimate_split_seconds(po)) if po is not None and has_split else None

    def rate_at_hr(target_hr):
        return _interp_y_at_x(zones_df, "hr_bpm", "rate_spm", target_hr) if target_hr is not None else None

    rows = []
    for z in zones:
        rate_low, rate_high = rate_at_hr(z["HR_low"]), rate_at_hr(z["HR_high"])
        rows.append({
            "Zone": z["Zone"],
            "HR_low": rounded(z["HR_low"], 0),
            "HR_high": rounded(z["HR_high"], 0),
            "PO_low": rounded(z["PO_low"], 1),
            "PO_high": rounded(z["PO_high"], 1),
            "Split_low": split(z["PO_low"]),
            "Split_high": split(z["PO_high"]),
            "Rate_low": rounded(rate_low, 1),
            "Rate_high": rounded(rate_high, 1),
            "Notes": z["Notes"],
        })
    return rows


def dataframe_to_store_records(df):
    if df is None or df.empty:
        return []

    out = df.copy()
    out = out.replace({np.nan: None})

    for col in ["test_date", "session_ts"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: x.isoformat() if hasattr(x, "isoformat") and pd.notna(x) else None
            )

    return out.to_dict("records")


def original_records_by_uuid(records):
    rows = {}
    for rec in records or []:
        payload = _extract_record_payload(rec)
        record_uuid = payload.get("__record_uuid") or rec.get("uuid") if isinstance(rec, dict) else None
        if record_uuid:
            rows[str(record_uuid)] = payload
    return rows


def merge_replacement_record(updated_row, replacement_record):
    replacement_payload = _extract_record_payload(replacement_record)
    merged = updated_row.copy()

    if replacement_payload.get("__record_uuid"):
        merged["__record_uuid"] = replacement_payload["__record_uuid"]
    if replacement_payload.get("__dataset_uuid"):
        merged["__dataset_uuid"] = replacement_payload["__dataset_uuid"]

    return merged


def fetch_single_record_from_dataset(dataset_uuid):
    if not dataset_uuid:
        return None

    records = wc.list_records(
        source_uuid=VO2_STEP_SOURCE_UUID,
        role="primary",
        page_size=10,
        extra_params={"dataset_uuid": dataset_uuid},
    )
    return records[0] if records else None

def create_step_record(data, dataset_uuid=None):
    """Create one step record, in the test's own dataset when possible.

    Falls back to a standalone ingest, which still groups with its test because
    reporting groups by `session_id`, not by dataset.
    """
    if dataset_uuid:
        try:
            return wc.create_record(data={"dataset": dataset_uuid, "data": data})
        except WarehouseClientError:
            pass

    dataset, created_count = wc.ingest_raw(
        source_uuid=VO2_STEP_SOURCE_UUID,
        records=[data],
        subject_field="profile_id",
        validate_client_side=False,
    )
    if created_count != 1:
        raise WarehouseClientError(f"Ingestion created {created_count} records instead of 1.")
    record = fetch_single_record_from_dataset(dataset.get("uuid"))
    if not record:
        raise WarehouseClientError("Ingestion succeeded but the new record could not be reloaded.")
    return record


def new_row_to_payload(row):
    payload_row = {col: row.get(col) for col in REPORT_DATA_COLUMNS}
    # The table shows a formatted split; work it out from power the way entry does.
    payload_row["split_sec_per_500"] = (
        estimate_split_seconds(row.get("actual_po_w"))
        if step_profile(row.get("test_type"))["has_split"]
        else None
    )
    return row_to_warehouse_payload(payload_row)


def _extract_record_payload(rec):
    """
    Try to pull the actual ingested payload out of a warehouse record.
    """
    if not isinstance(rec, dict):
        return {}

    # Raw fields already top-level
    if "profile_id" in rec or "session_id" in rec or "test_date" in rec:
        return rec

    # Common wrapped shapes
    for key in ["data", "record", "raw"]:
        if isinstance(rec.get(key), dict):
            payload = rec[key].copy()
            payload["__record_uuid"] = rec.get("uuid")
            payload["__dataset_uuid"] = rec.get("dataset_uuid")
            return payload
            payload = rec[key].copy()
            payload["__record_uuid"] = rec.get("uuid")
            payload["__dataset_uuid"] = rec.get("dataset_uuid")
            return payload

    return rec


def normalize_records_to_df(records):
    expected_cols = ["__record_uuid", "__dataset_uuid"] + REPORT_DATA_COLUMNS

    if not records:
        return pd.DataFrame(columns=expected_cols)

    extracted = [_extract_record_payload(r) for r in records]
    df = pd.DataFrame(extracted)

    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    for c in REPORT_NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    df["session_ts"] = pd.to_datetime(df["session_ts"], errors="coerce")

    return df


def apply_local_filters(df, profile_id=None, start_date=None, end_date=None, test_type=None, mode=None):
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "profile_id", "session_id", "session_ts", "test_date", "body_mass_kg",
            "test_type", "mode", "notes", "step_no", "step_type", "target_po_w",
            "actual_po_w", "hr_bpm", "lactate_mmol", "vo2", "rate_spm",
            "split_sec_per_500", "rpe", "time_s"
        ])

    dff = df.copy()

    if profile_id not in (None, "", []):
        dff = dff[dff["profile_id"] == pd.to_numeric(profile_id, errors="coerce")]

    if start_date:
        dff = dff[dff["test_date"] >= pd.to_datetime(start_date)]

    if end_date:
        # inclusive end date
        dff = dff[dff["test_date"] < (pd.to_datetime(end_date) + pd.Timedelta(days=1))]

    if test_type not in (None, "", "all"):
        dff = dff[dff["test_type"] == test_type]

    if mode not in (None, "", "all"):
        dff = dff[dff["mode"] == mode]

    dff = dff.sort_values(
        by=["test_date", "session_ts", "session_id", "step_no"],
        ascending=[False, False, False, True],
        na_position="last",
    )

    return dff


def fetch_step_test_data_from_warehouse(
    source_uuid,
    profile_id=None,
    start_date=None,
    end_date=None,
    test_type=None,
    mode=None,
):
    """
    Pull records from warehouse, then filter locally by ingested test_date.
    """
    if not source_uuid:
        raise ValueError("VO2_STEP_SOURCE_UUID is not set.")

    records = wc.list_records(
        source_uuid=source_uuid,
        subject=int(profile_id) if profile_id not in (None, "", []) else None,
        role="primary",
        page_size=500,
    )

    df = normalize_records_to_df(records)

    return apply_local_filters(
        df,
        profile_id=profile_id,
        start_date=start_date,
        end_date=end_date,
        test_type=test_type,
        mode=mode,
    )
def add_athlete_names(df, athlete_options):
    if df is None or df.empty:
        df = pd.DataFrame(columns=["profile_id"])
    dff = df.copy()

    id_to_name = {}
    for opt in athlete_options or []:
        try:
            id_to_name[int(opt["value"])] = opt["label"]
        except Exception:
            continue

    dff["athlete_name"] = dff["profile_id"].map(id_to_name).fillna(
        dff["profile_id"].astype("Int64").astype(str)
    )
    return dff


# =========================================================
# LAYOUT
# =========================================================
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.Div("PHYSIOLOGY", className="page-eyebrow"),
                    html.H1("Reporting", className="mb-1"),
                    html.P(
                        "Pull step test data from the warehouse and review athlete trends.",
                        className="text-muted mb-0",
                    ),
                ]
            ),
            className="page-title-header align-items-center mt-4 mb-4",
        ),

        dcc.Store(id="reporting-athlete-options-store"),
        dcc.Store(id="reporting-data-store"),
        dcc.Store(id="reporting-edit-mode-store", data=False),

        dbc.Tabs(
            [
                dbc.Tab(
                    label="Step Tests",
                    tab_id="tab-report-step-test",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    make_card(
                                        "Filters",
                                        [
                                            dbc.Label("Athlete"),
                                            dcc.Dropdown(
                                                id="reporting-athlete",
                                                options=[],
                                                placeholder="All athletes",
                                                value=None,
                                                clearable=True,
                                            ),
                                            html.Br(),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Start Date", html_for="reporting-start-date"),
                                                            dbc.Input(
                                                                type="date",
                                                                size="sm",
                                                                id="reporting-start-date",
                                                                value=(date.today() - timedelta(days=365)).isoformat(),
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("End Date", html_for="reporting-end-date"),
                                                            dbc.Input(
                                                                type="date",
                                                                size="sm",
                                                                id="reporting-end-date",
                                                                value=date.today().isoformat(),
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            html.Br(),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Equipment"),
                                                            dcc.Dropdown(
                                                                id="reporting-test-type",
                                                                options=[
                                                                    {"label": "All", "value": "all"},
                                                                    {"label": "Erg C2", "value": "erg_C2"},
                                                                    {"label": "Erg RP3", "value": "erg_RP3"},
                                                                    {"label": "On-Water", "value": "row"},
                                                                    {"label": "Bike", "value": "bike"},
                                                                    {"label": "Other", "value": "other"},
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Test intensity"),
                                                            dcc.Dropdown(
                                                                id="reporting-mode",
                                                                options=[
                                                                    {"label": "All", "value": "all"},
                                                                    {"label": "Max", "value": "Max"},
                                                                    {"label": "Submax", "value": "Submax"},
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            html.Br(),
                                            dbc.Label("Max HR (bpm)"),
                                            dbc.Input(
                                                id="reporting-max-hr",
                                                type="number",
                                                min=100,
                                                max=240,
                                                step=1,
                                                placeholder="optional",
                                                value=None,
                                            ),
                                            html.Br(),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Load Data",
                                                            id="reporting-load-btn",
                                                            color="primary",
                                                            className="w-100",
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Download CSV",
                                                            id="reporting-download-btn",
                                                            color="info",
                                                            outline=True,
                                                            className="w-100",
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            html.Br(),
                                            dbc.Button(
                                                "Edit Data",
                                                id="reporting-edit-btn",
                                                color="secondary",
                                                outline=True,
                                                className="w-100",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Revert Changes",
                                                        id="reporting-revert-btn",
                                                        color="secondary",
                                                        outline=True,
                                                        className="w-100",
                                                        disabled=True,
                                                    ),
                                                    dbc.Button(
                                                        "Review Changes",
                                                        id="reporting-update-btn",
                                                        color="success",
                                                        className="w-100 mt-2",
                                                        disabled=True,
                                                    ),
                                                ],
                                                className="mt-2",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label("Test", className="mt-3"),
                                                    dcc.Dropdown(
                                                        id="reporting-session-select",
                                                        options=[],
                                                        placeholder="Select a loaded test...",
                                                        clearable=True,
                                                    ),
                                                    dbc.Button(
                                                        "Add Step",
                                                        id="reporting-add-step-btn",
                                                        color="success",
                                                        outline=True,
                                                        className="w-100 mt-2",
                                                        disabled=True,
                                                    ),
                                                    dbc.Button(
                                                        "Delete Test",
                                                        id="reporting-delete-test-btn",
                                                        color="danger",
                                                        outline=True,
                                                        className="w-100 mt-2",
                                                        disabled=True,
                                                    ),
                                                    html.Small(
                                                        "Available in edit mode. New steps are saved with Review Changes.",
                                                        className="text-muted",
                                                    ),
                                                    dcc.ConfirmDialog(id="reporting-delete-confirm"),
                                                ],
                                            ),
                                            dcc.Download(id="reporting-download"),
                                            html.Hr(),
                                            dbc.Alert(id="reporting-status-msg", is_open=False),
                                            dbc.Alert(id="reporting-update-msg", is_open=False),
                                        ],
                                    ),
                                    md=3,
                                ),

                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(make_card("Rows", html.H4(id="reporting-rows", className="m-0")), md=3),
                                                dbc.Col(make_card("Sessions", html.H4(id="reporting-sessions", className="m-0")), md=3),
                                            ],
                                            className="g-2 mb-3",
                                        ),

                                        make_card(
                                            "Data Table",
                                            dash_table.DataTable(
                                                id="reporting-table",
                                                data=[],
                                                columns=build_reporting_table_columns(False),
                                                hidden_columns=["__record_uuid", "profile_id", "__row_issues"],
                                                editable=True,
                                                page_action="native",
                                                page_size=15,
                                                sort_action="native",
                                                filter_action="native",
                                                dropdown={
                                                    col: {"options": options}
                                                    for col, options in REPORT_DROPDOWN_OPTIONS.items()
                                                },
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "padding": "8px",
                                                    "fontFamily": "system-ui",
                                                    "fontSize": 14,
                                                    "textAlign": "left",
                                                    "minWidth": "100px",
                                                    "maxWidth": "220px",
                                                    "whiteSpace": "normal",
                                                },
                                                style_header={"fontWeight": "600"},
                                                style_header_conditional=[
                                                    {
                                                        "if": {"column_id": col},
                                                        "backgroundColor": "#e8f4ff",
                                                        "color": "#0b4f79",
                                                    }
                                                    for col in REPORT_EDITABLE_COLUMNS
                                                ],
                                                style_data_conditional=[
                                                    {
                                                        "if": {"column_id": col},
                                                        "backgroundColor": "#f3f9ff",
                                                    }
                                                    for col in REPORT_EDITABLE_COLUMNS
                                                ],
                                            ),
                                        ),
                                        html.Br(),
                                        make_card(
                                            "HR Training Zones (from Lactate Thresholds)",
                                            dash_table.DataTable(
                                                id="reporting-zones-table",
                                                data=ZONES_DEFAULT_ROWS,
                                                columns=ZONES_COLUMNS,
                                                editable=False,
                                                page_action="none",
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "padding": "8px",
                                                    "fontFamily": "system-ui",
                                                    "fontSize": 14,
                                                    "textAlign": "left",
                                                    "minWidth": "90px",
                                                    "maxWidth": "220px",
                                                    "whiteSpace": "normal",
                                                },
                                                style_header={"fontWeight": "600"},
                                            ),
                                        ),
                                        html.Small(id="reporting-zones-source", className="text-muted"),
                                    ],
                                    md=9,
                                ),
                            ],
                            className="g-3",
                        ),

                        html.Hr(),

                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="reporting-po-hr-plot", config={"displayModeBar": False}), md=6),
                                dbc.Col(dcc.Graph(id="reporting-po-la-plot", config={"displayModeBar": False}), md=6),
                            ],
                            className="g-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="reporting-session-trend-plot", config={"displayModeBar": False}), md=12),
                            ],
                            className="g-2 mt-2",
                        ),

                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Review changes")),
                                dbc.ModalBody(id="reporting-change-summary"),
                                dbc.ModalFooter(
                                    [
                                        dbc.Button("Cancel", id="reporting-cancel-update-btn", color="secondary", outline=True),
                                        dbc.Button(
                                            "Save changes",
                                            id="reporting-confirm-update-btn",
                                            color="success",
                                            disabled=True,
                                        ),
                                    ]
                                ),
                            ],
                            id="reporting-review-modal",
                            size="lg",
                            is_open=False,
                            scrollable=True,
                        ),
                    ],
                ),

                # The erg tab carries its own filters, table and review
                # modal; see erg_editor.py.
                dbc.Tab(
                    label="Erg Scores",
                    tab_id="tab-report-erg",
                    children=[erg_editor.layout],
                ),
            ],
            id="reporting-tabs",
            active_tab="tab-report-step-test",
            className="mt-2",
        ),
    ],
    fluid=True,
    className="physiology-page reporting-page",
)


# Keep report browsing separate from record maintenance.
_report_tab = layout.children[-1].children[0]
_report_row = _report_tab.children[0]
_filter_body = _report_row.children[0].children.children[1]
_controls = _filter_body.children
_edit_button = next(c for c in _controls if getattr(c, "id", None) == "reporting-edit-btn")
_edit_index = _controls.index(_edit_button)
_review_controls = _controls[_edit_index + 1]
_session_controls = _controls[_edit_index + 2]
_session_label, _session_picker = _session_controls.children[:2]
_session_label.children = "Test to edit"
_session_controls.children = _session_controls.children[2:]
_filter_body.children = (_controls[:_edit_index] + [
    dbc.Label("Sub-max test for training zones"),
    dcc.Dropdown(id="reporting-zone-session-select", options=[], clearable=False,
                 placeholder="No sub-max tests loaded"), html.Hr(),
    _edit_button, html.Div([_session_label, _session_picker, _review_controls, _session_controls], id="reporting-edit-tools",
    style={"display": "none"})] + _controls[_edit_index + 3:])
_results = _report_row.children[1].children
# Summary, zones, provenance, then the detailed editable records.
_report_row.children[1].children = [_results[0], _results[3], _results[4], *_report_tab.children[2:4], html.Br(),
    html.Details([html.Summary("Step data and editing"), _results[1]], open=True)]
_report_tab.children = [_report_row] + _report_tab.children[4:]


# =========================================================
# LOAD ATHLETE OPTIONS
# =========================================================
@dash.callback(
    Output("reporting-athlete-options-store", "data"),
    Input("reporting-load-btn", "id"),
)
def load_athlete_options(_):
    try:
        token = auth.get_token()
        filters = {"sport_org_id": 13}
        names = fetch_profiles(token, filters)
    except Exception:
        raise PreventUpdate

    return [
        {
            "label": f"{p['person']['first_name']} {p['person']['last_name']}",
            "value": int(p["id"]),
        }
        for p in names
    ]


@dash.callback(
    Output("reporting-athlete", "options"),
    Input("reporting-athlete-options-store", "data"),
)
def apply_athlete_options(options):
    return options or []


# =========================================================
# LOAD DATA
# =========================================================
@dash.callback(
    Output("reporting-data-store", "data"),
    Output("reporting-status-msg", "children"),
    Output("reporting-status-msg", "color"),
    Output("reporting-status-msg", "is_open"),
    Input("reporting-load-btn", "n_clicks"),
    State("reporting-athlete", "value"),
    State("reporting-start-date", "value"),
    State("reporting-end-date", "value"),
    State("reporting-test-type", "value"),
    State("reporting-mode", "value"),
    prevent_initial_call=True,
    running=[(Output("reporting-load-btn", "disabled"), True, False),
             (Output("reporting-load-btn", "children"), "Loading…", "Apply filters")],
)
def load_reporting_data(n_clicks, athlete_id, start_date, end_date, test_type, mode):
    if not n_clicks:
        raise PreventUpdate

    try:
        df = fetch_step_test_data_from_warehouse(
            source_uuid=VO2_STEP_SOURCE_UUID,
            profile_id=athlete_id,
            start_date=safe_date_str(start_date),
            end_date=safe_date_str(end_date),
            test_type=test_type,
            mode=mode,
        )

        if df.empty:
            return [], "No rows found for the selected filters.", "warning", True

        records = dataframe_to_store_records(df)
        return records, f"Loaded {len(df)} row(s) from warehouse.", "success", True

    except (WarehouseClientError, ValueError, AttributeError) as e:
        return [], f"Load failed: {e}", "danger", True

    except Exception as e:
        return [], f"Unexpected error: {e}", "danger", True


# =========================================================
# TABLE + SUMMARY
# =========================================================
@dash.callback(
    Output("reporting-table", "data"),
    Output("reporting-rows", "children"),
    Output("reporting-sessions", "children"),
    Input("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
)
def update_reporting_table(records, athlete_options):
    df = normalize_records_to_df(records)

    if df.empty:
        return [], "0", "0"

    df_display = df.copy()
    df_display = add_athlete_names(df_display, athlete_options)
    df_display["test_date"] = df_display["test_date"].astype(str)
    df_display["split_sec_per_500"] = df_display["split_sec_per_500"].apply(format_split_mmss)
    df_display["__row_issues"] = ""

    rows_txt = str(len(df_display))
    sessions_txt = str(df_display["session_id"].dropna().nunique())



    return df_display.to_dict("records"), rows_txt, sessions_txt


@dash.callback(
    Output("reporting-zones-table", "data"),
    Output("reporting-zones-source", "children"),
    Input("reporting-data-store", "data"),
    Input("reporting-max-hr", "value"),
    Input("reporting-zone-session-select", "value"),
    State("reporting-athlete-options-store", "data"),
)
def update_reporting_zones(table_rows, max_hr_input, selected_session, athlete_options):
    df = submax_zone_rows(normalize_records_to_df(table_rows))
    if df.empty:
        return ZONES_DEFAULT_ROWS, "No sub-max tests loaded. Training zones require a sub-max test."

    if selected_session:
        df = df[df.apply(session_key, axis=1) == selected_session]
    if df.empty:
        return ZONES_DEFAULT_ROWS, "Select a loaded sub-max test to view training zones."
    latest = add_athlete_names(latest_session_rows(df), athlete_options).iloc[0]
    n_tests = df.apply(session_key, axis=1).nunique()
    source = (
        f"Zones from {latest.get('athlete_name') or 'athlete'}'s test on "
        f"{safe_date_str(latest.get('test_date')) or 'unknown date'} "
        f"({latest.get('mode') or 'mode unknown'})"
    )
    if n_tests > 1:
        source += f" - latest of {n_tests} loaded tests. Select a test to view its zones."
    return compute_report_zones_from_df(df, max_hr_input), source


@dash.callback(
    Output("reporting-edit-mode-store", "data"),
    Input("reporting-edit-btn", "n_clicks"),
    Input("reporting-data-store", "data"),
    State("reporting-edit-mode-store", "data"),
)
def toggle_reporting_edit_mode(edit_clicks, records, edit_mode):
    trigger = ctx.triggered_id
    if trigger == "reporting-edit-btn":
        return not bool(edit_mode)
    if trigger == "reporting-data-store":
        return False
    return bool(edit_mode)


@dash.callback(
    Output("reporting-table", "columns"),
    Output("reporting-table", "style_data_conditional"),
    Output("reporting-edit-btn", "children"),
    Output("reporting-edit-btn", "color"),
    Output("reporting-revert-btn", "disabled"),
    Output("reporting-update-btn", "disabled"),
    Input("reporting-edit-mode-store", "data"),
    Input("reporting-table", "data"),
    State("reporting-data-store", "data"),
)
def update_reporting_edit_controls(edit_mode, table_rows, original_records):
    changes = get_changed_cells(original_records, table_rows)
    issues = validate_reporting_rows(table_rows)
    styles = build_reporting_table_styles(changes, issues)
    has_changes = bool(changes) or bool(get_new_rows(table_rows))
    edit_mode = bool(edit_mode)

    return (
        build_reporting_table_columns(edit_mode),
        styles,
        "Exit Edit Mode" if edit_mode else "Edit Data",
        "warning" if edit_mode else "secondary",
        not (edit_mode and has_changes),
        not (edit_mode and has_changes),
    )


@dash.callback(
    Output("reporting-table", "data", allow_duplicate=True),
    Input("reporting-table", "data_timestamp"),
    State("reporting-table", "data"),
    State("reporting-table", "data_previous"),
    prevent_initial_call=True,
)
def apply_test_date_to_session(_, table_rows, previous_rows):
    updated = propagate_session_dates(table_rows, previous_rows)
    if updated is None:
        raise PreventUpdate
    return updated


@dash.callback(
    Output("reporting-session-select", "options"),
    Output("reporting-session-select", "value"),
    Input("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
    State("reporting-session-select", "value"),
)
def update_session_options(records, athlete_options, current):
    options = build_session_options(records, athlete_options)
    keep = current if any(opt["value"] == current for opt in options) else None
    if keep is None and records:
        latest = latest_session_rows(normalize_records_to_df(records))
        if not latest.empty:
            keep = session_key(latest.iloc[0])
    return options, keep


@dash.callback(
    Output("reporting-add-step-btn", "disabled"),
    Output("reporting-delete-test-btn", "disabled"),
    Input("reporting-edit-mode-store", "data"),
    Input("reporting-session-select", "value"),
)
def toggle_session_buttons(edit_mode, session):
    disabled = not (edit_mode and session)
    return disabled, disabled


@dash.callback(
    Output("reporting-table", "data", allow_duplicate=True),
    Output("reporting-update-msg", "children", allow_duplicate=True),
    Output("reporting-update-msg", "color", allow_duplicate=True),
    Output("reporting-update-msg", "is_open", allow_duplicate=True),
    Input("reporting-add-step-btn", "n_clicks"),
    State("reporting-session-select", "value"),
    State("reporting-table", "data"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def add_step_to_session(n_clicks, session, table_rows, original_records):
    if not n_clicks or not session:
        raise PreventUpdate

    rows = list(table_rows or [])
    new_row, insert_at = build_new_step_row(rows, original_records, session)
    if new_row is None:
        return (
            no_update,
            "That test is not in the loaded data - reload and try again.",
            "warning",
            True,
        )

    rows.insert(insert_at, new_row)
    return (
        rows,
        f"Added step {new_row['step_no']} (green row). Fill it in, then Review Changes to save.",
        "info",
        True,
    )


@dash.callback(
    Output("reporting-delete-confirm", "displayed"),
    Output("reporting-delete-confirm", "message"),
    Output("reporting-update-msg", "children", allow_duplicate=True),
    Output("reporting-update-msg", "color", allow_duplicate=True),
    Output("reporting-update-msg", "is_open", allow_duplicate=True),
    Input("reporting-delete-test-btn", "n_clicks"),
    State("reporting-session-select", "value"),
    State("reporting-session-select", "options"),
    State("reporting-table", "data"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def ask_before_delete_test(n_clicks, session, options, table_rows, original_records):
    if not n_clicks or not session:
        raise PreventUpdate

    # Deleting reloads the table from the store, which would silently drop
    # anything typed but not yet saved.
    if get_changed_cells(original_records, table_rows) or get_new_rows(table_rows):
        return (
            False,
            no_update,
            "Save or revert your unsaved changes before deleting a test.",
            "warning",
            True,
        )

    label = next((opt["label"] for opt in options or [] if opt["value"] == session), session)
    message = (
        f"Permanently delete this test from the warehouse?\n\n{label}\n\n"
        "Every step in it will be removed. This cannot be undone."
    )
    return True, message, no_update, no_update, no_update


@dash.callback(
    Output("reporting-data-store", "data", allow_duplicate=True),
    Output("reporting-update-msg", "children", allow_duplicate=True),
    Output("reporting-update-msg", "color", allow_duplicate=True),
    Output("reporting-update-msg", "is_open", allow_duplicate=True),
    Input("reporting-delete-confirm", "submit_n_clicks"),
    State("reporting-session-select", "value"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def delete_test(confirm_clicks, session, original_records):
    if not confirm_clicks or not session:
        raise PreventUpdate

    records = original_records or []
    targets = [
        rec for rec in records
        if session_key(_extract_record_payload(rec)) == session
    ]
    if not targets:
        return no_update, "That test is no longer loaded - reload and try again.", "warning", True

    deleted = set()
    error = None
    for rec in targets:
        record_uuid = _extract_record_payload(rec).get("__record_uuid")
        if not record_uuid:
            continue
        try:
            wc.delete_record(record_uuid=str(record_uuid))
            deleted.add(str(record_uuid))
        except WarehouseClientError as e:
            error = str(e)
            break

    # Keep the store honest about what is actually gone, even on a partial failure.
    remaining = [
        rec for rec in records
        if str(_extract_record_payload(rec).get("__record_uuid")) not in deleted
    ]

    if error:
        return (
            remaining,
            f"Deleted {len(deleted)} of {len(targets)} step(s), then failed: {error} "
            "Delete the test again to remove the rest.",
            "danger",
            True,
        )
    return remaining, f"Deleted test ({len(deleted)} step(s)). You can re-enter it on the Data Entry page.", "success", True


@dash.callback(
    Output("reporting-table", "data", allow_duplicate=True),
    Input("reporting-revert-btn", "n_clicks"),
    State("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
    prevent_initial_call=True,
)
def revert_reporting_table_changes(n_clicks, records, athlete_options):
    if not n_clicks:
        raise PreventUpdate

    table_rows, _, _, _, _ = update_reporting_table(records, athlete_options)
    return table_rows


@dash.callback(
    Output("reporting-review-modal", "is_open"),
    Output("reporting-change-summary", "children"),
    Output("reporting-confirm-update-btn", "disabled"),
    Input("reporting-update-btn", "n_clicks"),
    Input("reporting-cancel-update-btn", "n_clicks"),
    State("reporting-review-modal", "is_open"),
    State("reporting-table", "data"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def toggle_reporting_review_modal(review_clicks, cancel_clicks, is_open, table_rows, original_records):
    trigger = ctx.triggered_id
    if trigger == "reporting-cancel-update-btn":
        return False, no_update, True

    if trigger != "reporting-update-btn":
        raise PreventUpdate

    changes = get_changed_cells(original_records, table_rows)
    new_rows = get_new_rows(table_rows)
    issues = validate_reporting_rows(table_rows)
    return True, build_change_summary(changes, issues, new_rows), bool(issues or not (changes or new_rows))


@dash.callback(
    Output("reporting-data-store", "data", allow_duplicate=True),
    Output("reporting-update-msg", "children"),
    Output("reporting-update-msg", "color"),
    Output("reporting-update-msg", "is_open"),
    Output("reporting-review-modal", "is_open", allow_duplicate=True),
    Input("reporting-confirm-update-btn", "n_clicks"),
    State("reporting-table", "data"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def update_warehouse_records(n_clicks, edited_rows, original_records):
    if not n_clicks:
        raise PreventUpdate

    original_df = normalize_records_to_df(original_records)
    edited_df = pd.DataFrame(edited_rows or [])

    if original_df.empty or edited_df.empty:
        return no_update, "No rows loaded to update.", "warning", True, False

    if "__record_uuid" not in original_df.columns or "__record_uuid" not in edited_df.columns:
        return no_update, "Update failed: warehouse record UUIDs are missing. Reload the data and try again.", "danger", True, False

    issues = validate_reporting_rows(edited_rows)
    if issues:
        return no_update, "Update blocked: resolve validation issues before saving.", "danger", True, True

    original_by_uuid = {
        str(row["__record_uuid"]): row
        for row in original_df.to_dict("records")
        if row.get("__record_uuid")
    }
    original_payload_by_uuid = original_records_by_uuid(original_records)

    updated_count = 0
    local_updates = {}

    try:
        for edited_row in edited_df.to_dict("records"):
            record_uuid = edited_row.get("__record_uuid")
            if not record_uuid:
                continue

            record_uuid = str(record_uuid)
            original_row = original_by_uuid.get(record_uuid)
            if not original_row:
                continue

            changed_cols = [
                col
                for col in REPORT_EDITABLE_COLUMNS
                if not values_equal(edited_row.get(col), original_row.get(col), col)
            ]

            if not changed_cols:
                continue

            updated_row = original_payload_by_uuid.get(record_uuid, original_row).copy()
            for col in changed_cols:
                updated_row[col] = clean_cell_value(edited_row.get(col), col)
            updated_row["__record_uuid"] = record_uuid

            patch_payload = {"data": row_to_warehouse_payload(updated_row)}
            if updated_row.get("__dataset_uuid"):
                patch_payload["dataset"] = updated_row["__dataset_uuid"]

            try:
                wc.patch_record(
                    record_uuid=record_uuid,
                    data=patch_payload,
                )
                replacement_uuid = record_uuid
            except WarehouseClientError as patch_error:
                try:
                    wc.put_record(
                        record_uuid=record_uuid,
                        data=patch_payload,
                    )
                    replacement_uuid = record_uuid
                except WarehouseClientError as put_error:
                    try:
                        if not patch_payload.get("dataset"):
                            raise WarehouseClientError("Skipped direct create: missing dataset UUID.")
                        replacement_record = wc.create_record(data=patch_payload)
                    except WarehouseClientError as create_error:
                        try:
                            replacement_dataset, created_count = wc.ingest_raw(
                                source_uuid=VO2_STEP_SOURCE_UUID,
                                records=[patch_payload["data"]],
                                subject_field="profile_id",
                                validate_client_side=False,
                            )
                            if created_count != 1:
                                raise WarehouseClientError(
                                    f"Ingestion created {created_count} records instead of 1."
                                )
                            replacement_record = fetch_single_record_from_dataset(
                                replacement_dataset.get("uuid")
                            )
                            if not replacement_record:
                                raise WarehouseClientError(
                                    "Ingestion succeeded but the replacement record could not be reloaded."
                                )
                        except WarehouseClientError as ingest_error:
                            raise WarehouseClientError(
                                f"PATCH, PUT, direct create, and ingestion replacement all failed for {record_uuid}. "
                                f"PATCH: {patch_error}. PUT: {put_error}. "
                                f"CREATE: {create_error}. INGEST: {ingest_error}."
                            ) from ingest_error

                    replacement_uuid = replacement_record.get("uuid")
                    try:
                        wc.delete_record(record_uuid=record_uuid)
                    except WarehouseClientError as delete_error:
                        raise WarehouseClientError(
                            "Replacement record was created, but deleting the old record failed. "
                            f"Old record: {record_uuid}. Replacement record: {replacement_uuid}. "
                            f"DELETE: {delete_error}."
                        ) from delete_error

                    updated_row = merge_replacement_record(updated_row, replacement_record)

            local_updates[record_uuid] = updated_row
            if replacement_uuid and replacement_uuid != record_uuid:
                local_updates[str(replacement_uuid)] = updated_row
            updated_count += 1

    except WarehouseClientError as e:
        return no_update, f"Update failed: {e}", "danger", True, False
    except Exception as e:
        return no_update, f"Unexpected update error: {e}", "danger", True, False

    # New steps go in one at a time. If one fails, the ones already created are
    # kept in the store below, so saving again cannot create them twice.
    created_rows = []
    create_error = None
    for new_row in get_new_rows(edited_rows):
        try:
            data = new_row_to_payload(new_row)
            record = create_step_record(data, new_row.get("__dataset_uuid"))
            created_rows.append(merge_replacement_record(data, record))
        except Exception as e:
            create_error = f"Could not add {row_label(new_row)}: {e}"
            break

    if not updated_count and not created_rows:
        if create_error:
            return no_update, create_error, "danger", True, False
        return no_update, "No editable changes detected.", "info", True, False

    refreshed = []
    for row in original_df.to_dict("records"):
        record_uuid = str(row.get("__record_uuid")) if row.get("__record_uuid") else None
        refreshed.append(local_updates.get(record_uuid, row))
    refreshed.extend(created_rows)

    parts = []
    if updated_count:
        parts.append(f"Updated {updated_count} warehouse record(s).")
    if created_rows:
        parts.append(f"Added {len(created_rows)} new step(s).")
    if create_error:
        parts.append(f"{create_error} Any unsaved steps were cleared; add them again.")

    return (
        dataframe_to_store_records(pd.DataFrame(refreshed)),
        " ".join(parts),
        "warning" if create_error else "success",
        True,
        False,
    )


# =========================================================
# PLOTS
# =========================================================
@dash.callback(
    Output("reporting-po-hr-plot", "figure"),
    Output("reporting-po-la-plot", "figure"),
    Output("reporting-session-trend-plot", "figure"),
    Input("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
)
def update_reporting_plots(records, athlete_options):
    df = normalize_records_to_df(records)

    # ---------- Empty figs ----------
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        title="Apply filters to load tests and view results",
    )

    if df.empty:
        return empty_fig, empty_fig, empty_fig

    df = add_athlete_names(df, athlete_options)
    df = make_session_label(df)

    # ---------- Empty figs ----------
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        title="Apply filters to load tests and view results",
    )

    if df.empty:
        return empty_fig, empty_fig, empty_fig

    # sort for cleaner line drawing
    df = df.sort_values(["athlete_name", "test_date", "session_id", "step_no", "actual_po_w"])

    # =========================================================
    # PO vs HR with linear fits per session
    # =========================================================
    df_hr = df.dropna(subset=["actual_po_w", "hr_bpm"]).copy()

    fig_hr = go.Figure()
    if df_hr.empty:
        fig_hr = empty_fig
    else:
        # color by session/test day
        session_labels_hr = list(df_hr["session_label"].dropna().unique())
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

        color_map_hr = {
            lab: palette[i % len(palette)]
            for i, lab in enumerate(session_labels_hr)
        }

        for sess_key, g in df_hr.groupby("session_key", dropna=False):
            sess_label = g["session_label"].iloc[0]
            session_keys_hr = list(df_hr["session_key"].dropna().unique())
            color_map_hr = {
                key: palette[i % len(palette)]
                for i, key in enumerate(session_keys_hr)
            }
            g = g.sort_values("actual_po_w")
            color = color_map_hr.get(sess_key, None)

            athlete_name = g["athlete_name"].iloc[0] if "athlete_name" in g.columns and len(g) else "Athlete"

            # scatter points
            fig_hr.add_trace(
                go.Scatter(
                    x=g["actual_po_w"],
                    y=g["hr_bpm"],
                    mode="markers",
                    name=f"{sess_label} points",
                    legendgroup=str(sess_label),
                    marker=dict(size=8, color=color),
                    customdata=np.stack(
                        [
                            g["athlete_name"].astype(str),
                            g["test_date"].astype(str),
                            g["step_no"].astype(str),
                            g["test_type"].astype(str),
                            g["mode"].astype(str),
                            g["session_id"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Session: %{customdata[5]}<br>"
                        "Date: %{customdata[1]}<br>"
                        "Step: %{customdata[2]}<br>"
                        "Test Type: %{customdata[3]}<br>"
                        "Mode: %{customdata[4]}<br>"
                        "PO: %{x:.1f} W<br>"
                        "HR: %{y:.1f} bpm<extra></extra>"
                    ),
                )
            )

            # linear fit if enough data
            fit_df = g.dropna(subset=["actual_po_w", "hr_bpm"]).copy()
            fit_df = fit_df.groupby("actual_po_w", as_index=False)["hr_bpm"].mean().sort_values("actual_po_w")

            if len(fit_df) >= 2 and fit_df["actual_po_w"].nunique() >= 2:
                x = fit_df["actual_po_w"].to_numpy(dtype=float)
                y = fit_df["hr_bpm"].to_numpy(dtype=float)

                try:
                    coeffs = np.polyfit(x, y, deg=1)
                    poly = np.poly1d(coeffs)

                    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    y_fit = poly(x_fit)

                    # optional R²
                    y_pred = poly(x)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                    fig_hr.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode="lines",
                            name=f"{sess_label} fit",
                            legendgroup=str(sess_label),
                            line=dict(width=3, color=color),
                            hovertemplate=(
                                f"{athlete_name}<br>"
                                f"{sess_label}<br>"
                                f"Linear fit"
                                + (f"<br>R²: {r2:.3f}" if pd.notna(r2) else "")
                                + "<br>PO: %{x:.1f} W<br>HR: %{y:.1f} bpm<extra></extra>"
                            ),
                        )
                    )
                except Exception:
                    pass

        fig_hr.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            title="Heart Rate vs Actual PO (Linear Fit by Test Day)",
            xaxis_title="Actual PO (W)",
            yaxis_title="Heart Rate (bpm)",
            legend_title_text="Test Day / Session",
        )

    # =========================================================
    # PO vs Lactate with quadratic fits per session
    # =========================================================
    df_la = df.dropna(subset=["actual_po_w", "lactate_mmol"]).copy()

    fig_la = go.Figure()
    if df_la.empty:
        fig_la = empty_fig
    else:
        session_labels_la = list(df_la["session_label"].dropna().unique())
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

        color_map_la = {
            lab: palette[i % len(palette)]
            for i, lab in enumerate(session_labels_la)
        }

        for sess_key, g in df_la.groupby("session_key", dropna=False):
            sess_label = g["session_label"].iloc[0]
            session_keys_la = list(df_la["session_key"].dropna().unique())
            color_map_la = {
                key: palette[i % len(palette)]
                for i, key in enumerate(session_keys_la)
            }
            color = color_map_la.get(sess_key, None)

            athlete_name = g["athlete_name"].iloc[0] if "athlete_name" in g.columns and len(g) else "Athlete"

            # scatter points
            fig_la.add_trace(
                go.Scatter(
                    x=g["actual_po_w"],
                    y=g["lactate_mmol"],
                    mode="markers",
                    name=f"{sess_label} points",
                    legendgroup=str(sess_label),
                    marker=dict(size=8, color=color),
                    customdata=np.stack(
                        [
                            g["athlete_name"].astype(str),
                            g["test_date"].astype(str),
                            g["step_no"].astype(str),
                            g["test_type"].astype(str),
                            g["mode"].astype(str),
                            g["session_id"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Session: %{customdata[5]}<br>"
                        "Date: %{customdata[1]}<br>"
                        "Step: %{customdata[2]}<br>"
                        "Test Type: %{customdata[3]}<br>"
                        "Mode: %{customdata[4]}<br>"
                        "PO: %{x:.1f} W<br>"
                        "Lactate: %{y:.2f} mmol/L<extra></extra>"
                    ),
                )
            )

            # quadratic fit if enough points, otherwise linear fallback
            fit_df = g.dropna(subset=["actual_po_w", "lactate_mmol"]).copy()
            fit_df = (
                fit_df.groupby("actual_po_w", as_index=False)["lactate_mmol"]
                .mean()
                .sort_values("actual_po_w")
            )

            n_unique = fit_df["actual_po_w"].nunique()

            if n_unique >= 2:
                x = fit_df["actual_po_w"].to_numpy(dtype=float)
                y = fit_df["lactate_mmol"].to_numpy(dtype=float)

                deg = 2 if n_unique >= 3 else 1

                try:
                    coeffs = np.polyfit(x, y, deg=deg)
                    poly = np.poly1d(coeffs)

                    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 250)
                    y_fit = poly(x_fit)

                    # optional R²
                    y_pred = poly(x)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                    fit_name = "Quadratic fit" if deg == 2 else "Linear fit"

                    fig_la.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode="lines",
                            name=f"{sess_label} fit",
                            legendgroup=str(sess_label),
                            line=dict(width=3, color=color),
                            hovertemplate=(
                                f"{athlete_name}<br>"
                                f"{sess_label}<br>"
                                f"{fit_name}"
                                + (f"<br>R²: {r2:.3f}" if pd.notna(r2) else "")
                                + "<br>PO: %{x:.1f} W<br>Lactate: %{y:.2f} mmol/L<extra></extra>"
                            ),
                        )
                    )
                except Exception:
                    pass

        fig_la.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            title="Blood Lactate vs Actual PO (Fit by Test Day)",
            xaxis_title="Actual PO (W)",
            yaxis_title="Blood Lactate (mmol/L)",
            legend_title_text="Test Day / Session",
        )

    # =========================================================
    # Threshold trend
    # =========================================================
    trend_df = build_lactate_threshold_trend_df(df)

    if trend_df.empty:
        fig_trend = empty_fig
    else:
        trend_long = trend_df.melt(
            id_vars=[
                "session_id",
                "profile_id",
                "athlete_name",
                "test_date_dt",
                "fit_degree",
                "n_points",
                "max_po",
                "max_la",
            ],
            value_vars=["lt_2_w", "lt_4_w", "lt_6_w"],
            var_name="threshold",
            value_name="power_w",
        )

        threshold_map = {
            "lt_2_w": "2 mmol",
            "lt_4_w": "4 mmol",
            "lt_6_w": "6 mmol",
        }
        trend_long["threshold_label"] = trend_long["threshold"].map(threshold_map)
        trend_long = trend_long.dropna(subset=["test_date_dt", "power_w"]).copy()

        if trend_long.empty:
            fig_trend = empty_fig
        else:
            fig_trend = px.line(
                trend_long,
                x="test_date_dt",
                y="power_w",
                color="athlete_name",
                line_dash="threshold_label",
                markers=True,
                hover_data={
                    "session_id": True,
                    "athlete_name": True,
                    "threshold_label": True,
                    "power_w": ":.1f",
                    "fit_degree": True,
                    "n_points": True,
                    "max_po": ":.1f",
                    "max_la": ":.2f",
                    "test_date_dt": False,
                },
                title="Estimated Power at 2, 4, and 6 mmol Lactate Thresholds",
                labels={
                    "test_date_dt": "Test Date",
                    "power_w": "Estimated Power (W)",
                    "athlete_name": "Athlete",
                    "threshold_label": "Threshold",
                },
            )

            fig_trend.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
                legend_title_text="Athlete / Threshold",
            )

            fig_trend.update_traces(
                mode="lines+markers",
                marker=dict(size=8),
            )

    return fig_hr, fig_la, fig_trend
# =========================================================
# DOWNLOAD
# =========================================================
@dash.callback(
    Output("reporting-download", "data"),
    Input("reporting-download-btn", "n_clicks"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def download_reporting_csv(n_clicks, records):
    if not n_clicks:
        raise PreventUpdate

    df = normalize_records_to_df(records)
    if df.empty:
        raise PreventUpdate

    df_out = df.copy()
    df_out["test_date"] = df_out["test_date"].astype(str)
    return dict(
        content=df_out.to_csv(index=False),
        filename="warehouse_step_test_reporting.csv",
        type="text/csv",
    )



@dash.callback(
    Output("reporting-edit-tools", "style"),
    Input("reporting-edit-mode-store", "data"),
)
def show_reporting_edit_tools(edit_mode):
    return {} if edit_mode else {"display": "none"}


@dash.callback(
    Output("reporting-zone-session-select", "options"),
    Output("reporting-zone-session-select", "value"),
    Input("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
    State("reporting-zone-session-select", "value"),
)
def update_zone_session_options(records, athlete_options, current):
    eligible = submax_zone_rows(normalize_records_to_df(records))
    return update_session_options(eligible.to_dict("records"), athlete_options, current)
