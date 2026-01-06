
# pages/entry.py


import os
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy as sp


from auth_setup import auth
from utils import fetch_options, fetch_profiles, fetch_profile


# Register this file as a page in the larger app
dash.register_page(__name__, path="/entry", name="Entry")


# ✅ Keys must match the "id" values
DEFAULT_ROWS = [{
    "step_no": 1,
    "Type": "Submax",
    "T_PO": None,
    "A_PO": None,
    "HR": None,
    "La": None,
    "V02": None,
    "rate": None,
    "split": None,   # computed
    "rpe": None,
},    
{"step_no": 2,
    "Type": "Submax",
    "T_PO": None,
    "A_PO": None,
    "HR": None,
    "La": None,
    "V02": None,
    "rate": None,
    "split": None,   # computed
    "rpe": None,
} , 
{"step_no": 3,
    "Type": "Submax",
    "T_PO": None,
    "A_PO": None,
    "HR": None,
    "La": None,
    "V02": None,
    "rate": None,
    "split": None,   # computed
    "rpe": None,
} 
]

TABLE_COLUMNS = [
    {"name": "Step Number", "id": "step_no", "type": "numeric"},
    {"name": "Submax/Max", "id": "Type", "type": "text"},
    {"name": "Target PO", "id": "T_PO", "type": "numeric"},
    {"name": "Actual PO", "id": "A_PO", "type": "numeric"},
    {"name": "Heart Rate", "id": "HR", "type": "numeric"},
    {"name": "Blood Lactate", "id": "La", "type": "numeric"},
    {"name": "V02", "id": "V02", "type": "numeric"},
    {"name": "Rate", "id": "rate", "type": "numeric"},

    # ✅ computed column (non-editable)
    {"name": "Split time", "id": "split", "type": "numeric", "editable": False},

    {"name": "RPE", "id": "rpe", "type": "numeric"},
]

#Threshold table

THRESH_DEFAULT_ROWS = [{
    "Threshold": 'TH1',
    "Lactate": None,
    "LaB": None, "LaA": None,
    "PoB": None, "PoA": None,
    "HrB": None, "HrA": None,
}]

THRESH_COLUMNS = [
    {"name": "Threshold", "id": "threshold", "type": "text", "editable": False},
    {"name": "Lactate", "id": "La_output", "type": "numeric", "editable": True},
    {"name": "Power Output", "id": "pow_output", "type": "numeric", "editable": False},
    {"name": "Split", "id": "split_output", "type": "numeric", "editable": False},
    {"name": "HR Output", "id": "HR_output", "type": "numeric", "editable": False},
]

#Helpers

def make_card(title, body):
    return dbc.Card(
        [dbc.CardHeader(html.B(title)), dbc.CardBody(body)],
        className="shadow-sm",
    )

def safe_float(x):
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def estimate_split_seconds(power_w):
    """
    Returns estimated split (sec/500m) from power + (optional) rate.
    Base uses the common erg relationship split ~ power^(-1/3).
    Then a small rate adjustment is applied (tweak as you like).
    """
    p = to_float(power_w)

    if p is None or p <= 0:
        return None

    # Base pace from watts (sec/500m)
    pace = 500.0 * ((2.8 / p) ** (1.0 / 3.0))



    return round(pace, 2)

def add_poly_fit(fig, x, y, degree=2, name="Fit", color = 'red'):
    """
    Adds a polynomial fit line to an existing Plotly figure.
    Safely exits if not enough data.
    """
    if len(x) < degree + 1:
        return fig

    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name=name,
            line=dict(dash="solid", color = color),
        )
    )
    return fig


layout = dbc.Container(
    [
        html.H2("V02 Step Testing"),
        html.Div("Text inputs, numeric inputs, radio buttons, and an editable DataTable."),
        html.Hr(),

        dbc.Row(
            [
                dbc.Col(
                    make_card(
                        "Entry Fields",
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Full Name"),
                                            dcc.Dropdown(
                                                id="form-name",
                                                options = [],
                                                placeholder="select athlete for analysis",
                                                value=None,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Body Mass (kg)"),
                                            dbc.Input(
                                                id="form-mass",
                                                type="number",
                                                min = 0, 
                                                step = 0.1,
                                                placeholder=0.0,
                                                value=None,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="g-3",
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Test Type"),
                                            dbc.RadioItems(
                                                id="form-status",
                                                options=[
                                                    {"label": "Erg C2", "value": "erg_C2"},
                                                    {"label": "Erg RP3", "value": "erg_RP3"},
                                                    {"label": "On-Water", "value": "row"},
                                                    {"label": "Bike", "value": "bike"},
                                                    {"label": "Other", "value": "other"},
                                                ],
                                                value="erg_C2",
                                                inline=True,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Mode"),
                                            dbc.RadioItems(
                                                id="form-mode",
                                                options=[
                                                    {"label": "Max", "value": "Max"},
                                                    {"label": "Submax", "value": "Submax"},

                                                ],
                                                value="Submax",
                                                inline=True,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="g-3",
                            ),
                            html.Br(),
                            dbc.Label("Notes (multi-line)"),
                            dbc.Textarea(
                                id="form-notes",
                                placeholder="Anything you want to capture...",
                                value="",
                                style={"height": "110px"},
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button("Submit", id="form-submit", color="primary", className="w-100"),
                                        md=4,
                                    ),
                                    dbc.Col(
                                        dbc.Button("Reset", id="form-reset", color="secondary", outline=True, className="w-100"),
                                        md=4,
                                    ),
                                    dbc.Col(
                                        dbc.Button("Download JSON", id="form-download-btn", color="info", outline=True, className="w-100"),
                                        md=4,
                                    ),
                                ],
                                className="g-2",
                            ),
                            dcc.Download(id="form-download-json"),
                            html.Hr(),
                            dbc.Alert(id="form-status-msg", color="success", is_open=False),
                        ],
                    ),
                    md=4,
                ),

                dbc.Col(
                    make_card(
                        "Editable Items Table",
                        [
                            html.Div(
                                [
                                    dbc.Button("Add row", id="form-add-row", color="success", size="sm", className="me-2"),
                                    dbc.Button("Delete selected", id="form-delete-rows", color="danger", size="sm", outline=True),
                                ],
                                className="mb-2",
                            ),
                            dash_table.DataTable(
                                id="form-items-table",
                                data=DEFAULT_ROWS,
                                columns=TABLE_COLUMNS,
                                editable=True,
                                row_selectable="multi",
                                selected_rows=[],
                                page_action="native",
                                page_size=8,
                                style_table={"overflowX": "auto"},
                                style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14},
                                style_header={"fontWeight": "600"},
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(make_card("Average P0", html.H4(id="form-avg-PO", className="m-0")), md=3),
                                    dbc.Col(make_card("Average HR", html.H4(id="form-avg-HR", className="m-0")), md=3),
                                    dbc.Col(make_card("Average Rate", html.H4(id="form-avg-rate", className="m-0")), md=3),
                                    dbc.Col(make_card("Step Count", html.H4(id="form-row-count", className="m-0")), md=3),
                                ],
                                className="g-2",
                            ),
                        ],
                    ),
                    md=8,
                ),
            ],
            className="g-3",
        ),

        html.Hr(),
        make_card("Submission Preview", html.Pre(id="form-preview", className="m-0")),
        dcc.Store(id="form-last-payload"),
        ################
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="plot-la-vs-po", config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(id="plot-hr-vs-po", config={"displayModeBar": False}), md=6),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col(make_card("HR~PO Slope", html.H5(id="hr-fit-slope", className="m-0")), md=3),
                dbc.Col(make_card("HR~PO Intercept", html.H5(id="hr-fit-intercept", className="m-0")), md=3),
            ],
            className="g-2",
        ),

        ################
        html.Hr(),
        make_card(
            "Lactate Threshold Table",
            [
                html.Div(
                    [
                        dbc.Button("Add row", id="thresh-add-row", color="success", size="sm", className="me-2"),
                        dbc.Button("Delete selected", id="thresh-delete-rows", color="danger", size="sm", outline=True),
                    ],
                    className="mb-2",
                ),
                dash_table.DataTable(
                    id="thresh-table",
                    data=THRESH_DEFAULT_ROWS,
                    columns=THRESH_COLUMNS,
                    editable=True,              # only LaThreshold is editable (per-column)
                    row_selectable="multi",
                    selected_rows=[],
                    page_action="native",
                    page_size=6,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14},
                    style_header={"fontWeight": "600"},
                ),
            ],
        ),

        ################
    ],
    fluid=True,
)

@dash.callback(
    Output("form-name", "options"),  # Update the dropdown options for full name
    Input("form-status", "value"),  # Trigger this whenever form-status changes (or use other trigger)
)
def populate_name_dropdown(selected_value):
    try:
        # Retrieve token (authentication step)
        token = auth.get_token()
    except Exception as e:
        raise PreventUpdate  # If auth fails, don't update anything

    # Fetch names using your custom function
    # Replace 'SPORT_ORG_ENDPOINT' with your actual endpoint for fetching names (or profiles)
    filters = {'sport_org_id': 13}
    names = fetch_profiles(token, filters)  # Assuming this fetches profiles with full names and ids

    # Format the names for the dropdown
    rv = [{'label':f"{p['person']['first_name']} {p['person']['last_name']}",'value':f"{p['person']['first_name']} {p['person']['last_name']}"} for p in names]

    
    return rv


@dash.callback(
    Output("form-items-table", "data"),
    Output("form-items-table", "selected_rows"),
    Input("form-add-row", "n_clicks"),
    Input("form-delete-rows", "n_clicks"),
    State("form-items-table", "data"),
    State("form-items-table", "selected_rows"),
    State("form-mode", "value"),
    prevent_initial_call=True,
)
def modify_table(add_clicks, del_clicks, rows, selected_rows, mode):
    rows = rows or []
    selected_rows = selected_rows or []
    mode = mode or "Submax"

    if ctx.triggered_id == "form-add-row":
        rows.append({
            "step_no": None,
            "Type": mode,      # ✅ auto-populate
            "T_PO": None,
            "A_PO": None,
            "HR": None,
            "La": None,
            "V02": None,
            "rate": None,
            "split": None,     # computed
            "rpe": None,
        })
        return rows, []

    if ctx.triggered_id == "form-delete-rows":
        if not selected_rows:
            return no_update, no_update
        keep = [r for i, r in enumerate(rows) if i not in set(selected_rows)]
        return keep, []

    return no_update, no_update

@dash.callback(
    Output("form-items-table", "data", allow_duplicate=True),
    Input("form-mode", "value"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)
def apply_mode_to_all_rows(mode, rows):
    rows = rows or []
    mode = mode or "Submax"

    for r in rows:
        r["Type"] = mode

    return rows


@dash.callback(
    Output("form-avg-PO", "children"),
    Output("form-avg-HR", "children"),
    Output("form-avg-rate", "children"),
    Output("form-row-count", "children"),
    Input("form-items-table", "data"),
)
def update_summary_cards(rows):
    rows = rows or []

    po_vals = [to_float(r.get("A_PO")) for r in rows]
    hr_vals = [to_float(r.get("HR")) for r in rows]
    rate_vals = [to_float(r.get("rate")) for r in rows]

    po_vals = [v for v in po_vals if v is not None]
    hr_vals = [v for v in hr_vals if v is not None]
    rate_vals = [v for v in rate_vals if v is not None]

    po_avg = (sum(po_vals) / len(po_vals)) if po_vals else None
    hr_avg = (sum(hr_vals) / len(hr_vals)) if hr_vals else None
    rate_avg = (sum(rate_vals) / len(rate_vals)) if rate_vals else None

    po_txt = f"{po_avg:.1f} W" if po_avg is not None else "—"
    hr_txt = f"{hr_avg:.1f} bpm" if hr_avg is not None else "—"
    rate_txt = f"{rate_avg:.1f} spm" if rate_avg is not None else "—"

    return po_txt,hr_txt, rate_txt, str(len(rows))

@dash.callback(
    Output("form-items-table", "data", allow_duplicate=True),
    Input("form-items-table", "data_timestamp"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)

def compute_split_column(_, rows):
    rows = rows or []
    changed = False

    for r in rows:
        new_split = estimate_split_seconds(r.get("A_PO"))
        if r.get("split") != new_split:
            r["split"] = new_split
            changed = True

    return rows if changed else no_update



@dash.callback(
    Output("form-last-payload", "data"),
    Output("form-status-msg", "children"),
    Output("form-status-msg", "is_open"),
    Input("form-submit", "n_clicks"),
    Input("form-reset", "n_clicks"),
    State("form-name", "value"),
    State("form-mass", "value"),
    State("form-status", "value"),
    State("form-notes", "value"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)
def submit_or_reset(submit_clicks, reset_clicks, name, mass, status, notes, table_rows):

    trig = ctx.triggered_id

    if trig == "form-reset":
        payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), "reset": True}
        return payload, "Form reset requested.", True

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "form": {
            "name": (name or "").strip(),
            "mass": mass,
            "status": status,
            "notes": (notes or "").strip(),
        },
        "items": table_rows or [],
    }
    return payload, "Submitted successfully.", True

@dash.callback(
    Output("form-preview", "children"),
    Input("form-last-payload", "data"),
)
def show_preview(payload):
    if not payload:
        return "Nothing submitted yet."
    return json.dumps(payload, indent=2)

@dash.callback(
    Output("form-download-json", "data"),
    Input("form-download-btn", "n_clicks"),
    State("form-last-payload", "data"),
    prevent_initial_call=True,
)
def download_payload(_, payload):
    if not payload:
        return no_update
    return dict(
        content=json.dumps(payload, indent=2),
        filename="form_submission.json",
        type="application/json",
    )

@dash.callback(
    Output("plot-la-vs-po", "figure"),
    Output("plot-hr-vs-po", "figure"),
    Output("hr-fit-slope", "children"),
    Output("hr-fit-intercept", "children"),
    Input("form-items-table", "data"),
)
def update_plots(rows):

    rows = rows or []
    df = pd.DataFrame(rows)

    # Ensure required columns exist even if table is empty
    for c in ["A_PO", "La", "HR"]:
        if c not in df.columns:
            df[c] = None

    # Coerce numeric
    df["A_PO"] = pd.to_numeric(df["A_PO"], errors="coerce")
    df["La"] = pd.to_numeric(df["La"], errors="coerce")
    df["HR"] = pd.to_numeric(df["HR"], errors="coerce")

    df_la = df.dropna(subset=["A_PO", "La"])
    df_hr = df.dropna(subset=["A_PO", "HR"])

    fig_la = px.scatter(
        df_la,
        x="A_PO",
        y="La",
        labels={"A_PO": "Actual PO (W)", "La": "Blood Lactate"},
        title="Blood Lactate vs Actual PO",

    )
    fig_la.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig_la.update_traces(marker=dict(color='blue'))

    fig_la = add_poly_fit(
        fig_la,
        df_la["A_PO"].values,
        df_la["La"].values,
        degree=2,
        name="La Power Fit",
        color = 'blue', 
        
    )
    

    fig_hr = px.scatter(
        df_hr,
        x="A_PO",
        y="HR",
        labels={"A_PO": "Actual PO (W)", "HR": "Heart Rate (bpm)"},
        title="Heart Rate vs Actual PO",
    )
    fig_hr.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig_hr.update_traces(marker=dict(color="red"))

    # --- Linear regression + line ---
    slope_txt = "—"
    intercept_txt = "—"

    df_hr_fit = df_hr.dropna(subset=["A_PO", "HR"])
    if len(df_hr_fit) >= 2:
        lr = sp.stats.linregress(df_hr_fit["A_PO"].values, df_hr_fit["HR"].values)
        slope = lr.slope
        intercept = lr.intercept

        slope_txt = f"{slope:.4f} bpm/W"
        intercept_txt = f"{intercept:.2f} bpm"

        x_fit = np.linspace(df_hr_fit["A_PO"].min(), df_hr_fit["A_PO"].max(), 100)
        y_fit = intercept + slope * x_fit

        fig_hr.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=f"Linear fit (R²={lr.rvalue**2:.3f})",
                line=dict(color="red"),
            )
        )
    return fig_la, fig_hr, slope_txt, intercept_txt


@dash.callback(
    Output("thresh-table", "data"),
    Output("thresh-table", "selected_rows"),
    Input("thresh-add-row", "n_clicks"),
    Input("thresh-delete-rows", "n_clicks"),
    State("thresh-table", "data"),
    State("thresh-table", "selected_rows"),
    prevent_initial_call=True,
)
def modify_thresh_table(add_clicks, del_clicks, rows, selected_rows):
    rows = rows or []
    selected_rows = selected_rows or []

    if ctx.triggered_id == "thresh-add-row":
        rows.append({
            "Threshold": None,
            "Lactate": None, "Power Output": None,
            "Split": None, "HR Output": None,
        })
        return rows, []

    if ctx.triggered_id == "thresh-delete-rows":
        if not selected_rows:
            return no_update, no_update
        keep = [r for i, r in enumerate(rows) if i not in set(selected_rows)]
        return keep, []

    return no_update, no_update


def _clean_num(s):
    return pd.to_numeric(s, errors="coerce")

@dash.callback(
    Output("thresh-table", "data", allow_duplicate=True),
    Input("thresh-table", "data_timestamp"),
    Input("form-items-table", "data"),
    State("thresh-table", "data"),
    prevent_initial_call=True,
)
def compute_threshold_table(_, step_rows, thresh_rows):
    thresh_rows = thresh_rows or []
    step_rows = step_rows or []

    df = pd.DataFrame(step_rows)
    if df.empty:
        return no_update

    # Need La, A_PO, HR
    for c in ["La", "A_PO", "HR"]:
        if c not in df.columns:
            df[c] = None

    df["La"] = _clean_num(df["La"])
    df["A_PO"] = _clean_num(df["A_PO"])
    df["HR"] = _clean_num(df["HR"])

    df = df.dropna(subset=["La", "A_PO", "HR"]).sort_values("La")
    if df.empty:
        # nothing to compute from
        updated = []
        for r in thresh_rows:
            r = dict(r)
            for k in ["LaB","LaA","PoB","PoA","HrB","HrA"]:
                r[k] = None
            updated.append(r)
        return updated

    changed = False
    updated = []

    for r in thresh_rows:
        r2 = dict(r)
        thr = to_float(r2.get("LaThreshold"))

        # default blanks
        out = {"LaB": None, "LaA": None, "PoB": None, "PoA": None, "HrB": None, "HrA": None}

        if thr is not None:
            below = df[df["La"] <= thr].tail(1)
            above = df[df["La"] >= thr].head(1)

            if not below.empty:
                out["LaB"] = float(below["La"].iloc[0])
                out["PoB"] = float(below["A_PO"].iloc[0])
                out["HrB"] = float(below["HR"].iloc[0])

            if not above.empty:
                out["LaA"] = float(above["La"].iloc[0])
                out["PoA"] = float(above["A_PO"].iloc[0])
                out["HrA"] = float(above["HR"].iloc[0])

        # write outputs
        for k, v in out.items():
            if r2.get(k) != v:
                r2[k] = v
                changed = True

        updated.append(r2)

    return updated if changed else no_update




