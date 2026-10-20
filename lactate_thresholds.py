"""Lactate thresholds and the HR training zones built on them.

One method, used everywhere a threshold is shown -- the entry page zones, the
reporting zones, and the reporting threshold trend -- so the 2 mmol power in a
zones table is the same 2 mmol power that is plotted.

  * Lactate vs power is a quadratic fit (linear with only two powers), the
    curve already drawn on the lactate plots. A threshold is where that curve
    rises through the target: an early lactate dip makes the curve U-shaped,
    and the falling side also crosses low targets, but is not the threshold.
    This matches RCA-Coach's athlete profile in every realistic test. A target
    the fitted curve never reaches within the tested power range is None --
    never a stand-in value from the first or last step.
  * HR vs power is a straight-line fit (the HR~PO slope/intercept shown on the
    entry page), and every zone boundary's HR comes from it, so adjacent zones
    always meet exactly.

Kept free of Dash so a script can import it.
"""

import numpy as np
import pandas as pd

LA_LOW_C6 = 1.5
LA_2 = 2.0
LA_4 = 4.0
Z1_FLOOR_HR = 100.0

ZONE_CODES = [("Z1", "C7"), ("Z2", "C6"), ("Z3", "C5"), ("Z4", "C4"), ("Z5", "C3"), ("Z6", "C2/C1")]


def _clean_pairs(x, y):
    d = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"),
                      "y": pd.to_numeric(pd.Series(y), errors="coerce")}).dropna()
    # Repeated powers collapse to their mean, as the lactate plots do.
    d = d.groupby("x", as_index=False)["y"].mean().sort_values("x")
    return d["x"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float)


def fit_lactate_curve(power, lactate):
    """np.poly1d of lactate on power and the (min, max) tested power, or None."""
    x, y = _clean_pairs(power, lactate)
    if len(x) < 2 or x.min() == x.max():
        return None
    deg = 2 if len(x) >= 3 else 1
    try:
        poly = np.poly1d(np.polyfit(x, y, deg=deg))
    except Exception:
        return None
    return {"poly": poly, "lo": float(x.min()), "hi": float(x.max()), "degree": deg, "n_points": len(x)}


def power_at_lactate(curve, target, grid_n=2000):
    """Power where the fitted curve rises through `target`, within the tested range."""
    if curve is None:
        return None
    grid = np.linspace(curve["lo"], curve["hi"], grid_n)
    diff = curve["poly"](grid) - float(target)

    if np.isclose(diff[0], 0.0, atol=1e-9) and diff[1] > diff[0]:
        return float(grid[0])
    for i in range(len(diff) - 1):
        d1, d2 = diff[i], diff[i + 1]
        if d1 < 0 <= d2:  # rising through the target
            return float(grid[i] - d1 * (grid[i + 1] - grid[i]) / (d2 - d1))
    return None


def lactate_thresholds(power, lactate, targets=(2, 4, 6)):
    """{"lt_2_w": ..., "lt_4_w": ..., "fit_degree": ..., "n_points": ...}; NaN where not reached."""
    curve = fit_lactate_curve(power, lactate)
    out = {}
    for t in targets:
        value = power_at_lactate(curve, t)
        out[f"lt_{t:g}_w".replace(".", "_")] = value if value is not None else np.nan
    out["fit_degree"] = curve["degree"] if curve else np.nan
    out["n_points"] = curve["n_points"] if curve else len(_clean_pairs(power, lactate)[0])
    return out


def fit_hr_power(power, hr):
    """(slope, intercept) of HR on power, or None. Same line as the HR~PO cards:
    every step is a point, repeated powers included."""
    d = pd.DataFrame({"x": pd.to_numeric(pd.Series(power), errors="coerce"),
                      "y": pd.to_numeric(pd.Series(hr), errors="coerce")}).dropna()
    x, y = d["x"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float)
    if len(x) < 2 or x.min() == x.max():
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    if slope <= 0:  # HR not rising with power: no usable conversion
        return None
    return float(slope), float(intercept)


def compute_hr_zones(power, hr, lactate, max_hr=None, rate=None):
    """Zone rows (without split/rate formatting) from one test's steps.

    Returns a list of dicts with Zone, HR_low/high, PO_low/high, Rate_low/high
    and Notes; values that cannot be worked out are None and say why.
    """
    power = pd.to_numeric(pd.Series(power), errors="coerce")
    hr = pd.to_numeric(pd.Series(hr), errors="coerce")
    lactate = pd.to_numeric(pd.Series(lactate), errors="coerce")

    curve = fit_lactate_curve(power, lactate)
    hr_line = fit_hr_power(power, hr)
    if curve is None or hr_line is None:
        return None

    slope, intercept = hr_line

    def hr_at(po):
        return None if po is None else slope * po + intercept

    def po_at(h):
        return None if h is None else max((h - intercept) / slope, 0.0)

    po_15 = power_at_lactate(curve, LA_LOW_C6)
    po_2 = power_at_lactate(curve, LA_2)
    po_4 = power_at_lactate(curve, LA_4)
    po_mid = (po_2 + po_4) / 2.0 if po_2 is not None and po_4 is not None else None

    max_hr_note = ""
    if max_hr in (None, ""):
        max_hr = hr.max()
        max_hr_note = " (max HR not entered: using highest step HR)"
    max_hr = float(max_hr) if pd.notna(max_hr) else None

    hr_15, hr_2, hr_4 = hr_at(po_15), hr_at(po_2), hr_at(po_4)
    hr_mid = hr_at(po_mid)
    hr_56 = (hr_4 + max_hr) / 2.0 if hr_4 is not None and max_hr is not None else None

    def missing(*named):
        gaps = [label for label, value in named if value is None]
        return f"{' and '.join(gaps)} not reached in this test" if gaps else ""

    def row(i, hr_lo, hr_hi, po_lo, po_hi, note):
        zone, label = ZONE_CODES[i]
        return {"Zone": f"{zone}/{label}", "HR_low": hr_lo, "HR_high": hr_hi,
                "PO_low": po_lo, "PO_high": po_hi, "Notes": note}

    rows = [
        row(0, Z1_FLOOR_HR, hr_15, po_at(Z1_FLOOR_HR), po_15,
            missing(("1.5 mmol", po_15)) or "100 bpm → 1.5 mmol"),
        row(1, hr_15, hr_2, po_15, po_2,
            missing(("1.5 mmol", po_15), ("2 mmol", po_2)) or "1.5 → 2 mmol"),
        row(2, hr_2, hr_mid, po_2, po_mid,
            missing(("2 mmol", po_2), ("4 mmol", po_4)) or "2 mmol → midpoint of 2–4 mmol power"),
        row(3, hr_mid, hr_4, po_mid, po_4,
            missing(("2 mmol", po_2), ("4 mmol", po_4)) or "Midpoint → 4 mmol"),
        row(4, hr_4, hr_56, po_4, None,
            (missing(("4 mmol", po_4)) or "4 mmol → halfway to max HR") + max_hr_note),
        row(5, hr_56, max_hr, po_at(hr_56), None,
            (missing(("4 mmol", po_4)) or "Halfway → max HR") + max_hr_note),
    ]
    # An end that could not be worked out stays None (blank in the table);
    # the other end is still a real boundary, so it is kept.
    return rows
