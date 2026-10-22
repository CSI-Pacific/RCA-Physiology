"""How each step-test type labels and bounds its movement-rate column.

A step test stores one rate per step in `rate_spm`, but what that number means
depends on the test: strokes per minute on an erg or on the water, pedal
revolutions per minute on a bike. The column id stays the same so the warehouse
schema does not fork; only the label, the unit and the plausible range change.

Kept out of pages/ so the entry form and the reporting editor read the same
table and cannot disagree on what a bike cadence of 95 is allowed to be.
"""

_ROWING = {
    "rate_label": "Stroke Rate",
    "rate_unit": "spm",
    "rate_range": (0, 80),
    # The split is derived from power with the Concept2 pace formula, which
    # only means something for rowing.
    "has_split": True,
}

STEP_TEST_PROFILES = {
    "erg_C2": _ROWING,
    "erg_RP3": _ROWING,
    "row": _ROWING,
    "bike": {
        "rate_label": "Cadence",
        "rate_unit": "rpm",
        "rate_range": (0, 200),
        "has_split": False,
    },
    "other": {
        "rate_label": "Rate",
        "rate_unit": "",
        "rate_range": (0, 200),
        "has_split": True,
    },
}

# Records with a missing or unrecognised test type get the loosest bounds, so
# nothing is flagged just because its type was never filled in.
DEFAULT_STEP_PROFILE = STEP_TEST_PROFILES["other"]


def step_profile(test_type):
    return STEP_TEST_PROFILES.get(test_type, DEFAULT_STEP_PROFILE)


def rate_header(test_type):
    """Column header, e.g. "Cadence (rpm)"."""
    p = step_profile(test_type)
    return f"{p['rate_label']} ({p['rate_unit']})" if p["rate_unit"] else p["rate_label"]


def rate_range_message(test_type):
    p = step_profile(test_type)
    low, high = p["rate_range"]
    unit = f" {p['rate_unit']}" if p["rate_unit"] else ""
    return f"{p['rate_label']} should be between {low:g} and {high:g}{unit}."


def rate_out_of_range(test_type, value):
    """True when a numeric rate falls outside the test type's bounds."""
    low, high = step_profile(test_type)["rate_range"]
    return value < low or value > high


# MAP and VO2max are entered by hand for a maximal erg test, rather than
# derived from the steps. They are session-level values, carried on every step
# record of the test the same way body mass and notes are.
MANUAL_MAX_TEST_TYPES = ("erg_C2", "erg_RP3")
MANUAL_MAX_FIELDS = ("map_w", "vo2max")
MANUAL_MAX_RANGES = {
    "map_w": (100, 1000, "MAP should be between 100 and 1000 W."),
    "vo2max": (1, 10, "VO2max should be between 1 and 10 L/min."),
}


def is_max_mode(mode):
    return str(mode or "").strip().lower().replace(" ", "").replace("-", "") == "max"


def takes_manual_max_results(test_type, mode):
    return test_type in MANUAL_MAX_TEST_TYPES and is_max_mode(mode)


def manual_max_issues(values):
    """Range messages for whichever of map_w / vo2max are filled in."""
    issues = []
    for field, (low, high, message) in MANUAL_MAX_RANGES.items():
        raw = values.get(field)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            issues.append(message)
            continue
        if value < low or value > high:
            issues.append(message)
    return issues


def is_submax_test(mode):
    """Accept current and legacy spellings; unknown intensity is not eligible."""
    return str(mode or "").strip().lower().replace(" ", "").replace("-", "") == "submax"
