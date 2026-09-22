"""Write templates/*.csv from the definitions the app's download buttons use.

Run after changing bulk_templates.py so the files anyone emails to a
practitioner stay identical to what the Entry page hands out:

    python scripts/generate_templates.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bulk_templates import erg_template_csv, step_template_csv  # noqa: E402

TEMPLATES = {
    "step_test_bulk_template.csv": step_template_csv,
    "erg_bulk_template.csv": erg_template_csv,
}


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "templates")
    os.makedirs(out_dir, exist_ok=True)

    for filename, render in TEMPLATES.items():
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(render())
        print(f"wrote {os.path.relpath(path, here)}")


if __name__ == "__main__":
    main()
