# Bulk upload templates

Two CSV templates for loading a whole squad's testing into the warehouse in one
go, instead of typing each athlete into the Entry page by hand.

You can always get the current version of either file from the app itself —
**Download CSV Template** on the relevant tab — which is the same file as the
one here.

| Template | Use it for | Where it goes |
| --- | --- | --- |
| `step_test_bulk_template.csv` | Step tests (many athletes, many steps each) | Entry → **Bulk Step Upload** tab |
| `erg_bulk_template.csv` | 2000 m / 6000 m erg results | Entry → **Erg Test** tab |

## How to fill them in

1. Download the template and **delete the example athletes** (`Example Athlete`
   and `Sample Athlete-Two`). They are only there to show the shape of the data.
2. Put each athlete's name in the `athlete` column as **first name, space, last
   name** — `Jane Smith` — exactly as it reads in the athlete dropdown on the
   Step Test tab. See [Writing the name](#writing-the-name) below.
3. Dates go in as `YYYY-MM-DD` (e.g. `2026-09-15`).
4. Leave anything you didn't measure blank. Only the athlete, the test date and
   the step number are genuinely required.

### Writing the name

| In your CSV | Works? |
| --- | --- |
| `Jane Smith` | Yes — use this |
| `Smith, Jane` | Yes — with a comma, the two halves get swapped for you |
| `jane   smith` | Yes — capitalisation and extra spaces are ignored |
| `Smith Jane` | **No** — without a comma there's no way to tell the order |

Hyphens and apostrophes in a surname are fine. Spelling still has to match the
dropdown, so a nickname or an initial in place of a first name won't be found.

If a name won't match no matter what, put the warehouse profile ID in the
`profile_id` column and leave `athlete` blank — the ID always wins.

A name that can't be matched is **reported, never guessed at**: the upload is
rejected and tells you which names failed, so a misspelling can't quietly file a
test against the wrong athlete. That also means forgetting to delete the example
rows is safe — the upload just stops and names them.

### Step test template specifically

One row per step. The athlete, date, body mass, test type, mode and notes repeat
on **every** step row of that test — fill them in on the first row and drag down.
Rows are grouped into one test session by athlete, date, test type and notes, so
an athlete who did two different tests on one day just needs a different test
type or a different note on the second one.

- `test_type` is one of `erg_C2`, `erg_RP3`, `row`, `bike`, `other`. Friendly
  spellings like `Erg C2`, `C2 erg` and `On-Water` are understood too.
- `mode` and `step_type` are `Max` or `Submax`. `mode` describes the test,
  `step_type` describes the individual step — so a ramp that finishes with a max
  effort is `mode = Submax` with `step_type = Max` on the last row only. Leave
  `step_type` blank and it follows `mode`.
- Don't add a split column; split per 500 m is calculated from actual power, the
  same way the single-athlete form does it.

### Erg template specifically

One row per athlete per distance, so an athlete who did both a 2 k and a 6 k gets
two rows. `distance_m` is `2000` or `6000`.

Times are **whole minutes plus the leftover seconds**: a 7:12.4 2 k is
`time_min = 7`, `time_s = 12.4` — not `7` and `432.4`. If your sheet has a single
`time` column written as `7:12.4`, that works too; use it instead of the two.

If you already keep a wide sheet with one row per athlete and columns like
`2000m Erg Power` and `6000m Erg Rate`, upload that directly instead; it gets
split into one row per distance for you.

## What happens after you upload

Neither tab sends anything to the warehouse on upload. You get a preview first —
sessions found for the step test tab, an editable table of rows for the erg tab —
and nothing is written until you press **Push to Warehouse**.

If anything in the file can't be read, the whole upload is rejected and you get a
list of every problem with the spreadsheet line number, rather than a half-loaded
import to unpick. Fix them all and upload again.

## Regenerating these files

The column lists live in `bulk_templates.py`, shared by the app's download
buttons and by this folder. After editing it, run:

```
python scripts/generate_templates.py
```
