import re
import requests
from settings import SITE_URL

def fetch_options(path, token, label_key, value_key, params=None, limit=1000):
    headers = {"Authorization": f"Bearer {token}"}

    if params and isinstance(params,dict):
        params.update({"limit": limit})
    else:
        params = {"limit": limit}

    resp = requests.get(f"{SITE_URL}{path}", params=params, headers=headers, timeout=5)
    resp.raise_for_status()

    # print("=========================")
    # print(path)

    items = resp.json()

    # print(items)

    if 'results' in items:
        rv = [{"label": item[label_key], "value": item[value_key]} for item in items["results"]]
    elif isinstance(items, list):
        rv = [{"label": val, "value": val} for val in items if val]
    else:
        rv = []

    return rv

def fetch_profile(token, profile_id):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{SITE_URL}/api/registration/profile/{profile_id}"
    params = {}  # choose a reasonable chunk size
    all_records = []

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    payload = r.json()

    return payload

def fetch_profiles(token, filters):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{SITE_URL}/api/registration/profile/"
    params = {**filters, "limit": 100, "offset": 0}  # choose a reasonable chunk size
    all_records = []

    while url:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        payload = r.json()
        all_records.extend(payload["results"])

        # Move to the next page
        url = payload.get("next")
        # Once we switch to using `next`, we no longer need `params`
        params = None

    return all_records

def restructure_profile(profile, format='profile'):
    if not format:
        format = 'profile'


    if format == 'profile':
        record = {
            'role': profile['role_slug'] if profile['role_slug'] else None,
            'first_name': profile['person']['first_name'] if profile['person'] else None,
            'last_name': profile['person']['last_name'] if profile['person'] else None,
            'email': profile['person']['email'] if profile['person'] else None,
            'sport': profile['sport']['name'] if profile['sport'] else None,
            'org':None,
            'dob' :profile['person']['dob'] if profile['person'] else None,
            'majority_age': profile['person']['majority_age'] if profile['person'] else None,
            # 'enrollment_status': profile['current_enrollment']['enrollment_status'] if profile[
            #     'current_enrollment'] else None

            'birthplace':f"{profile['birth_city']['name_ascii']}, {profile['birth_city']['province_territory']}" if 'birth_city' in profile and profile['birth_city'] else None,
            'residence': f"{profile['residence_city']['name_ascii']}, {profile['residence_city']['province_territory']}" if 'residence_city' in profile and profile['residence_city'] else None,

            'enrollment_expiry': profile['current_enrollment']['end_date'] if profile[
                'current_enrollment'] else None
        }
    elif format == 'contact':
        record = {
            'role': profile['role_slug'] if profile['role_slug'] else None,
            'first_name': profile['person']['first_name'] if profile['person'] else None,
            'last_name': profile['person']['last_name'] if profile['person'] else None,
            'email': profile['person']['email'] if profile['person'] else None,
            'sport': profile['sport']['name'] if profile['sport'] else None,
            'org': None,
            'dob': profile['person']['dob'] if profile['person'] else None,
            'majority_age': profile['person']['majority_age'] if profile['person'] else None,
            'guardian': f"{profile['person']['guardian']['first_name']} {profile['person']['guardian']['last_name']}" if profile['person']['guardian'] else None,
            'guardian_email': profile['person']['guardian']['email'] if profile['person']['guardian'] else None,
            'emergency_contact': f"{profile['person']['emergency_contact']['first_name']} {profile['person']['emergency_contact']['last_name']} ({profile['person']['emergency_contact']['relationship']})" if profile['person']['emergency_contact'] else None,
            'emergency_contact_phone': profile['person']['emergency_contact']['phone_number'] if profile['person']['emergency_contact'] else None,
        }
    elif format == 'social':
        record = {
            'role': profile['role_slug'] if profile['role_slug'] else None,
            'first_name': profile['person']['first_name'] if profile['person'] else None,
            'last_name': profile['person']['last_name'] if profile['person'] else None,
            'email': profile['person']['email'] if profile['person'] else None,
            'sport': profile['sport']['name'] if profile['sport'] else None,
            'org':None,
        }

        if profile['person']['social_media_accounts']:
            for act in profile['person']['social_media_accounts']:
                record[act['platform']] = act['username']

    if profile['role_slug'] == 'staff':
        record['org'] = profile['organization']['name'] if profile['organization'] else None
    else:
        record['org'] = profile['current_nomination']['organization']['name'] if profile['current_nomination'] else None

    return record

CSV_ENCODINGS = ("utf-8-sig", "cp1252", "mac_roman", "latin-1")

# Characters that legitimate spreadsheet text almost never contains. Used to
# tell cp1252 and mac_roman apart, since both decode any byte without error but
# only one of them gets accented names right.
_IMPLAUSIBLE_CHARS = re.compile(r"[\u0080-\u009f\u00a0\u00ad\ufffd\u017d\u017e\u02c6-\u02dd\u2020-\u2026]")


def decode_csv_bytes(raw, encodings=CSV_ENCODINGS):
    """Decode CSV bytes, falling back through common spreadsheet encodings.

    Excel exports (especially from Windows or older Macs) are rarely UTF-8, so a
    strict utf-8 decode blows up on accented characters. utf-8 wins whenever it
    is valid; otherwise the single-byte candidates are scored and the one
    producing the fewest implausible characters is returned.
    """
    best = None
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding.startswith("utf-8"):
            return text
        score = len(_IMPLAUSIBLE_CHARS.findall(text))
        if best is None or score < best[0]:
            best = (score, text)
        if score == 0:
            break
    if best is None:
        raise ValueError("Could not decode the uploaded CSV; try saving it as UTF-8.")
    return best[1]
