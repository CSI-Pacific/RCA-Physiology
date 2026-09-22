"""Configuration, resolved once at import.

Everything that differs between a laptop and a deployment comes from the
environment. Local runs get theirs from a gitignored .env (see .env.example);
a deployment sets real environment variables in its host's settings panel.
load_dotenv() does not override variables that are already set, so the host
always wins over a stray .env that rode along in a build.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _env(name, default=None):
    """An environment variable, treating empty and whitespace-only as unset.

    os.environ.get(name, default) returns "" when the variable exists but is
    blank, which a hosting panel makes easy to do by accident -- and a blank
    client id reaches the OAuth provider as "Invalid client_id parameter
    value" rather than as a missing-configuration error. Falling back to the
    default keeps a blank variable from being worse than no variable.
    """
    value = os.environ.get(name)
    value = value.strip() if value else ""
    return value or default


SITE_URL = _env("SITE_URL", "https://apps.csipacific.ca").rstrip("/")

# Where this app is served. The OAuth provider redirects the practitioner back
# here after login, and it only honours redirect URIs registered against the
# client -- so this cannot be read off the incoming request, it has to be
# declared.
#
# The default is the DEPLOYED url, not localhost, on purpose: a deploy that
# forgets to set APP_URL still sends people somewhere real, whereas the other
# way round every deploy silently bounced its users to 127.0.0.1 and the only
# symptom was a login that never came back. Local work overrides it in .env.
DEPLOYED_APP_URL = "https://019c390a-d5fb-ead7-0df0-118fba4280e6.share.connect.posit.cloud/"
APP_URL = _env("APP_URL", DEPLOYED_APP_URL)

# dash_auth_external builds the redirect URI by appending to this, and the
# registered URI has exactly one slash there.
if not APP_URL.endswith("/"):
    APP_URL += "/"

AUTH_URL = f"{SITE_URL}/o/authorize"
TOKEN_URL = f"{SITE_URL}/o/token/"

# Credentials for the OAuth application registered at SITE_URL.
#
# The client id is a public identifier, so it is a literal here and the
# deployment needs no variable for it. Do not set CLIENT_ID in a hosting
# panel: a blank or mistyped value there reaches the provider as "Invalid
# client_id parameter value", which is far harder to read than this line.
CLIENT_ID =  "bDf3z9KwxSzCFtxabQ10UwlnHCMl2IsE5teZWLu4"

# The secret is the one value this repo cannot carry -- the repo is public.
# Set CLIENT_SECRET in the deployment's environment variables.
#
# To run the deployment with no configuration at all instead, paste the
# secret as the second argument below. It then ships to GitHub in the clear
# on the next push, so only do that with a secret you are willing to treat as
# public, and rotate it if that stops being true.
CLIENT_SECRET =  "em7L8NeqjKP8vxTEYRz7LrnHKz7aU8pm7t0DfbCiyQkljgz2YEyf7j2wCfWuN3m21QfKehzAwkwBc8boXGYSOJWFm6PAif4iHQ3kbT5xZ5safDeBlt03YDgqr5EhooYR"
FLASK_SECRET_KEY = _env("FLASK_SECRET_KEY")

SPORT_ORG_ENDPOINT = "/api/registration/organization/"
PROFILE_ENDPOINT = "/api/registration/profile/"

RAW_INGEST_ENDPOINT = "/api/warehouse/ingestion/primary/"

VO2_STEP_SOURCE_UUID = _env(
    "VO2_STEP_SOURCE_UUID", "144f56a2-f10e-4c4b-bd8a-98afdc025f93"
)
ERG_TEST_SOURCE_UUID = _env(
    "ERG_TEST_SOURCE_UUID", "992c95a6-86ba-47e8-8bf4-0d67dd1838e4"
)


# One line in the startup log so a misconfigured deployment is identifiable
# without guessing. No secret is printed -- the client id is a public
# identifier and only its tail is shown, enough to tell two of them apart.
print(
    f"[settings] APP_URL={APP_URL} SITE_URL={SITE_URL} "
    f"client_id=...{CLIENT_ID[-6:]} client_secret={'set' if CLIENT_SECRET else 'MISSING'}",
    file=sys.stderr,
)
