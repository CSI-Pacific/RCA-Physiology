"""Configuration, resolved once at import.

Everything that differs between a laptop and a deployment comes from the
environment. Local runs get theirs from a gitignored .env (see .env.example);
a deployment sets real environment variables in its host's settings panel.
load_dotenv() does not override variables that are already set, so the host
always wins over a stray .env that rode along in a build.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env for local work, or set "
            f"{name} in the deployment's environment variables."
        )
    return value


SITE_URL = os.environ.get("SITE_URL", "https://apps.csipacific.ca").rstrip("/")

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
APP_URL = (os.environ.get("APP_URL") or DEPLOYED_APP_URL).strip()

# dash_auth_external builds the redirect URI by appending to this, and the
# registered URI has exactly one slash there.
if not APP_URL.endswith("/"):
    APP_URL += "/"

AUTH_URL = f"{SITE_URL}/o/authorize"
TOKEN_URL = f"{SITE_URL}/o/token/"

# The client id is a public identifier and is fine in the repo. The secret is
# not -- it lives only in the environment.
CLIENT_ID = os.environ.get("CLIENT_ID", "bDf3z9KwxSzCFtxabQ10UwlnHCMl2IsE5teZWLu4")
CLIENT_SECRET = _require("CLIENT_SECRET")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")

SPORT_ORG_ENDPOINT = "/api/registration/organization/"
PROFILE_ENDPOINT = "/api/registration/profile/"

RAW_INGEST_ENDPOINT = "/api/warehouse/ingestion/primary/"

VO2_STEP_SOURCE_UUID = os.environ.get(
    "VO2_STEP_SOURCE_UUID", "144f56a2-f10e-4c4b-bd8a-98afdc025f93"
)
ERG_TEST_SOURCE_UUID = os.environ.get(
    "ERG_TEST_SOURCE_UUID", "992c95a6-86ba-47e8-8bf4-0d67dd1838e4"
)
