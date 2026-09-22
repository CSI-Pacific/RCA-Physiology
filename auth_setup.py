from dash_auth_external import DashAuthExternal
from settings import AUTH_URL, TOKEN_URL, APP_URL, CLIENT_ID, CLIENT_SECRET, FLASK_SECRET_KEY


stable_secret_key = FLASK_SECRET_KEY or CLIENT_SECRET

auth = DashAuthExternal(
    AUTH_URL,
    TOKEN_URL,
    app_url=APP_URL,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    _secret_key=stable_secret_key,
)
server = auth.server  # expose the Flask server for app.py
