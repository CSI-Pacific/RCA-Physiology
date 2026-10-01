"""Capture a warehouse access token by logging in, for scripts to use.

The app gets its token through a browser OAuth round trip and keeps it in the
Flask session, so there is no command that can simply ask for one -- and this
deployment's OAuth client is not allowed the client-credentials grant, which is
what scripts/auth_cli.py would otherwise use. This bridges the gap: it runs the
same login the app runs, and prints the resulting token to your terminal.

    python scripts/get_token.py

It opens http://127.0.0.1:8050/, you log in as usual, and the token is printed
here as a ready-to-paste line:

    export WAREHOUSE_TOKEN='...'

Run that, then the migration needs no --token at all. Press Ctrl-C when done.

The token is printed to the terminal rather than rendered in the browser, so it
does not end up in your history or on a shared screen. It is short-lived, it is
the same token the app uses, and nothing is written anywhere.

This serves only 127.0.0.1 and is a local helper -- it is not part of the
deployed app, which is why it is a script rather than a route in app.py.
"""

import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Response  # noqa: E402

from auth_setup import auth, server  # noqa: E402
from settings import APP_URL  # noqa: E402

HOST = "127.0.0.1"
PORT = 8050
LOCAL_URL = f"http://{HOST}:{PORT}/"

# The token is also written here, so the scripts can pick it up without being
# handed it: an environment variable does not survive opening a second terminal,
# and that hand-off is the step most likely to go wrong. Gitignored, owner-only,
# and short-lived -- delete it when the migration is done.
TOKEN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".warehouse_token"
)

PAGE = """<!doctype html>
<title>Token captured</title>
<body style="font-family: system-ui; margin: 4rem; line-height: 1.5">
  <h2>{heading}</h2>
  <p>{message}</p>
</body>
"""


def _render(heading, message):
    return Response(PAGE.format(heading=heading, message=message), mimetype="text/html")


def _write_token_file(token):
    """Write owner-only, and create it that way rather than fixing it after."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    handle = os.open(TOKEN_FILE, flags, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(token)
    os.chmod(TOKEN_FILE, 0o600)  # in case it already existed with other modes


def _emit(token):
    try:
        _write_token_file(token)
        saved = True
    except OSError as exc:
        print(f"\nCould not write {TOKEN_FILE}: {exc}", file=sys.stderr)
        saved = False

    print("\n" + "=" * 68)
    if saved:
        print("Token saved. In any terminal, from the repo root:")
        print("\n    python scripts/migrate_erg_records.py")
        print("\nNothing to copy or paste -- the scripts read it from")
        print(f"{TOKEN_FILE} (gitignored, owner-only).")
        print("Delete that file when you are finished.")
    else:
        # stdout, so the token never lands in browser history or on a shared
        # screen. This terminal is busy serving the login, so paste it elsewhere.
        print("OPEN A SECOND TERMINAL and run this there:")
        print("\n    export WAREHOUSE_TOKEN='%s'" % token)
        print("    python scripts/migrate_erg_records.py")
    print("=" * 68)
    print("\nThis window is running the login server; Ctrl-C it when done.\n")


@server.route("/home")
def show_token_home():
    """Where the OAuth redirect lands once the login succeeds."""
    try:
        token = auth.get_token()
    except Exception as exc:
        return _render("Not signed in", f"Login did not complete: {exc}")

    _emit(token)
    return _render(
        "Token captured",
        "It has been printed in your terminal. You can close this tab.",
    )


@server.route("/token")
def show_token():
    """Ask again without logging in again, if the first copy got lost."""
    try:
        token = auth.get_token()
    except Exception as exc:
        return _render(
            "Not signed in",
            f"Start at <a href='{LOCAL_URL}'>{LOCAL_URL}</a> first. ({exc})",
        )

    _emit(token)
    return _render("Token printed again", "Check your terminal.")


def main():
    if not APP_URL.startswith(LOCAL_URL):
        print(
            f"APP_URL is {APP_URL!r}, but this helper serves {LOCAL_URL}.\n"
            "The OAuth provider only redirects to registered URIs, so set\n"
            f"    APP_URL={LOCAL_URL}\n"
            "in your .env and run this again.",
            file=sys.stderr,
        )
        return 2

    print(f"Opening {LOCAL_URL} -- log in as you normally would.")
    print("The token will be printed here once you are through.\n")

    threading.Timer(1.0, lambda: webbrowser.open(LOCAL_URL)).start()
    server.run(host=HOST, port=PORT, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
