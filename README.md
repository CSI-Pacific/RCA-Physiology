# RCA-Physiology

Rowing Canada Step Test analysis. 

Entries include:
- Target Power
- Acutal Power
- Heart Rate
- Lactate
- Time in step
- RPE
- Rate

Outputs include: 
- Heart rate zones

## Configuration

Settings resolve from the environment, so the same commit runs locally and
deployed without editing any file.

**Local**

```bash
cp .env.example .env    # then fill in CLIENT_SECRET
python app.py
```

`.env` is gitignored. It sets `APP_URL=http://127.0.0.1:8050/` so the OAuth
login redirects back to your local server.

**Deployed**

Set these as environment variables in the hosting platform (Posit Connect
Cloud: app → Settings → Variables). Do not deploy a `.env`.

| Variable | Required | Notes |
| --- | --- | --- |
| `CLIENT_SECRET` | yes | OAuth client secret. Never committed. |
| `APP_URL` | no | Defaults to `DEPLOYED_APP_URL` in `settings.py`. Set it when the deployment URL changes, and register that URL as a redirect URI on the OAuth client. |
| `SITE_URL` | no | Warehouse base URL. |
| `FLASK_SECRET_KEY` | recommended | Keeps sessions valid across restarts and workers. |

`APP_URL` must match a redirect URI registered on the OAuth application — the
provider rejects anything else, which is why it is declared rather than read
off the request.
