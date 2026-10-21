# Deploying to the VPS

**Legacy guide.** The Hetzner dashboard is retired. The current serving app uses
saved snapshots and native models; use [COMMUNITY_CLOUD.md](COMMUNITY_CLOUD.md).
The instructions below preserve the former PostgreSQL/cron hosting context and
are not required for the snapshot dashboard.

Written for you to run by hand on the VPS (no remote access from this
environment). Each step assumes the same reverse-proxy pattern used by other
DAWS projects -- adjust paths/domain to match.

## 1. Get the code + environment onto the VPS

```bash
git clone <your-repo-url> /opt/housing-price-drivers
cd /opt/housing-price-drivers
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p logs data/raw
```

## 2. Configure `.env`

Copy `.env.example` to `.env` and fill in:
- Real Postgres credentials for this project's database (create the DB and
  apply `sql/01_schema.sql` then `sql/02_fct_transactions.sql` once, exactly
  as in local dev -- see `M1_notes.md` "What was built").
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` if you want the cron failure
  alert live (it's safe to leave blank -- `refresh.sh` skips the alert with
  a one-line log notice instead of failing).

## 3. Run the pipeline once manually

```bash
bash scripts/refresh.sh
```

Confirms the DB connection, ingest endpoint, and full pipeline work on the
VPS before anything is automated. If you need the 3-year historical
backfill too, run `ingest.py --seasons 112S3 115S1` then
`load.py --seasons 112S3 115S1` once (see `M1_notes.md` "Backfill"), then
`quality.py` and `sql/03_dim_district.sql` to rebuild from the full history.

## 4. systemd service (keeps the dashboard running, restarts on crash)

```bash
sudo cp deploy/housing-dashboard.service /etc/systemd/system/
# edit WorkingDirectory/ExecStart/User in the unit file if your paths differ
sudo systemctl daemon-reload
sudo systemctl enable --now housing-dashboard
sudo systemctl status housing-dashboard   # confirm it's running
curl -s http://127.0.0.1:8501/_stcore/health   # should print "ok"
```

## 5. Reverse proxy (public link)

Use whichever you already run for other DAWS projects:
- nginx: add the block in `deploy/nginx.conf.snippet` to your existing
  server block, then `sudo nginx -t && sudo systemctl reload nginx`.
- Caddy: add the block in `deploy/Caddyfile.snippet`, then
  `sudo systemctl reload caddy`.

Visit the public URL and confirm all three tabs (Driver Explorer, District
Comparison, Price Estimator) render and the estimator button returns a
range.

## 6. Cron (recurring refresh)

```bash
crontab -e
# paste the line from deploy/crontab.txt, adjusting the path
```

Cadence: data updates on the 1st/11th/21st (verified, see `M0_NOTES.md`);
the cron line runs the day after each (2nd/12th/22nd) to give the batch
time to fully publish.

## 7. Verify the failure alert end-to-end

Locally, the alert mechanism was verified by forcing a real failure (wrong
DB password) and confirming the trap fires and exits non-zero -- see
`M2_notes.md`-style verification in this milestone's notes. That confirmed
the *mechanism*; it did not confirm actual Telegram delivery, since no real
bot token was available. On the VPS, with `TELEGRAM_BOT_TOKEN`/`CHAT_ID`
filled in:

```bash
# temporarily break something to force a failure, e.g.:
mv data/raw/a_lvr_land_a.csv data/raw/a_lvr_land_a.csv.bak 2>/dev/null
bash scripts/refresh.sh   # should fail and a Telegram message should arrive
mv data/raw/a_lvr_land_a.csv.bak data/raw/a_lvr_land_a.csv 2>/dev/null
```

(That particular file is re-downloaded by `ingest.py`'s first step, so it
won't actually trigger a failure -- pick something that will, e.g. a
temporarily wrong `DB_PASSWORD` in `.env`, the same way it was tested
locally. Revert immediately after.)
