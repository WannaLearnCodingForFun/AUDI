# Deploying AUDI in cloud mode (upgrade 6.1)

The edge (drone / ground station) runs the API and the engine against its own SQLite database
and syncs through the outbox. Cloud mode adds the authoritative memory, image storage, the
maintenance portal and monitoring:

```
            HTTPS (Caddy, Let's Encrypt)
 browser ──────────────► web ──/api, /ws──► api  (portal backend, engine, Local DB volume)
                          │                  │ outbox: records + images
                          ├──/cloud────────► cloud ──► Postgres (records)
                          │                         └► MinIO / S3 (observation images)
                          └──/grafana──────► grafana ◄── prometheus ◄── api /metrics
 edge devices ── HTTPS /cloud/push, /cloud/sites/{id}/pull ──► cloud
```

## 1. Configure

```bash
cp .env.example .env                       # fill every secret (see the comments in the file)
python scripts/make_cloud_config.py audi.example.org   # deploy/config.cloud.yaml (cloud overrides)
grep AUDI_METRICS_TOKEN .env | cut -d= -f2 > deploy/metrics_token   # Prometheus bearer token
mkdir -p models && cp /path/to/condition_classifier.pt models/      # trained weights (not in git)
```

Set `AUDI_DOMAIN` in `.env` to the public DNS name; Caddy obtains certificates automatically
(ports 80 and 443 must reach the host). With `AUDI_DOMAIN=localhost` Caddy uses its internal CA.

## 2. Start

```bash
docker compose up -d --build
docker compose exec api python scripts/seed_users.py      # admin / operator / maintenance / sync users
docker compose logs -f api cloud
```

* Portal: `https://<domain>/`  · API: `https://<domain>/api/health`
* Grafana: `https://<domain>/grafana/` (admin / `GRAFANA_ADMIN_PASSWORD`) → AUDI → *AUDI live mission*
* MinIO console is not exposed; use `docker compose exec minio mc …` or publish port 9001 on a private network.

## 3. Connect edge devices

On each edge device set `cloud.base_url: https://<domain>` in `config.yaml`, and the sync
credentials `AUDI_CLOUD_USER` / `AUDI_CLOUD_PASSWORD` (same as in `.env`). The edge works offline
and flushes the outbox — records **and** images (`cloud.sync_images`) — when it reconnects.

Register drones in the portal (*Merge, graph & drones*) to get per-drone API keys with scopes;
uploads to `/api/drone/frames` are HMAC-signed (see `api/routers/drones.py`).

## 4. Operations

| Task | Command |
|---|---|
| Backup Postgres | `docker compose exec db pg_dump -U audi audi \| gzip > audi_$(date +%F).sql.gz` |
| Backup images | `docker compose exec minio-init mc mirror m/audi-images /backup/audi-images` |
| Monthly owner report | `docker compose exec api python scripts/owner_report.py --site <site> --month 2026-10` |
| Rotate the image key | images are sealed with Fernet; keep the old key until every image is re-sealed |
| Upgrade | `git pull && docker compose up -d --build` (the Local DB migrates new columns automatically) |

## Security checklist (6.5)

Reviewed against the OWASP API Security Top 10 (2023):

| Risk | Control in AUDI |
|---|---|
| API1 broken object-level authorisation | drones may act only on their own sites / sessions; maintenance only receives escalations for subscribed sites |
| API2 broken authentication | bcrypt passwords, short-lived JWTs, per-drone API keys stored as SHA-256, revocable |
| API3 property-level authorisation | API keys returned once; `password_hash` / `api_key_hash` never serialised |
| API4 unrestricted resource consumption | token-bucket rate limiting per client, upload size limit (`api.max_upload_mb`) |
| API5 function-level authorisation | role guards on every route; `tests/test_api.py::test_every_route_requires_auth` covers all routers |
| API6 sensitive business flows | merges / splits / relations / drone keys are admin-only and written to the audit log |
| API7 SSRF | the API fetches no user-supplied URLs (site-map paths are local files, admin-only) |
| API8 misconfiguration | strict CORS list, security headers and TLS in Caddy, secrets only from env / mode-600 files |
| API9 inventory | all routes in `api/app.py:ROUTERS`; OpenAPI at `/docs` |
| API10 unsafe consumption | cloud pulls are validated by the ORM models; images are decoded with Pillow before use |

Images are encrypted at rest when `security.encrypt_images` is on (enabled in cloud mode); the
S3 bucket additionally requests SSE-S3.
