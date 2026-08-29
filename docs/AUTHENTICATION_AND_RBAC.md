# Authentication and Role-Based Access Control

## Security boundary

Every `/v1` API request requires an opaque bearer credential. The backend stores
only configured SHA-256 token digests in memory and compares them in constant
time. Plaintext credentials belong in a secret manager or process environment;
they are never committed to the repository or persisted in the database.

Configure principals before starting the API:

```powershell
$token = "replace-with-a-long-random-secret"
$digest = [Convert]::ToHexString(
  [Security.Cryptography.SHA256]::HashData(
    [Text.Encoding]::UTF8.GetBytes($token)
  )
).ToLower()
$env:AUTH_PRINCIPALS_JSON = '[{"subject":"replay-service","role":"operator","token_sha256":"' + $digest + '"}]'
```

Requests use `Authorization: Bearer <token>`. `GET /v1/auth/me` returns the
authenticated subject and effective role. Invalid or absent credentials return
HTTP 401; insufficient roles return HTTP 403. An empty configuration denies all
requests by design.

## Roles

| Capability | Viewer | Operator | Admin |
|---|:---:|:---:|:---:|
| Read assets, telemetry, predictions, alerts, and work orders | Yes | Yes | Yes |
| Submit telemetry and direct prediction windows | No | Yes | Yes |
| Advance alert and work-order state | No | Yes | Yes |
| Register or rename assets | No | No | Yes |

Operator and admin workflow mutations persist the authenticated subject in
`last_updated_by`. Alembic revision `0003` adds these audit fields.

## Replay and console

Set `RELIABILITY_API_TOKEN` or pass `--token` when running the adjustable replay
client. The operations console accepts an issued credential and retains it only
in the current browser tab (`sessionStorage`). Production deployments should
deliver credentials out of band and rotate them through the environment-backed
principal configuration.

This phase intentionally uses provider-neutral opaque credentials. It secures
the current API and automation clients without locking the platform to a human
identity provider; enterprise SSO and short-lived token issuance can replace
credential verification later while preserving the role dependencies.

AUTHENTICATION AND RBAC STATUS: COMPLETE
