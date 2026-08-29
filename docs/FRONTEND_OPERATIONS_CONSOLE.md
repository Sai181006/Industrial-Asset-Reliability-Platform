# Frontend Operations Console

## Status

The responsive operations console provides one prioritized view of fleet risk,
predicted remaining useful life, active alerts, and maintenance work orders. It
uses the existing FastAPI asset-detail endpoints and does not reproduce model or
decision logic in the browser.

When the API is reachable and a valid bearer credential is supplied, the console
loads every registered asset and its current persisted state. The credential is
kept only in the current browser tab. When the API is unavailable or rejects the
credential, the screen uses an explicitly
identified FD001 demonstration snapshot so the interface remains reviewable
without presenting sample values as live data.

## Local operation

Start the backend from the repository root, then start the frontend:

```powershell
uvicorn backend.app.main:app --reload
cd frontend
npm install
npm run dev
```

The default API address is `http://127.0.0.1:8000`. Copy `.env.example` to a
local `.env` and set `NEXT_PUBLIC_API_BASE_URL` when the backend uses another
origin. The API accepts local console requests by default; production frontend
origins must be supplied as a comma-separated `FRONTEND_ORIGINS` value before
backend startup.

## Product boundary

This slice remains read-only. Authentication and actor identity are now present
in the API; alert acknowledgement, work-order assignment, and other maintenance
mutations are the next dedicated console workflow slice.

FRONTEND OPERATIONS CONSOLE STATUS: COMPLETE
