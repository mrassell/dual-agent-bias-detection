# Frontend (React + Vite)

This frontend is intentionally independent from the current backend so you can deploy UI now.

## Local development

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Connect to backend later

1. Copy `.env.example` to `.env`.
2. Set:

```bash
VITE_API_BASE_URL=https://your-future-api-domain.com
```

If `VITE_API_BASE_URL` is unset, the app runs in demo mode and scores text locally.

## Railway deploy (frontend only)

Create a Railway service from this repo and set the **Root Directory** to `frontend`.

- Build command: `npm run build`
- Start command: `npm run start`

Railway provides `PORT` automatically.
