# Frontend

React/Vite/Tailwind frontend for the ArXiv RAG Assistant. The app lives in `frontend/` and is intended for Vercel deployment.

## Runtime Behavior

- Authenticated chat uses REST through `src/api.js` and `/conversations/{id}/query`.
- The backend exposes SSE at `/query/stream`, but the main React chat does not currently stream tokens.
- Set `VITE_API_URL` to the deployed FastAPI backend URL in Vercel.

## Local Run

```bash
npm install
npm run dev
```
