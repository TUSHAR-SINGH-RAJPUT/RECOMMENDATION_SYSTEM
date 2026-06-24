# RoomSense Frontend

This folder contains the React + Vite user interface for RoomSense.

## Main Responsibilities

- Search furniture products using embedding, collaborative, and hybrid recommendation modes.
- Show product cards, scores, response times, and model comparison metrics.
- Provide a conversational chatbot UI that streams Server-Sent Events from the FastAPI backend.

## Important Files

| Path | Purpose |
| --- | --- |
| `src/main.jsx` | React application bootstrap |
| `src/App.jsx` | Top-level app wrapper |
| `src/pages/Home.jsx` | Main recommendation page |
| `src/api/recommenderApi.js` | API client for backend recommendation endpoints |
| `src/hooks/useRecommendations.js` | Recommendation state management |
| `src/components/Chatbot.jsx` | Streaming conversational UI |
| `src/data/embedding_metadata.json` | Product metadata fallback used by the UI |
| `src/index.css` | Global styles and design tokens |

## Setup

```bash
npm install
```

## Run Locally

```bash
npm run dev
```

Default URL:

```text
http://127.0.0.1:5173
```

The frontend calls the backend at `http://127.0.0.1:8000` by default. To override this:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

PowerShell:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

## Build

```bash
npm run build
```

## Lint

```bash
npm run lint
```

## Backend Dependency

Start the FastAPI app before using recommendation features:

```bash
cd ../BACKEND
python -m uvicorn SPRINT_03.WEEK_06.api.main:app --reload --host 127.0.0.1 --port 8000
```
