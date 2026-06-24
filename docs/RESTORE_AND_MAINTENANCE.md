# Restore and Maintenance Guide

This guide is written for future use after the local copy has been deleted. It explains how to restore the project from GitHub, recreate local dependencies, run the application, and understand which files are important.

## 1. Restore From GitHub

```bash
git clone <your-github-repo-url>
cd <repo-name>
```

Check that these important folders exist:

```text
BACKEND/
FRONTEND/
docs/
run_all.sh
README.md
```

## 2. Recreate Backend Environment

The Python virtual environment is not stored in GitHub. Recreate it:

```bash
cd BACKEND
python -m venv myenv
myenv\Scripts\activate
pip install -r requirement.txt
```

If you are using Git Bash instead of PowerShell:

```bash
source myenv/Scripts/activate
```

## 3. Recreate Frontend Environment

```bash
cd FRONTEND
npm install
```

## 4. Run the Full App

From the repository root:

```bash
bash run_all.sh
```

Expected URLs:

```text
Frontend: http://127.0.0.1:5173
API docs: http://127.0.0.1:8000/docs
Health:   http://127.0.0.1:8000/health
```

## 5. Run Backend and Frontend Separately

Backend:

```bash
cd BACKEND
myenv\Scripts\activate
python -m uvicorn SPRINT_03.WEEK_06.api.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd FRONTEND
npm run dev
```

## 6. Optional Chat Setup

The conversational route uses a local Ollama model. If you only need normal recommendations, this step is optional.

```bash
ollama pull mistral
ollama serve
```

Environment variables:

```text
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=mistral
CHAT_MEMORY_TURNS=8
```

## 7. Critical Files to Preserve in GitHub

These files make the recommendation demos work without rerunning every sprint notebook:

| Path | Purpose |
| --- | --- |
| `BACKEND/SPRINT_01/WEEK_02/data/user_item_matrix.csv` | Collaborative filtering matrix |
| `BACKEND/SPRINT_01/WEEK_02/data/processed_user_behavior.csv` | Processed interaction data |
| `BACKEND/SPRINT_02/WEEK_03/embeddings/embedding_metadata.json` | Product metadata |
| `BACKEND/SPRINT_02/WEEK_03/embeddings/product_embeddings.npy` | Product embeddings |
| `BACKEND/SPRINT_02/WEEK_03/vector_db/chroma_db/` | ChromaDB vector store |
| `BACKEND/SPRINT_02/WEEK_04/Models/Model_checkpoints/xgb_ranker.pkl` | XGBoost ranker checkpoint |
| `FRONTEND/src/data/embedding_metadata.json` | Frontend fallback product metadata |

If GitHub rejects large files, move the large artifacts to Git LFS or attach them to a GitHub Release, then document the download location here.

## 8. What Not to Commit

These are generated locally and should remain ignored:

```text
BACKEND/myenv/
FRONTEND/node_modules/
FRONTEND/dist/
__pycache__/
.env
*.log
```

## 9. Project Architecture Notes

```text
FRONTEND/src/pages/Home.jsx
  Uses recommendation hook and UI components.

FRONTEND/src/api/recommenderApi.js
  Sends requests to FastAPI and handles streaming chat events.

BACKEND/SPRINT_03/WEEK_06/api/main.py
  Creates the FastAPI app and registers all routes.

BACKEND/SPRINT_03/WEEK_06/api/routes/
  Defines API endpoints for embedding, collaborative, hybrid, and conversational modes.

BACKEND/SPRINT_03/WEEK_06/api/services/recommendation_service.py
  Normalizes outputs from older sprint models into one product response format.

BACKEND/SPRINT_03/WEEK_06/api/services/conversation_service.py
  Handles product intent detection, product retrieval, prompt building, and SSE streaming.
```

## 10. API Smoke Tests

Run these after starting the backend.

Health:

```bash
curl http://127.0.0.1:8000/health
```

Embedding:

```bash
curl -X POST http://127.0.0.1:8000/recommend/embedding/ \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"modern wooden chair\",\"top_k\":5}"
```

Hybrid:

```bash
curl -X POST http://127.0.0.1:8000/recommend/hybrid/ \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":1,\"query\":\"comfortable chair\",\"top_k\":5,\"use_xgboost\":true}"
```

On PowerShell, use `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

## 11. Suggested GitHub Push Checklist

Before deleting the local project:

1. Run `git status` and confirm all documentation/source changes are intentional.
2. Run `npm run build` inside `FRONTEND`.
3. Start the backend and open `http://127.0.0.1:8000/docs`.
4. Push to GitHub.
5. Open the GitHub repository page and verify the README renders correctly.
6. Confirm all important artifact files are visible in the GitHub file browser or stored in the documented Git LFS/release location.
7. Only delete the local folder after a fresh clone test works.

## 12. Fresh Clone Test

Use a different temporary folder:

```bash
git clone <your-github-repo-url> roomsense-restore-test
cd roomsense-restore-test
cd BACKEND
python -m venv myenv
myenv\Scripts\activate
pip install -r requirement.txt
cd ..\FRONTEND
npm install
npm run build
```

If the frontend builds and the backend starts, the GitHub copy is safe for future restoration.
