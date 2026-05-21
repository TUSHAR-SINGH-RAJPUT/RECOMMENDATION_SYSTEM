from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.collaborative import router as collaborative_router
from routes.embedding import router as embedding_router
from routes.hybrid import router as hybrid_router
from routes.conversational import router as conversational_router

app = FastAPI(
    title="GenAI Recommender API",
    description="Hyper Personalized Recommendation System",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(collaborative_router)
app.include_router(embedding_router)
app.include_router(hybrid_router)
app.include_router(conversational_router)


@app.get("/")
def home():
    return {
        "message": "GenAI Recommender API Running Successfully"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }