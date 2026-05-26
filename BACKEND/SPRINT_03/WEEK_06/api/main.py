from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# IMPORT API ROUTERS
# ============================================================
# Each router handles a specific recommendation module
#
# collaborative_router  -> Collaborative Filtering APIs
# embedding_router      -> Embedding / Content-Based APIs
# hybrid_router         -> Hybrid Recommendation APIs
# conversational_router -> Conversational AI APIs
# ============================================================

from SPRINT_03.WEEK_06.api.routes.collaborative import (
    router as collaborative_router
)

from SPRINT_03.WEEK_06.api.routes.embedding import (
    router as embedding_router
)

from SPRINT_03.WEEK_06.api.routes.hybrid import (
    router as hybrid_router
)

from SPRINT_03.WEEK_06.api.routes.conversational import (
    router as conversational_router
)


# ============================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================
# title:
#   API title shown in Swagger docs
#
# description:
#   Brief overview of the project
#
# version:
#   Current API version
# ============================================================
app = FastAPI(
    title="RoomSense Recommender API",
    description="Hyper-personalized furniture recommendation system",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================
# CORS = Cross-Origin Resource Sharing
#
# Allows frontend applications (React, Next.js, etc.)
# to communicate with this backend API.
#
# allow_origins=["*"]
#   Allows requests from all domains
#
# allow_methods=["*"]
#   Allows all HTTP methods (GET, POST, PUT, DELETE)
#
# allow_headers=["*"]
#   Allows all headers
#
# NOTE:
# In production, replace "*" with actual frontend URL
# for better security.
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REGISTER API ROUTES
# ============================================================
# Adds all routers into the FastAPI application
#
# Available endpoints:
#
# /recommend/collaborative
# /recommend/embedding
# /recommend/hybrid
# /recommend/conversational
# ============================================================
app.include_router(collaborative_router)
app.include_router(embedding_router)
app.include_router(hybrid_router)
app.include_router(conversational_router)


# ============================================================
# HOME ENDPOINT
# ============================================================
# GET /
#
# Basic endpoint used to verify that the API is running
# ============================================================
@app.get("/")
def home():

    return {
        "message": "RoomSense Recommender API Running Successfully"
    }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================
# GET /health
#
# Used for:
#   - Monitoring
#   - Deployment health checks
#   - Docker/Kubernetes readiness checks
# ============================================================
@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }
