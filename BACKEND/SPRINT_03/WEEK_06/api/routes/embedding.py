from fastapi import APIRouter, HTTPException

from SPRINT_03.WEEK_06.api.schemas import SearchProductsRequest
from SPRINT_03.WEEK_06.api.services.recommendation_service import (
    get_embedding_recommendations,
)


# ============================================================
# ROUTER CONFIGURATION
# ============================================================
# Creates a router for embedding-based recommendation endpoints
#
# prefix:
#   All routes in this file will start with:
#   /recommend/embedding
#
# tags:
#   Groups endpoints together in FastAPI Swagger docs
# ============================================================
router = APIRouter(
    prefix="/recommend/embedding",
    tags=["Embedding Recommendations"]
)


# ============================================================
# EMBEDDING RECOMMENDATION ENDPOINT
# ============================================================
# POST /recommend/embedding/
#
# This endpoint:
#   1. Accepts a user query
#   2. Converts query into embeddings
#   3. Searches similar products using vector similarity
#   4. Returns top matching products
#
# Request Body:
# {
#     "query": "modern wooden sofa",
#     "top_k": 5,
#     "category": "Living Room"
# }
#
# Returns:
# {
#     "model": "Embedding Based Recommendation",
#     "query": "...",
#     "recommendations": [...]
# }
# ============================================================
@router.post("/")
def embedding_recommend(request: SearchProductsRequest):
    """Semantic search endpoint backed by sentence embeddings and ChromaDB."""

    try:
        recommendations = get_embedding_recommendations(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding recommendation failed: {exc}",
        ) from exc

    return {
        "model": "Embedding Based Recommendation",
        "query": request.query,
        "count": len(recommendations),
        "recommendations": recommendations,
    }
