from fastapi import APIRouter, HTTPException
from SPRINT_03.WEEK_06.api.schemas import RecommendationRequest
from SPRINT_03.WEEK_06.api.services.recommendation_service import (
    get_hybrid_recommendations,
)

router = APIRouter(
    prefix="/recommend/hybrid",
    tags=["Hybrid Recommendations"]
)

@router.post("/")
def hybrid_recommend(request: RecommendationRequest):
    """Hybrid endpoint combining semantic retrieval, CF, and LTR-style ranking."""

    try:
        recommendations = get_hybrid_recommendations(
            user_id=request.user_id,
            query=request.query,
            top_k=request.top_k,
            category=request.category,
            use_xgboost=request.use_xgboost,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid recommendation failed: {exc}",
        ) from exc

    return {
        "model": "Hybrid Recommendation System",
        "user_id": request.user_id,
        "query": request.query,
        "count": len(recommendations),
        "recommendations": recommendations
    }
