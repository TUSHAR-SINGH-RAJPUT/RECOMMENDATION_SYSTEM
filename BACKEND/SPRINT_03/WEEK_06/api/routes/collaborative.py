from fastapi import APIRouter, HTTPException

from SPRINT_03.WEEK_06.api.schemas import RecommendationRequest
from SPRINT_03.WEEK_06.api.services.recommendation_service import (
    get_collaborative_recommendations,
)

router = APIRouter(
    prefix="/recommend/collaborative",
    tags=["Collaborative Filtering"]
)


@router.post("/")
def collaborative_recommend(request: RecommendationRequest):
    """Collaborative filtering endpoint using user-user interaction similarity."""

    try:
        recommendations = get_collaborative_recommendations(
            user_id=request.user_id,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Collaborative recommendation failed: {exc}",
        ) from exc

    return {
        "model": "Collaborative Filtering",
        "user_id": request.user_id,
        "query": request.query,
        "count": len(recommendations),
        "recommendations": recommendations
    }
