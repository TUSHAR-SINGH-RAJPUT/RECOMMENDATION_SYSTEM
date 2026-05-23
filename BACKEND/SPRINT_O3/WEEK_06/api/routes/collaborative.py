from fastapi import APIRouter

from SPRINT_O3.WEEK_06.api.schemas import RecommendationRequest

from SPRINT_01.WEEK_02.models.collaborative_filter import recommend_cf

router = APIRouter(
    prefix="/recommend/collaborative",
    tags=["Collaborative Filtering"]
)


@router.post("/")
def collaborative_recommend(request: RecommendationRequest):

    recommendations = recommend_cf(
        request.user_id,
        request.query,
        request.top_k
    )

    return {
        "model": "Collaborative Filtering",
        "user_id": request.user_id,
        "recommendations": recommendations
    }