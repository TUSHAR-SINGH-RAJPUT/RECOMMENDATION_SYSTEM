from fastapi import APIRouter
from SPRINT_03.WEEK_06.api.schemas import RecommendationRequest
from SPRINT_02.WEEK_04.Models.hybrid_recommender import hybrid_recommender

router = APIRouter(
    prefix="/recommend/hybrid",
    tags=["Hybrid Recommendations"]
)

@router.post("/")
def hybrid_recommend(request: RecommendationRequest):

    recommendations = hybrid_recommender(
        user_id=request.user_id,
        query=request.query,
        top_k=request.top_k,
        category=request.category,
        use_xgboost=request.use_xgboost
    )

    return {
        "model": "Hybrid Recommendation System",
        "query": request.query,
        "recommendations": recommendations
    }