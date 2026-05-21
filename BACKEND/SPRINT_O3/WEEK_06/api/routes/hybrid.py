from fastapi import APIRouter
from schemas import RecommendationRequest

router = APIRouter(
    prefix="/recommend/hybrid",
    tags=["Hybrid Recommendations"]
)


@router.post("/")
def hybrid_recommend(request: RecommendationRequest):

    recommendations = [
        {
            "product_id": 301,
            "product_name": "Luxury Leather Sofa",
            "category": "Sofa",
            "score": 0.98
        },
        {
            "product_id": 302,
            "product_name": "Premium TV Stand",
            "category": "Living Room",
            "score": 0.94
        },
        {
            "product_id": 303,
            "product_name": "Designer Coffee Table",
            "category": "Table",
            "score": 0.91
        }
    ]

    return {
        "model": "Hybrid Recommendation System",
        "query": request.query,
        "recommendations": recommendations
    }
