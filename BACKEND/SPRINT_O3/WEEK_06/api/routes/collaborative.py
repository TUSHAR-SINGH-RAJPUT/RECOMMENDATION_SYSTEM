from fastapi import APIRouter
from schemas import RecommendationRequest

router = APIRouter(
    prefix="/recommend/collaborative",
    tags=["Collaborative Filtering"]
)


@router.post("/")
def collaborative_recommend(request: RecommendationRequest):

    recommendations = [
        {
            "product_id": 101,
            "product_name": "Modern Grey Sofa",
            "category": "Sofa",
            "score": 0.95
        },
        {
            "product_id": 102,
            "product_name": "Wooden Coffee Table",
            "category": "Table",
            "score": 0.90
        },
        {
            "product_id": 103,
            "product_name": "Luxury Recliner",
            "category": "Chair",
            "score": 0.87
        }
    ]

    return {
        "model": "Collaborative Filtering",
        "user_id": request.user_id,
        "recommendations": recommendations
    }