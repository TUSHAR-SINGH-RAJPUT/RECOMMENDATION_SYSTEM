from fastapi import APIRouter
from schemas import RecommendationRequest

router = APIRouter(
    prefix="/recommend/embedding",
    tags=["Embedding Recommendations"]
)


@router.post("/")
def embedding_recommend(request: RecommendationRequest):

    recommendations = [
        {
            "product_id": 201,
            "product_name": "Minimalist White Chair",
            "category": "Chair",
            "score": 0.93
        },
        {
            "product_id": 202,
            "product_name": "Scandinavian Table",
            "category": "Table",
            "score": 0.89
        },
        {
            "product_id": 203,
            "product_name": "Modern TV Unit",
            "category": "Living Room",
            "score": 0.86
        }
    ]

    return {
        "model": "Embedding Based Recommendation",
        "query": request.query,
        "recommendations": recommendations
    }