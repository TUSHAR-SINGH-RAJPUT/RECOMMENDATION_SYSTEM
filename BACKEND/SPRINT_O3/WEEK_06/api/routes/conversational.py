from fastapi import APIRouter
from schemas import ChatRequest

router = APIRouter(
    prefix="/recommend/conversational",
    tags=["Conversational RAG"]
)


@router.post("/")
def conversational_recommend(request: ChatRequest):

    response = (
        f"Based on your interest in '{request.query}', "
        "I recommend modern minimalist furniture with neutral colors "
        "and wooden textures for a clean aesthetic."
    )

    products = [
        {
            "product_id": 401,
            "product_name": "Minimalist Sofa",
            "category": "Sofa",
            "score": 0.97
        },
        {
            "product_id": 402,
            "product_name": "Nordic Coffee Table",
            "category": "Table",
            "score": 0.93
        },
        {
            "product_id": 403,
            "product_name": "Modern Floor Lamp",
            "category": "Lighting",
            "score": 0.89
        }
    ]

    return {
        "model": "Conversational RAG Recommender",
        "response": response,
        "products": products
    }