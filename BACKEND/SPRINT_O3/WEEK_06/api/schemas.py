from pydantic import BaseModel
from typing import Optional, List


class RecommendationRequest(BaseModel):
    user_id: int
    query: Optional[str] = None
    top_k: int = 5


class ChatRequest(BaseModel):
    user_id: int
    query: str


class Product(BaseModel):
    product_id: int
    product_name: str
    category: str
    score: float


class RecommendationResponse(BaseModel):
    recommendations: List[Product]


class ChatResponse(BaseModel):
    response: str
    products: List[Product]