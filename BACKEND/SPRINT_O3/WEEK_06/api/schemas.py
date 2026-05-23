from pydantic import BaseModel
from typing import Optional, List


class RecommendationRequest(BaseModel):
    user_id: int
    query: str
    top_k: int = 10
    category: Optional[str] = None
    use_xgboost: bool = True


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

class Search_products(BaseModel):
    query: str
    top_k: int = 10
    category: Optional[str] = None
    