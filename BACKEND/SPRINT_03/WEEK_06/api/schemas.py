from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RecommendationRequest(BaseModel):
    """Shared request body for model-backed recommendation endpoints."""

    user_id: int = Field(default=1, ge=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    category: Optional[str] = None
    use_xgboost: bool = True


class ChatRequest(BaseModel):
    """Request body for conversational RAG chat."""

    user_id: int = Field(default=1, ge=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)
    session_id: Optional[str] = None


class Product(BaseModel):
    product_id: str
    product_name: str
    category: str
    score: float = 0.0
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    model: str
    query: str
    user_id: Optional[int] = None
    count: int
    recommendations: List[Product]


class ChatResponse(BaseModel):
    response: str
    products: List[Product]


class SearchProductsRequest(BaseModel):
    """Request body for semantic product search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    category: Optional[str] = None
