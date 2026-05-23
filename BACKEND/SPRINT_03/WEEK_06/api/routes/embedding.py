from fastapi import APIRouter

# Import embedding/content-based recommender function
from SPRINT_02.WEEK_03.src.content_recommender import search_products

# Import request schema
from SPRINT_03.WEEK_06.api.schemas import Search_products


# ============================================================
# ROUTER CONFIGURATION
# ============================================================
# Creates a router for embedding-based recommendation endpoints
#
# prefix:
#   All routes in this file will start with:
#   /recommend/embedding
#
# tags:
#   Groups endpoints together in FastAPI Swagger docs
# ============================================================
router = APIRouter(
    prefix="/recommend/embedding",
    tags=["Embedding Recommendations"]
)


# ============================================================
# EMBEDDING RECOMMENDATION ENDPOINT
# ============================================================
# POST /recommend/embedding/
#
# This endpoint:
#   1. Accepts a user query
#   2. Converts query into embeddings
#   3. Searches similar products using vector similarity
#   4. Returns top matching products
#
# Request Body:
# {
#     "query": "modern wooden sofa",
#     "top_k": 5,
#     "category": "Living Room"
# }
#
# Returns:
# {
#     "model": "Embedding Based Recommendation",
#     "query": "...",
#     "recommendations": [...]
# }
# ============================================================
@router.post("/")
def embedding_recommend(request: Search_products):

    # --------------------------------------------------------
    # Generate embedding-based recommendations
    #
    # query:
    #   User search text
    #
    # top_k:
    #   Number of recommendations to return
    # --------------------------------------------------------
    recommendations = search_products(
        query=request.query,
        top_k=request.top_k,
        category=request.category
    )

    # --------------------------------------------------------
    # API Response
    # --------------------------------------------------------
    return {
        "model": "Embedding Based Recommendation",
        "query": request.query,
        "recommendations": recommendations
    }