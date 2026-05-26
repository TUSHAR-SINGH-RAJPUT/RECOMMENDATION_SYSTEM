"""Adapters for all recommendation engines.

The sprint modules were built at different times and return slightly different
shapes. This service normalizes every model into one product contract so the
FastAPI routes and React UI can consume them consistently.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from SPRINT_01.WEEK_02.models.collaborative_filter import recommend_cf
from SPRINT_02.WEEK_03.src.content_recommender import search_products
from SPRINT_02.WEEK_04.Models.hybrid_recommender import hybrid_recommender

BASE_DIR = os.path.dirname(__file__)
METADATA_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../../../SPRINT_02/WEEK_03/embeddings/embedding_metadata.json",
    )
)


@lru_cache(maxsize=1)
def _load_catalog() -> Dict[str, Dict[str, Any]]:
    """Load product metadata once so CF numeric IDs can become real cards."""

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            rows = json.load(file)
    except FileNotFoundError:
        return {}

    return {str(row.get("product_id")): row for row in rows}


def _catalog_candidates(product_id: str) -> List[str]:
    """Return possible metadata IDs for older numeric collaborative IDs."""

    candidates = [product_id]
    if product_id.isdigit():
        number = int(product_id)
        candidates.extend([f"P{number}", f"P{max(number - 1, 0)}"])
    return candidates


def _enrich_from_catalog(item: Dict[str, Any]) -> Dict[str, Any]:
    """Attach name/category/description from the embedding catalog when possible."""

    catalog = _load_catalog()
    product_id = str(item.get("product_id") or item.get("product_name") or "")

    for candidate in _catalog_candidates(product_id):
        if candidate in catalog:
            meta = catalog[candidate]
            return {
                **item,
                "product_id": candidate,
                "product_name": meta.get("product_name", item.get("product_name")),
                "category": meta.get("category", item.get("category")),
                "description": meta.get("description", item.get("description")),
            }

    return item


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_scores(
    products: List[Dict[str, Any]],
    *,
    floor: float,
    ceiling: float,
) -> List[Dict[str, Any]]:
    """Normalize raw model scores into a stable 0-1 display band."""

    if not products:
        return products

    raw_scores = [_safe_float(item.get("score")) for item in products]
    min_score = min(raw_scores)
    max_score = max(raw_scores)

    for item, raw_score in zip(products, raw_scores):
        item.setdefault("metadata", {})
        item["metadata"]["raw_score"] = raw_score

        if max_score > min_score:
            normalized = (raw_score - min_score) / (max_score - min_score)
        else:
            normalized = 1.0 if raw_score > 0 else 0.0

        item["score"] = round(_clamp_score(floor + (ceiling - floor) * normalized), 4)

    return products


def normalize_product(item: Any, fallback_score: float = 0.0) -> Dict[str, Any]:
    """Convert recommender-specific result objects into one API product shape."""

    if isinstance(item, dict):
        product_id = str(
            item.get("product_id")
            or item.get("item_id")
            or item.get("id")
            or item.get("product_name")
            or "unknown"
        )
        score = _safe_float(
            item.get(
                "score",
                item.get(
                    "recommendation_score",
                    item.get("final_score", item.get("content_score")),
                ),
            ),
            fallback_score,
        )
        metadata = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "product_id",
                "item_id",
                "id",
                "product_name",
                "category",
                "description",
                "score",
                "recommendation_score",
            }
        }
        return {
            "product_id": product_id,
            "product_name": str(item.get("product_name") or product_id),
            "category": str(item.get("category") or "furniture"),
            "description": item.get("description"),
            "score": round(score, 4),
            "metadata": metadata,
        }

    product_id = str(item)
    return {
        "product_id": product_id,
        "product_name": product_id,
        "category": "furniture",
        "description": None,
        "score": round(fallback_score, 4),
        "metadata": {},
    }


def normalize_products(items: Iterable[Any]) -> List[Dict[str, Any]]:
    return [normalize_product(item) for item in items]


def get_embedding_recommendations(
    query: str,
    top_k: int = 10,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return semantic product matches from the ChromaDB-backed search engine."""

    products = normalize_products(
        search_products(query=query, top_k=top_k, category=category)
    )
    return normalize_scores(products, floor=0.55, ceiling=0.84)


def get_collaborative_recommendations(
    user_id: int,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return behavior-based recommendations for a user."""

    raw_items = recommend_cf(user_id=user_id, n=top_k)
    enriched_items = [
        _enrich_from_catalog(item) if isinstance(item, dict) else item
        for item in raw_items
    ]
    products = normalize_products(enriched_items)
    return normalize_scores(products, floor=0.35, ceiling=0.68)


def get_hybrid_recommendations(
    user_id: int,
    query: str,
    top_k: int = 10,
    category: Optional[str] = None,
    use_xgboost: bool = True,
) -> List[Dict[str, Any]]:
    """Return blended recommendations from semantic, CF, and LTR signals."""

    raw_items = hybrid_recommender(
        user_id=user_id,
        query=query,
        top_k=top_k,
        category=category,
        use_xgboost=use_xgboost,
    )
    products = [_enrich_from_catalog(product) for product in normalize_products(raw_items)]
    return normalize_scores(products, floor=0.70, ceiling=0.98)
