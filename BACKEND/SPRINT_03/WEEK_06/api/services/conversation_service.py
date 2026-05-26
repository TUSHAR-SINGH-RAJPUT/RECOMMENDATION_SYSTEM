"""Conversational RAG service with Ollama streaming.

Flow:
  user query -> product retrieval -> prompt construction -> Ollama streaming
  -> Server-Sent Events for the React chatbot.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from typing import AsyncGenerator, Deque, Dict, Iterable, List

import httpx

from SPRINT_03.WEEK_06.api.services.recommendation_service import (
    get_embedding_recommendations,
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
MEMORY_TURNS = int(os.getenv("CHAT_MEMORY_TURNS", "8"))

ConversationMemory = Deque[Dict[str, str]]
_memory: Dict[str, ConversationMemory] = defaultdict(
    lambda: deque(maxlen=MEMORY_TURNS)
)


def _session_key(user_id: int, session_id: str | None) -> str:
    return session_id or f"user-{user_id}"


def _format_products(products: Iterable[Dict]) -> str:
    lines = []
    for rank, product in enumerate(products, start=1):
        lines.append(
            "\n".join(
                [
                    f"{rank}. {product['product_name']}",
                    f"   ID: {product['product_id']}",
                    f"   Category: {product['category']}",
                    f"   Score: {product.get('score', 0)}",
                    f"   Details: {product.get('description') or 'No description available.'}",
                ]
            )
        )
    return "\n\n".join(lines) or "No matching products were retrieved."


def _format_history(history: ConversationMemory) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"{turn['role'].title()}: {turn['content']}" for turn in history
    )


def build_prompt(query: str, products: List[Dict], history: ConversationMemory) -> str:
    """Build a grounded shopping-assistant prompt for RAG generation."""

    return f"""You are RoomSense, a concise and helpful furniture shopping assistant.
Use the retrieved product context to recommend relevant furniture. Stay grounded
in the product context, mention product names naturally, and ask one useful
follow-up question when more preference details would improve the recommendation.

Conversation history:
{_format_history(history)}

Retrieved product context:
{_format_products(products)}

User query:
{query}

Answer in a natural conversational style with practical buying guidance."""


def _sse(event: str, data: Dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_chat_response(
    *,
    user_id: int,
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a RAG answer as Server-Sent Events."""

    key = _session_key(user_id, session_id)
    history = _memory[key]
    products = get_embedding_recommendations(query=query, top_k=top_k)
    prompt = build_prompt(query=query, products=products, history=history)

    yield _sse(
        "products",
        {
            "products": products,
            "session_id": key,
            "created_at": int(time.time()),
        },
    )

    answer_parts: List[str] = []

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.7},
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        answer_parts.append(token)
                        yield _sse("token", {"token": token})
                    if chunk.get("done"):
                        break

    except httpx.HTTPError as exc:
        yield _sse(
            "error",
            {
                "message": (
                    "Ollama is not reachable. Start Ollama and make sure the "
                    f"'{OLLAMA_MODEL}' model is available."
                ),
                "detail": str(exc),
            },
        )
        return

    answer = "".join(answer_parts).strip()
    if answer:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

    yield _sse("done", {"response": answer, "session_id": key})
