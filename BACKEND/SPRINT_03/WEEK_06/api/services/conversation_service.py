"""Conversational assistant service with Ollama and product RAG.

Flow:
  user query -> intent routing -> optional product retrieval -> prompt
  construction -> Ollama streaming -> Server-Sent Events for the React chatbot.
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
PRODUCT_HINTS = {
    "recommend",
    "recommendation",
    "recommendations",
    "suggest",
    "show",
    "find",
    "buy",
    "product",
    "products",
    "bed",
    "beds",
    "chair",
    "chairs",
    "sofa",
    "sofas",
    "table",
    "tables",
    "desk",
    "desks",
    "lamp",
    "lamps",
    "lighting",
    "wood",
    "wooden",
    "metal",
    "glass",
    "modern",
    "minimalist",
    "luxury",
    "classic",
    "budget",
    "price",
    "under",
    "room",
    "furniture",
    "interior",
    "decor",
}

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


def _clean_query(query: str) -> str:
    punctuation = "?!.,;:()[]{}\"'"
    cleaned = query.lower().strip()
    for mark in punctuation:
        cleaned = cleaned.replace(mark, " ")
    return " ".join(cleaned.split())


def _has_product_intent(query: str) -> bool:
    cleaned = _clean_query(query)
    return any(hint in cleaned.split() for hint in PRODUCT_HINTS)


def build_product_prompt(
    query: str,
    products: List[Dict],
    history: ConversationMemory,
) -> str:
    """Build a grounded shopping-assistant prompt for RAG generation."""

    return f"""You are RoomSense, a concise, human furniture shopping assistant.
Do not sound like a scripted sales bot.

Rules:
- Answer in 1-3 short sentences.
- Mention at most two product names.
- Do not list every retrieved product.
- Ask at most one natural follow-up question.
- If the user is just greeting or chatting, do not recommend products.
- Stay grounded in the product context when recommending.

Conversation history:
{_format_history(history)}

Retrieved product context:
{_format_products(products)}

User query:
{query}

Answer like a real person helping in a store, not a report."""


def build_general_prompt(query: str, history: ConversationMemory) -> str:
    """Build a light conversational prompt for non-product chat."""

    return f"""You are RoomSense, a friendly AI assistant inside a furniture recommendation app.

Rules:
- Answer naturally in 1-3 short sentences.
- Be warm, useful, and conversational.
- If the user asks for furniture, products, prices, styles, rooms, or recommendations, tell them you can search the catalog for them.
- Do not invent product details unless product context is provided.

Conversation history:
{_format_history(history)}

User query:
{query}

Answer like a helpful person, not a report."""


def _sse(event: str, data: Dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_ollama(prompt: str, temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


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
    product_intent = _has_product_intent(query)
    products = get_embedding_recommendations(query=query, top_k=top_k) if product_intent else []
    prompt = (
        build_product_prompt(query=query, products=products, history=history)
        if product_intent
        else build_general_prompt(query=query, history=history)
    )

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
        async for token in _stream_ollama(
            prompt=prompt,
            temperature=0.45 if product_intent else 0.65,
            max_tokens=100 if product_intent else 120,
        ):
            answer_parts.append(token)
            yield _sse("token", {"token": token})

    except httpx.HTTPError as exc:
        fallback = (
            "I can chat through local Mistral once Ollama is running. Start Ollama and make sure the model is available."
            if not product_intent
            else (
                "I found matching products, but local Mistral is not reachable to write the response. "
                "Start Ollama and make sure the model is available."
            )
        )
        yield _sse(
            "error",
            {
                "message": fallback,
                "detail": str(exc),
            },
        )
        return

    answer = "".join(answer_parts).strip()
    if answer:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

    yield _sse("done", {"response": answer, "session_id": key})
