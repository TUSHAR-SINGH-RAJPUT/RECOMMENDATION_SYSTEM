from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from SPRINT_03.WEEK_06.api.schemas import ChatRequest
from SPRINT_03.WEEK_06.api.services.conversation_service import stream_chat_response

router = APIRouter(
    prefix="/recommend/conversational",
    tags=["Conversational RAG"]
)


@router.post("/")
def conversational_recommend(request: ChatRequest):
    """Stream a retrieval-augmented Ollama response as Server-Sent Events."""

    return StreamingResponse(
        stream_chat_response(
            user_id=request.user_id,
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
