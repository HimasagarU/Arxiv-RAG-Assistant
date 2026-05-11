"""
chat.py — Conversation and message management endpoints.

Endpoints:
    POST   /conversations                   — Create new conversation
    GET    /conversations                   — List user's conversations
    GET    /conversations/{id}/messages      — Get message history
    POST   /conversations/{id}/query         — Chat within a conversation
    DELETE /conversations/{id}               — Soft-delete conversation

Features:
    - Sliding window context (last 4 turns) to control Groq token usage
    - 20-query-per-conversation hard cap
    - Redis session caching for fast page reload
    - Redis query cache for duplicate question avoidance
    - Optional paper_id scoping for document-specific chats
"""

import json
import logging
import os
import time
from typing import Optional
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.cache import (
    cache_message,
    get_cached_messages,
    get_cached_response,
    invalidate_session_cache,
    set_cached_response,
)
from db.app_database import get_app_db
from db.app_models import Conversation, Message, User

load_dotenv()

log = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Chat"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERIES_PER_CONVERSATION = 20
SLIDING_WINDOW_TURNS = 4  # last N user+assistant pairs sent to LLM


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CreateConversationRequest(BaseModel):
    title: str = Field(default="New Conversation", max_length=256)
    paper_id: Optional[str] = Field(default=None, max_length=64)


class ConversationResponse(BaseModel):
    id: str
    title: str
    paper_id: Optional[str]
    message_count: int
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources_json: Optional[str] = None
    created_at: str


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatQueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_trace: dict
    message_count: int
    max_queries: int = MAX_QUERIES_PER_CONVERSATION
    cached: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv_to_response(conv: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=str(conv.id),
        title=conv.title,
        paper_id=conv.paper_id,
        message_count=conv.message_count,
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
    )


def _msg_to_response(msg: Message) -> MessageResponse:
    return MessageResponse(
        id=str(msg.id),
        role=msg.role,
        content=msg.content,
        sources_json=msg.sources_json,
        created_at=msg.created_at.isoformat() if msg.created_at else "",
    )


def _build_sliding_window(messages: list[Message]) -> list[dict]:
    """Build last N turn pairs for LLM context (user + assistant)."""
    # Only include the most recent SLIDING_WINDOW_TURNS pairs
    pairs = []
    i = len(messages) - 1
    while i >= 0 and len(pairs) < SLIDING_WINDOW_TURNS * 2:
        pairs.insert(0, {
            "role": messages[i].role,
            "content": messages[i].content,
        })
        i -= 1
    return pairs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    body: CreateConversationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """Create a new conversation (optionally scoped to a paper)."""
    conv = Conversation(
        user_id=current_user.id,
        title=body.title,
        paper_id=body.paper_id,
    )
    db.add(conv)
    await db.flush()
    log.info(f"Conversation created: {conv.id} (user={current_user.email})")
    return _conv_to_response(conv)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """List the current user's conversations, most recent first."""
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == current_user.id,
            Conversation.is_deleted == False,
        )
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    convs = result.scalars().all()
    return [_conv_to_response(c) for c in convs]


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """Get message history for a conversation (tries Redis cache first)."""
    # Verify ownership
    conv = await _get_user_conversation(db, conversation_id, current_user.id)

    # Try Redis cache first for fast reload
    cached = get_cached_messages(str(conversation_id))
    if cached:
        log.debug(f"Chat session cache HIT for {conversation_id}")
        return [
            MessageResponse(
                id="cached",
                role=m["role"],
                content=m["content"],
                sources_json=m.get("sources_json"),
                created_at="",
            )
            for m in cached
        ]

    # Fallback to database
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()
    return [_msg_to_response(m) for m in messages]


@router.post("/{conversation_id}/query", response_model=ChatQueryResponse)
async def chat_query(
    conversation_id: UUID,
    body: ChatQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """
    Send a query within a conversation.

    Pipeline:
    1. Enforce 20-query limit
    2. Check Redis cache for duplicate query
    3. Build sliding window context from history
    4. Run retrieval (paper-scoped if conversation has paper_id)
    5. Generate answer via Groq
    6. Save messages to DB + Redis
    """
    conv = await _get_user_conversation(db, conversation_id, current_user.id)

    # --- 1. Query limit ---
    user_msg_count = conv.message_count // 2  # each query = 1 user + 1 assistant
    if user_msg_count >= MAX_QUERIES_PER_CONVERSATION:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Query limit reached ({MAX_QUERIES_PER_CONVERSATION} per conversation). "
                   f"Please start a new conversation.",
        )

    # --- 2. Check Redis query cache ---
    cached = get_cached_response(body.query, conv.paper_id)
    if cached:
        # Still save messages so history is consistent
        await _save_messages(
            db, conv, body.query, cached["answer"],
            json.dumps(cached.get("sources", []), default=str),
        )
        return ChatQueryResponse(
            answer=cached["answer"],
            sources=cached.get("sources", []),
            retrieval_trace=cached.get("retrieval_trace", {}),
            message_count=conv.message_count,
            cached=True,
        )

    # --- 3. Load history for sliding window ---
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    history = result.scalars().all()
    context_window = _build_sliding_window(list(history))

    # --- 4. Run retrieval pipeline ---
    from api.app import _state, build_prompt, generate_answer
    from api.retrieval import classify_query_intent

    if _state["retriever"] is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized.")

    t0 = time.time()
    intent = classify_query_intent(body.query)

    # Build retrieval kwargs
    retrieve_kwargs = {
        "top_n": body.top_k,
        "intent": intent,
    }
    # Paper-scoped retrieval
    if conv.paper_id:
        retrieve_kwargs["paper_id"] = conv.paper_id

    retrieval_result = _state["retriever"].retrieve(body.query, **retrieve_kwargs)
    passages = retrieval_result["passages"]
    trace = retrieval_result["trace"]

    # Compress context
    compressed = _state["retriever"].compress_context(body.query, passages, intent=intent)

    # --- Build prompt with conversation history ---
    if conv.paper_id:
        # Document-scoped grounding prompt
        prompt = _build_document_chat_prompt(body.query, compressed, passages, context_window)
    else:
        prompt = build_prompt(body.query, compressed, passages, intent=intent)

    # --- 5. Generate answer ---
    answer = generate_answer(prompt, intent=intent)
    trace["total_ms"] = round((time.time() - t0) * 1000, 1)

    # Build sources list
    sources = [
        {
            "chunk_id": p["chunk_id"],
            "paper_id": p["paper_id"],
            "title": p["title"],
            "authors": p.get("authors", ""),
            "chunk_text": p["chunk_text"],
            "rerank_score": p.get("rerank_score", 0.0),
        }
        for p in passages
    ]

    # --- 6. Save to DB + Redis ---
    sources_str = json.dumps(sources, default=str)
    await _save_messages(db, conv, body.query, answer, sources_str)

    # Cache in Redis
    set_cached_response(body.query, {
        "answer": answer,
        "sources": sources,
        "retrieval_trace": trace,
    }, paper_id=conv.paper_id)

    return ChatQueryResponse(
        answer=answer,
        sources=sources,
        retrieval_trace=trace,
        message_count=conv.message_count,
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_app_db),
):
    """Soft-delete a conversation."""
    conv = await _get_user_conversation(db, conversation_id, current_user.id)
    conv.is_deleted = True
    await db.flush()
    invalidate_session_cache(str(conversation_id))
    log.info(f"Conversation soft-deleted: {conversation_id}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _get_user_conversation(
    db: AsyncSession, conversation_id: UUID, user_id
) -> Conversation:
    """Fetch a conversation and verify ownership."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.is_deleted == False,
        )
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


async def _save_messages(
    db: AsyncSession,
    conv: Conversation,
    user_content: str,
    assistant_content: str,
    sources_json: Optional[str] = None,
) -> None:
    """Save user + assistant messages to DB and Redis session cache."""
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=user_content,
    )
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=assistant_content,
        sources_json=sources_json,
    )
    db.add(user_msg)
    db.add(assistant_msg)

    conv.message_count += 2
    # Auto-generate title from first query
    if conv.message_count == 2:
        conv.title = user_content[:100] + ("..." if len(user_content) > 100 else "")

    await db.flush()

    # Update Redis session cache
    cache_message(str(conv.id), "user", user_content)
    cache_message(str(conv.id), "assistant", assistant_content, sources_json)


def _build_document_chat_prompt(
    query: str,
    compressed_context: str,
    passages: list[dict],
    history: list[dict],
) -> str:
    """Build a strictly grounded prompt for paper-scoped chat."""
    # Build source references
    sources_block = []
    for i, p in enumerate(passages, 1):
        title = p.get("title", "Untitled")
        sources_block.append(f'[{i}] "{title}"')

    # Build conversation history string
    history_str = ""
    if history:
        history_lines = []
        for msg in history[-SLIDING_WINDOW_TURNS * 2:]:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg['content'][:500]}")
        history_str = "\n".join(history_lines)

    return f"""You are an expert AI research assistant. You are chatting with a user about a SPECIFIC document.

CRITICAL RULES:
1. Your answer MUST be grounded ONLY in the provided document chunks below.
2. Do NOT use any external knowledge. If the document chunks do not contain the answer, say:
   "The document does not contain information about this. Please try rephrasing your question."
3. Use numbered citations [1], [2] to reference the source chunks.
4. Be concise but thorough. Use the Feynman Technique for explanations.

CONVERSATION HISTORY:
{history_str if history_str else "(No prior messages)"}

DOCUMENT CHUNKS:
{compressed_context}

AVAILABLE SOURCES:
{chr(10).join(sources_block)}

USER QUESTION: {query}

ANSWER (grounded only in the document):"""
