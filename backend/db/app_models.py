"""
app_models.py — SQLAlchemy ORM models for the application database.

Tables:
  - users           : registered accounts
  - conversations   : chat sessions (corpus-wide or paper-scoped)
  - messages        : individual user / assistant messages
  - document_jobs   : async document ingestion tracking
"""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    display_name = Column(String(128), nullable=False, default="Researcher")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    conversations = relationship("Conversation", back_populates="user", lazy="selectin")
    document_jobs = relationship("DocumentJob", back_populates="user", lazy="selectin")

    def __repr__(self):
        return f"<User {self.email}>"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(256), nullable=False, default="New Conversation")
    # If paper_id is set, this is a document-scoped chat
    paper_id = Column(String(64), nullable=True, index=True)
    message_count = Column(Integer, default=0, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        lazy="selectin",
        order_by="Message.created_at",
    )

    def __repr__(self):
        return f"<Conversation {self.id} user={self.user_id}>"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(
        String(16),
        nullable=False,
        comment="user or assistant",
    )
    content = Column(Text, nullable=False)
    # Store source chunks as JSON string for assistant messages
    sources_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message {self.role} in {self.conversation_id}>"


# ---------------------------------------------------------------------------
# Document Ingestion Jobs
# ---------------------------------------------------------------------------

class DocumentJob(Base):
    __tablename__ = "document_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    arxiv_id = Column(String(64), nullable=False)
    pdf_url = Column(String(512), nullable=True)
    title = Column(String(512), nullable=True)
    status = Column(
        String(32),
        nullable=False,
        default="queued",
        comment="queued | downloading | chunking | embedding | done | failed | cancelled",
    )
    error_message = Column(Text, nullable=True)
    chunks_created = Column(Integer, default=0, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="document_jobs")

    def __repr__(self):
        return f"<DocumentJob {self.arxiv_id} status={self.status}>"
