import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Boolean, Enum, ForeignKey, DateTime, Text, Float, BigInteger
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from .database import Base


class SecretKey(Base):
    __tablename__ = "secret_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, nullable=False)


class UserRole(str, enum.Enum):
    admin = "admin"
    house_member = "house_member"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(Enum(UserRole), default=UserRole.house_member)
    enabled = Column(Boolean, default=True)

    conversations = relationship("Conversation", back_populates="user")
    documents = relationship("Document", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    start_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time = Column(DateTime, nullable=True)
    title = Column(String, nullable=True)
    status = Column(String, default="active")
    selected_document_ids = Column(ARRAY(UUID(as_uuid=True)), default=[])

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False)
    sender_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    content = Column(Text, nullable=False)
    llm_model = Column(String, nullable=False)
    tokens_generated = Column(Integer, nullable=True, default=0)
    response_time = Column(Float, nullable=True, default=0)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    size = Column(BigInteger, nullable=False)
    checksum = Column(String, nullable=False)

    user = relationship("User", back_populates="documents")
