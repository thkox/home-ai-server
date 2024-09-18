import uuid
from sqlalchemy import Column, Integer, String, Boolean, Enum, ForeignKey, DateTime, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from .database import Base
import enum
from sqlalchemy.orm import relationship
import datetime

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
    # documents = relationship("Document", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)  # FK to User
    start_time = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    end_time = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)  # Store chat summary
    status = Column(String, default="active")  # active, closed, etc.

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False)
    sender_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))  # The sender of the message
    content = Column(Text, nullable=False)
    llm_model = Column(String, nullable=False)
    response_time = Column(Float, nullable=True)  # Time in seconds to generate response
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")