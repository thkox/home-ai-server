import uuid
from sqlalchemy import Column, Integer, String, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID
from .database import Base
import enum

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
    enabled = Column(Boolean, default=True)  # New field to indicate if user is active
