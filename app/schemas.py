from pydantic import BaseModel
from typing import Optional

class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: str

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
