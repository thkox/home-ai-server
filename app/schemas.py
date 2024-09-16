from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional

class UserBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-zA-Z]+$')
    last_name: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-zA-Z]+$')
    email: EmailStr
    enabled: Optional[bool] = True

    # Field-level validators
    @field_validator('first_name', 'last_name', mode='before')
    def names_must_be_alphabetic(cls, value):
        if not value.isalpha():
            raise ValueError('Names must contain only alphabetic characters.')
        return value

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=50)

    @field_validator('password', mode='before')
    def validate_password(cls, password):
        if len(password) < 8:
            raise ValueError('Password must be at least 8 characters long.')
        if not any(char.isdigit() for char in password):
            raise ValueError('Password must contain at least one number.')
        if not any(char.isupper() for char in password):
            raise ValueError('Password must contain at least one uppercase letter.')
        if not any(char in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for char in password):
            raise ValueError('Password must contain at least one special character.')
        return password

class UserOut(UserBase):
    user_id: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
