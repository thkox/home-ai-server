import secrets
import uuid

from sqlalchemy.orm import Session

from .models import SecretKey

ASSISTANT_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def ensure_assistant_user_exists(db, User, UserRole):
    assistant_user = db.query(User).filter(User.user_id == ASSISTANT_UUID).first()
    if not assistant_user:
        new_user = User(
            user_id=ASSISTANT_UUID,
            first_name="Assistant",
            last_name="Bot",
            email="assistant@bot.com",
            hashed_password="",
            role=UserRole.house_member,
            enabled=True
        )
        db.add(new_user)
        db.commit()


def get_or_create_secret_key(db: Session) -> str:
    secret_key = db.query(SecretKey).first()
    if not secret_key:
        new_key = secrets.token_urlsafe(64)
        secret_key = SecretKey(key=new_key)
        db.add(secret_key)
        db.commit()
        db.refresh(secret_key)
    return secret_key.key
