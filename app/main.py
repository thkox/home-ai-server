from contextlib import asynccontextmanager
from datetime import timedelta
from typing import List, Optional

from fastapi import Depends, HTTPException, status, File, UploadFile
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from . import models, schemas, auth, conversations
from .auth import update_user_profile
from .database import engine
from .database import get_db
from .models import User, UserRole
from .utils import ensure_assistant_user_exists, get_or_create_secret_key

models.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = next(get_db())
    ensure_assistant_user_exists(db, User, UserRole)
    get_or_create_secret_key(db)

    yield

    db.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Home AI API"}


@app.post("/token", response_model=schemas.Token)
def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"user_id": str(user.user_id)},
        expires_delta=access_token_expires,
        db=db  # Pass the db session here
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/", response_model=schemas.UserOut)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return schemas.UserOut(
        user_id=str(new_user.user_id),
        first_name=new_user.first_name,
        last_name=new_user.last_name,
        email=new_user.email,
        enabled=new_user.enabled,
        role=new_user.role
    )


@app.put("/users/me", response_model=schemas.UserOut)
def update_my_profile(user: schemas.UserCreate, db: Session = Depends(get_db),
                      current_user: models.User = Depends(auth.get_current_user)):
    return update_user_profile(db, current_user, user)


@app.put("/users/{user_id}", response_model=schemas.UserOut)
def update_profile(user_id: str, user: schemas.UserCreate, db: Session = Depends(get_db),
                   current_user: models.User = Depends(auth.get_current_admin_user)):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if db_user.email != user.email:
        email_exists = db.query(models.User).filter(models.User.email == user.email).first()
        if email_exists:
            raise HTTPException(status_code=400, detail="Email already registered")

    return update_user_profile(db, db_user, user)


@app.post("/conversations/", response_model=schemas.ConversationOut)
def start_conversation(db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    new_convo = conversations.create_new_conversation(db, user_id=str(current_user.user_id))
    return new_convo


from pydantic import BaseModel


class ContinueConversationRequest(BaseModel):
    message: str
    selected_documents: Optional[List[str]] = None


@app.post("/conversations/{conversation_id}/continue", response_model=schemas.MessageOut)
def continue_existing_conversation(
        conversation_id: str,
        request: ContinueConversationRequest,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_user)
):
    llm_message = conversations.continue_conversation(
        db,
        conversation_id,
        user_id=str(current_user.user_id),
        message_content=request.message,
        selected_documents=request.selected_documents
    )
    return llm_message


@app.post("/documents/upload")
def upload_documents(
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_user)
):
    return conversations.upload_user_documents(
        db=db,
        user_id=str(current_user.user_id),
        files=files
    )


@app.delete("/documents/{document_id}")
def delete_user_document(
        document_id: str,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_user)
):
    return conversations.delete_document(db, document_id, user_id=str(current_user.user_id))


@app.get("/documents/me", response_model=List[schemas.DocumentOut])
def get_user_documents(
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_user)
):
    documents = conversations.list_user_documents(db, user_id=str(current_user.user_id))
    return documents


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db),
                        current_user: models.User = Depends(auth.get_current_user)):
    return conversations.delete_conversation(db, conversation_id, user_id=str(current_user.user_id))


@app.get("/conversations/me", response_model=List[schemas.ConversationOut])
def get_user_conversations(
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_user)
):
    user_conversations = db.query(models.Conversation).filter(
        models.Conversation.user_id == current_user.user_id
    ).all()

    return user_conversations


@app.get("/conversations/{conversation_id}/messages", response_model=List[schemas.MessageOut])
def get_conversation_messages(
        conversation_id: str,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_user)
):
    conversation = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id,
        models.Conversation.user_id == current_user.user_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(models.Message).filter(
        models.Message.conversation_id == conversation_id
    ).order_by(models.Message.timestamp.asc()).all()

    return messages
