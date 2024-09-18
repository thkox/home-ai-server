from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from . import models, schemas, auth, conversations
from .database import engine, get_db
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

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
        data={"sub": user.email, "user_id": str(user.user_id), "first_name": user.first_name, "last_name": user.last_name},
        expires_delta=access_token_expires
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
def update_my_profile(user: schemas.UserCreate, db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    current_user.first_name = user.first_name
    current_user.last_name = user.last_name
    current_user.email = user.email
    if user.password:
        current_user.hashed_password = auth.get_password_hash(user.password)
    db.commit()
    db.refresh(current_user)
    return schemas.UserOut(
        user_id=str(current_user.user_id),
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        email=current_user.email,
        enabled=current_user.enabled,
        role=current_user.role
    )

@app.put("/users/{user_id}", response_model=schemas.UserOut)
def update_profile(user_id: str, user: schemas.UserCreate, db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_admin_user)):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if the new email already exists in the database
    if db_user.email != user.email:
        email_exists = db.query(models.User).filter(models.User.email == user.email).first()
        if email_exists:
            raise HTTPException(status_code=400, detail="Email already registered")

    db_user.first_name = user.first_name
    db_user.last_name = user.last_name
    db_user.email = user.email
    if user.password:
        db_user.hashed_password = auth.get_password_hash(user.password)
    db.commit()
    db.refresh(db_user)
    return schemas.UserOut(
        user_id=str(db_user.user_id),
        first_name=db_user.first_name,
        last_name=db_user.last_name,
        email=db_user.email,
        enabled=db_user.enabled,
        role=db_user.role
    )

# Start a new conversation
@app.post("/conversations/", response_model=schemas.ConversationOut)
def start_conversation(db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    new_convo = conversations.create_new_conversation(db, user_id=str(current_user.user_id))
    return new_convo

# Continue a conversation
@app.post("/conversations/{conversation_id}/continue")
def continue_existing_conversation(conversation_id: str, message: str, db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    return conversations.continue_conversation(db, conversation_id, user_id=str(current_user.user_id), message_content=message)

# Delete a conversation
@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    return conversations.delete_conversation(db, conversation_id, user_id=str(current_user.user_id))