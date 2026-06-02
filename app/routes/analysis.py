from fastapi import APIRouter
from pydantic import BaseModel

from app.services.llm_service import ask_model

router = APIRouter(prefix="/analysis")

class QuestionRequest(BaseModel):
    question: str


@router.post("/trending")
def trending_analysis(request: QuestionRequest):

    response = ask_model(request.question)

    return {
        "response": response
    }