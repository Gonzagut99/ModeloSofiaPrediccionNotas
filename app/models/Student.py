from typing import Optional
from sqlmodel import Field,Relationship, SQLModel

# Definir el esquema de entrada
class StudentModel(SQLModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    math_score: Optional[int] = None
    reading_score: Optional[int] = None
    writing_score: Optional[int] = None
    average_score: Optional[float] = None