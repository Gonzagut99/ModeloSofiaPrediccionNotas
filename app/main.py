from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.single_course import single_course_predict_router

app = FastAPI()
app.title = "Sofia predictions API"
app.version = "0.0.1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(single_course_predict_router)

@app.get('/', tags = ['home'])
async def home():
    return {"message": "Welcome to the Sofia predictions API"}