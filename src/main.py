from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.chat_module.chat_controller import chat_router
from src.logs import configure_logging, LogLevels
import logging

configure_logging(LogLevels.debug)
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message":"Api is Working Live"}