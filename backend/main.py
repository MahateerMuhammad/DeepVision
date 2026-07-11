from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.cnn import router as cnn_router
from app.routers.network import router as network_router
from app.routers.training import router as training_router

app = FastAPI(
    title="Deep Learning Visualizer API",
    description="Backend for Deep Learning Visualizer",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Deep Learning Visualizer API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(network_router)
app.include_router(cnn_router)
app.include_router(training_router)
