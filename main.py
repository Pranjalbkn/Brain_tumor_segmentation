import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Routers
from segmentation import router as segmentation_router
from classification import router as classification_router


app = FastAPI(
    title="NeuroScan - Brain Tumor Segmentation & Classification",
    description="Modular FastAPI app assembling segmentation and classification routers",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(segmentation_router)
app.include_router(classification_router)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def home():
    return FileResponse(os.path.join(STATIC_DIR, "home.html"))


@app.get("/segmentation")
async def segmentation_page():
    return FileResponse(os.path.join(STATIC_DIR, "segmentation.html"))


@app.get("/classification")
async def classification_page():
    return FileResponse(os.path.join(STATIC_DIR, "classification.html"))


# Mount static at the end
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
