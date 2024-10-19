from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from analyzer import SceneAnalyzer
import logging
from typing import Dict, List
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    # Load the ML model
    logger.info("Initializing Scene Analyzer...")
    analyzer = SceneAnalyzer()
    logger.info("Scene Analyzer initialized successfully")
    yield
    # Clean up the ML models and release the resources
    del analyzer

app = FastAPI(title="Scene Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/categories")
async def get_categories() -> Dict[str, List[str]]:
    """Get available analysis categories"""
    return analyzer.get_categories()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image"""
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Only JPEG and PNG images are supported")
        
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Analyze the image
        logger.info(f"Analyzing image: {file.filename}")
        results = analyzer.analyze_image(image)
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(500, f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True, port=6000, host="0.0.0.0")