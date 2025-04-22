from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import shutil
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "best.pt"

class ProcessRequest(BaseModel):
    input_folder: str
    output_folder: str

def load_model():
    """Load YOLO model"""
    return YOLO(MODEL_PATH)

def process_image(image_path: str, model) -> List[Dict[str, Any]]:
    """Process image using YOLO model"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Make prediction
    results = model.predict(img)
    
    # Extract predictions
    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            predictions.append({
                'class': int(cls),
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    
    return predictions

def get_image_files(input_folder: str) -> List[str]:
    """Get all image files from input folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    image_files = []
    for ext in image_extensions:
        image_files.extend([str(f) for f in input_path.glob(f"*{ext}")])
        image_files.extend([str(f) for f in input_path.glob(f"*{ext.upper()}")])
    
    return image_files

@app.post("/process")
async def process_images(request: ProcessRequest):
    try:
        # Ensure folders exist
        input_path = Path(request.input_folder)
        output_path = Path(request.output_folder)
        
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=f"Input folder does not exist: {request.input_folder}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model = load_model()
        
        # Get all image files
        image_files = get_image_files(request.input_folder)
        
        if not image_files:
            return {
                "message": "No images found in the input folder",
                "processed_files": []
            }
        
        # Process each image
        processed_files = []
        for image_path in image_files:
            try:
                # Process image
                predictions = process_image(image_path, model)
                
                # Save predictions
                image_name = Path(image_path).stem
                output_file = output_path / f"{image_name}.json"
                
                with open(output_file, 'w') as f:
                    json.dump(predictions, f)
                
                processed_files.append({
                    "input_file": image_path,
                    "output_file": str(output_file),
                    "predictions_count": len(predictions)
                })
                
            except Exception as e:
                processed_files.append({
                    "input_file": image_path,
                    "error": str(e)
                })
        
        return {
            "message": "Batch processing completed",
            "total_files": len(image_files),
            "processed_files": processed_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 