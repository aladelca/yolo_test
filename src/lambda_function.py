import os
import json
import boto3
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

# Configuration
MODEL_BUCKET = 'project-mma'
MODEL_KEY = 'model/best.pt'
BUCKET = 'project-mma'

def get_model_from_s3():
    """Get model directly from S3"""
    logger.info(f"Downloading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
    try:
        response = s3_client.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        model_data = response['Body'].read()
        logger.info("Model downloaded successfully")
        return model_data
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def process_image(image_data, model_data):
    """Process image using YOLO model"""
    logger.info("Processing image with YOLO model")
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Load model from bytes and make prediction
        model = YOLO(model_data)
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
        
        logger.info(f"Image processed successfully. Found {len(predictions)} objects")
        return predictions
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def get_image_files(input_prefix):
    """Get all image files from input bucket"""
    logger.info(f"Listing images in s3://{BUCKET}/{input_prefix}")
    paginator = s3_client.get_paginator('list_objects_v2')
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_files = []
    try:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=input_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(key)
        
        logger.info(f"Found {len(image_files)} images in the input bucket")
        return image_files
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise

def lambda_handler(event, context):
    logger.info("Lambda function started")
    try:
        # Parse the request body
        body = json.loads(event['body'])
        logger.info(f"Request body: {json.dumps(body)}")
        
        # Get parameters from the request
        input_prefix = body.get('input_prefix', 'images/')
        output_prefix = body.get('output_prefix', 'results/')
        
        # Ensure prefixes end with '/'
        if not input_prefix.endswith('/'):
            input_prefix += '/'
        if not output_prefix.endswith('/'):
            output_prefix += '/'
        
        logger.info(f"Input prefix: {input_prefix}")
        logger.info(f"Output prefix: {output_prefix}")
        
        # Get model from S3
        model_data = get_model_from_s3()
        
        # Get all image files from input bucket
        image_files = get_image_files(input_prefix)
        
        if not image_files:
            logger.warning("No images found in the input bucket")
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'No images found in the input bucket',
                    'processed_files': []
                })
            }
        
        # Process each image
        processed_files = []
        for image_key in image_files:
            try:
                logger.info(f"Processing image: {image_key}")
                
                # Get image from S3
                response = s3_client.get_object(Bucket=BUCKET, Key=image_key)
                image_data = response['Body'].read()
                logger.info(f"Downloaded image: {image_key}")
                
                # Process image
                predictions = process_image(image_data, model_data)
                
                # Save predictions to S3
                relative_path = os.path.relpath(image_key, input_prefix)
                output_key = os.path.join(output_prefix, os.path.splitext(relative_path)[0] + '.json')
                
                s3_client.put_object(
                    Bucket=BUCKET,
                    Key=output_key,
                    Body=json.dumps(predictions),
                    ContentType='application/json'
                )
                logger.info(f"Saved predictions to: {output_key}")
                
                processed_files.append({
                    'input_file': image_key,
                    'output_file': output_key,
                    'predictions_count': len(predictions)
                })
                
            except Exception as e:
                logger.error(f"Error processing image {image_key}: {str(e)}")
                processed_files.append({
                    'input_file': image_key,
                    'error': str(e)
                })
        
        logger.info(f"Batch processing completed. Processed {len(image_files)} files")
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'Batch processing completed',
                'total_files': len(image_files),
                'processed_files': processed_files
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda function failed: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        } 