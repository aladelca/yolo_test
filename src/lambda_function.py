import os
import json
import boto3
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

s3_client = boto3.client('s3')

# Model configuration
MODEL_BUCKET = 'project-mma'
MODEL_KEY = 'model/best.pt'

def get_model_from_s3():
    """Get model directly from S3"""
    response = s3_client.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    model_data = response['Body'].read()
    return model_data

def process_image(image_data, model_data):
    """Process image using YOLO model"""
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
    
    return predictions

def get_image_files(bucket, prefix=''):
    """Get all image files from S3 bucket"""
    paginator = s3_client.get_paginator('list_objects_v2')
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if any(key.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(key)
    
    return image_files

def lambda_handler(event, context):
    try:
        # Parse the request body
        body = json.loads(event['body'])
        
        # Get parameters from the request
        input_bucket = body.get('input_bucket')
        output_bucket = body.get('output_bucket')
        
        if not all([input_bucket, output_bucket]):
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required parameters: input_bucket and output_bucket are required'
                })
            }
        
        # Get model from S3
        model_data = get_model_from_s3()
        
        # Get all image files from input bucket
        image_files = get_image_files(input_bucket)
        
        if not image_files:
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
                # Get image from S3
                response = s3_client.get_object(Bucket=input_bucket, Key=image_key)
                image_data = response['Body'].read()
                
                # Process image
                predictions = process_image(image_data, model_data)
                
                # Save predictions to S3
                output_key = f"predictions/{os.path.splitext(image_key)[0]}.json"
                
                s3_client.put_object(
                    Bucket=output_bucket,
                    Key=output_key,
                    Body=json.dumps(predictions),
                    ContentType='application/json'
                )
                
                processed_files.append({
                    'input_file': image_key,
                    'output_file': output_key,
                    'predictions_count': len(predictions)
                })
                
            except Exception as e:
                processed_files.append({
                    'input_file': image_key,
                    'error': str(e)
                })
        
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