import boto3
import requests
import os

def upload_mediapipe_to_s3():
    # Configuration
    BUCKET_NAME = 'my-ekyc-assets'
    S3_KEY = 'models/ekyc/blaze_face_short_range.tflite'
    SOURCE_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    TEMP_FILE = "blaze_face_short_range.tflite"

    try:
        # 1. Download the file from MediaPipe
        print(f"Downloading from {SOURCE_URL}...")
        response = requests.get(SOURCE_URL)
        response.raise_for_status()
        
        with open(TEMP_FILE, "wb") as f:
            f.write(response.content)
        
        # 2. Upload to S3
        print(f"Uploading to s3://{BUCKET_NAME}/{S3_KEY}...")
        s3 = boto3.client('s3')
        s3.upload_file(TEMP_FILE, BUCKET_NAME, S3_KEY)
        
        print("Done! MediaPipe model is now in your S3 bucket.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup local file
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)

if __name__ == "__main__":
    upload_mediapipe_to_s3()