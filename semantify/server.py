from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import zipfile
import httpx  # Make sure to install httpx for making HTTP requests
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update directory structure
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_zips")
EXTRACT_DIR = os.path.join(DATA_DIR, "extracted_files")

# Create all necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Endpoint for uploading files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Extract the zip file
    with zipfile.ZipFile(file_location, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # List extracted files
    extracted_files = os.listdir(EXTRACT_DIR)

    return JSONResponse(content={"files": extracted_files})

# endpoint for generating responses from DeepSeek
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

class GenerateRequest(BaseModel):
    prompt: str # makes sure to only take in string prompt

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    print(f"Received prompt: {request.prompt}")  # Log the received prompt
    prompt = request.prompt
    data = {
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OLLAMA_API_URL, json=data)

    if response.status_code != 200: # basically not successful request
        print(f"Error from DeepSeek: {response.status_code}, {response.text}") 
        raise HTTPException(status_code=response.status_code, detail="Error communicating with DeepSeek")

    return JSONResponse(content=response.json())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
