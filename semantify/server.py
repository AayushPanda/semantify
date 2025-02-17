
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import zipfile
import httpx  # Make sure to install httpx for making HTTP requests
from pydantic import BaseModel
from fastapi.responses import FileResponse

import pandas as pd
import asyncio
import numpy as np
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import organiser, rag


# Thread safe globals
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings_df, organised
    embeddings_df = pd.DataFrame()
    organised = False
    yield  # FastAPI will run the app while this context is active

app = FastAPI(lifespan=lifespan)
state_lock = asyncio.Lock()  # thread safety

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

# Model outputs
ORGANISED_DIR = os.path.join(DATA_DIR, "organised_files")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
ZIP_DIR = os.path.join(DATA_DIR, "zips")

# Create all necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(ORGANISED_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

@app.get("/embeddings")
async def get_embeddings():
    if not organised:
        raise HTTPException(status_code=400, detail="No embeddings to return")
    # return the json saved by calling main_worker. might be better in the future to just convert the dataframe to json here
    return FileResponse(os.path.join(EMBEDDINGS_DIR, "embeddings.json"))

# Downlaoding organised files
@app.get("/download")
async def download_generated_folder() -> FileResponse:
    if not organised:
        raise HTTPException(status_code=400, detail="No organised files to download")
    shutil.make_archive(os.path.join(ZIP_DIR, "organised_files"), 'zip', DATA_DIR, "organised_files")
    return FileResponse(os.path.join(ZIP_DIR, "organised_files.zip"))
    ...

# Endpoint for uploading files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global embeddings_df, organised
    
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    async with state_lock:
        organised = False
        embeddings_df = pd.DataFrame()

    # TODO: extract without preserving directory structure
    # pretty sure the code below works but dont want to deal with it until im done other work    
    # for filename in zip_ref.namelist():
    #     file = zip_ref.open(filename)
    #     with open(os.path.join(EXTRACT_DIR, filename), "wb") as out:
    #         out.write(file.read())

    # Extract the zip file
    with zipfile.ZipFile(file_location, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # List extracted files
    extracted_files = os.listdir(EXTRACT_DIR)

    async with state_lock:
        embeddings_df = organiser.main_worker(EXTRACT_DIR, ORGANISED_DIR, EMBEDDINGS_DIR)
        organised = True

    return JSONResponse(content={"files": extracted_files})

# endpoint for generating responses from DeepSeek
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

class GenerateRequest(BaseModel):
    prompt: str # makes sure to only take in string prompt

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    print(f"Received prompt: {request.prompt}")  # Log the received prompt

    if embeddings_df.empty:
        raise HTTPException(status_code=400, detail="No embeddings available to generate response")

    prompt = request.prompt

    hits = rag.rag(prompt, np.vstack(embeddings_df["embedding"].to_list()))
    ragtext = "\n\n".join(embeddings_df.loc[hits]["text"].to_list())
    prompt = f"""
    Using the following text segments to generate a response to the prompt: {request.prompt}.
    {ragtext}
    """

    print(f"Prompt after RAG: {prompt}")

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

