
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

import logging

import pandas as pd
import asyncio
import numpy as np
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import organiser, rag


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True  # Override existing logging configs
)


MODEL = 'llama2-uncensored'

# Thread safe globals
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings_df, organised
    embeddings_df = pd.DataFrame()
    organised = False
    yield  # FastAPI will run the app while this context is active
    # Clean up
    embeddings_df = pd.DataFrame()
    organised = False
    shutil.rmtree(UPLOAD_DIR)
    shutil.rmtree(EXTRACT_DIR)
    shutil.rmtree(ORGANISED_DIR)
    shutil.rmtree(EMBEDDINGS_DIR)
    shutil.rmtree(ZIP_DIR)
    shutil.rmtree(DATA_DIR)


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
os.makedirs(ZIP_DIR, exist_ok=True)

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
        embeddings_df["file"] = embeddings_df["file"].apply(lambda x: os.path.basename(x))
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
    if hits == []:
        ragfiles = []
        ragtext = "No relevant content found to generate a response. Respond asking for a diifferent prompt" 
    else:
        ragfiles = embeddings_df[["file", "cluster-path"]].loc[hits]
        print(ragfiles.head())
        ragfiles = ragfiles.apply(lambda row: os.path.join(row['cluster-path'], os.path.basename(row['file'])), axis=1).to_list()

    ragtext = "\n\n".join(embeddings_df.loc[hits]["text"].to_list())
    prompt = f"""
    Given the following text segments, please use only the most relevant content to generate a response to the prompt:

    Prompt: {request.prompt}

    Relevant Segments:
    {ragtext}

    If no relevant content is found, please inform the user with the message: "No relevant content found. Please refine your query."
    
    Make sure to focus exclusively on information that directly addresses the prompt, excluding any irrelevant details. Discuss extensively with respect to the prompt and segment content.
    """

    # print(f"Prompt after RAG: {prompt}")

    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OLLAMA_API_URL, json=data, timeout=None)

    if response.status_code != 200: # basically not successful request
        print(f"Error from DeepSeek: {response.status_code}, {response.text}") 
        raise HTTPException(status_code=response.status_code, detail="Error communicating with DeepSeek")
    
    response_json = {}
    response_json["response"] = response.json()["response"]
    response_json["files"] = ragfiles
    
    return JSONResponse(content=response_json)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

