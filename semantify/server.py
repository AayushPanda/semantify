from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import zipfile

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
