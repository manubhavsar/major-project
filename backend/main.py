from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from process import run_predictions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's local React UI
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/charts", StaticFiles(directory="../charts"), name="charts")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/api/predict")
async def predict_engagement(file: UploadFile = File(...)):
    # Save the file uniquely
    extension = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        results = run_predictions(file_path)
        return {
            "success": True,
            "filename": file.filename,
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Cleanup files to prevent disk leak over time
        if os.path.exists(file_path):
            os.remove(file_path)
        audio_path = file_path.rsplit('.', 1)[0] + ".wav"
        if os.path.exists(audio_path) and file_path != audio_path:
            os.remove(audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
