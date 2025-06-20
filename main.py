
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
import tempfile
import os
import time
import uuid
from gtts import gTTS
from typing import Optional
import logging
from pathlib import Path
from moviepy.editor import AudioFileClip  # لتحويل mp3 إلى wav
import sys
import os
import subprocess

# Download model if not exists
if not os.path.exists("Wav2Lip/checkpoints/wav2lip_gan.pth"):
    subprocess.run(["python", "download_model.py"])
    
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Nefertiti AI API",
    description="AI-powered chatbot that generates video responses as Queen Nefertiti",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WAV2LIP_MODEL_PATH = "wav2lip_gan.pth"
TEMP_DIR = "temp_files"
MAX_CLEANUP_AGE = 3600

os.makedirs(TEMP_DIR, exist_ok=True)

NEFERTITI_KNOWLEDGE = {
    "who are you": "I am Nefertiti, Great Royal Wife of Pharaoh Akhenaten. My name means 'the beautiful one has come'.",
    "tell me about your reign": "We established the cult of Aten, moved Egypt's capital to Amarna, and revolutionized art with realistic depictions.",
    "how did you die": "My fate remains a mystery—some say I ruled as pharaoh after Akhenaten, others that I fell from favor.",
    "what was your role": "I was not just a queen, but a powerful co-ruler who helped shape one of history's most revolutionary periods.",
    "tell me about akhenaten": "My beloved husband was a visionary pharaoh who dared to challenge Egypt's religious traditions by worshipping one god, Aten.",
    "what is your legacy": "My beauty, power, and mystery have captivated the world for millennia. My bust in Berlin remains an icon of ancient elegance.",
    "tell me about amarna": "Amarna was our magnificent capital, a city built from nothing to honor Aten and represent our new vision for Egypt.",
    "default": "My legacy endures through my iconic bust, now in Berlin's Neues Museum. Ask me about my reign, my role, or ancient Egypt."
}

class QuestionRequest(BaseModel):
    question: str
    language: Optional[str] = "en"

class VideoResponse(BaseModel):
    video_id: str
    message: str
    processing_time: Optional[float] = None

class StatusResponse(BaseModel):
    video_id: str
    status: str
    message: str

current_nefertiti_image = None

def cleanup_old_files():
    try:
        current_time = time.time()
        for file_path in Path(TEMP_DIR).glob("*"):
            if current_time - file_path.stat().st_mtime > MAX_CLEANUP_AGE:
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def get_response_text(question: str) -> str:
    question_lower = question.lower().strip().rstrip('?')
    if question_lower in NEFERTITI_KNOWLEDGE:
        return NEFERTITI_KNOWLEDGE[question_lower]
    for key, value in NEFERTITI_KNOWLEDGE.items():
        if key != "default" and any(word in question_lower for word in key.split()):
            return value
    return NEFERTITI_KNOWLEDGE["default"]

def generate_speech(text: str, language: str = "en") -> str:
    audio_id = str(uuid.uuid4())
    mp3_path = os.path.join(TEMP_DIR, f"audio_{audio_id}.mp3")
    wav_path = os.path.join(TEMP_DIR, f"audio_{audio_id}.wav")
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(mp3_path)
        audioclip = AudioFileClip(mp3_path)
        audioclip.write_audiofile(wav_path, fps=16000)
        audioclip.close()
        return wav_path
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate speech")

def create_lip_sync_video(image_path: str, audio_path: str) -> str:
    video_id = str(uuid.uuid4())
    output_path = os.path.join(TEMP_DIR, f"video_{video_id}.mp4")
    try:
        cmd = [
            sys.executable, "Wav2Lip/inference.py",
            "--checkpoint_path", WAV2LIP_MODEL_PATH,
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--face_det_batch_size", "1",
            "--wav2lip_batch_size", "1"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Wav2Lip error: {result.stderr}")
            raise HTTPException(status_code=500, detail="Failed to generate lip-sync video")
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Video generation failed - output file not found")
        return output_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Video generation timed out")
    except Exception as e:
        logger.error(f"Error creating lip-sync video: {e}")
        raise HTTPException(status_code=500, detail="Failed to create lip-sync video")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Nefertiti AI API...")
    if not os.path.exists(WAV2LIP_MODEL_PATH):
        logger.warning(f"Wav2Lip model not found at {WAV2LIP_MODEL_PATH}")
    if not os.path.exists("Wav2Lip"):
        logger.warning("Wav2Lip directory not found. Please ensure Wav2Lip is properly installed.")

@app.post("/upload-image/")
async def upload_nefertiti_image(file: UploadFile = File(...)):
    global current_nefertiti_image
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        image_id = str(uuid.uuid4())
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        image_path = os.path.join(TEMP_DIR, f"nefertiti_{image_id}.{file_extension}")
        with open(image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        current_nefertiti_image = image_path
        return {"message": "Nefertiti image uploaded successfully", "image_id": image_id, "filename": file.filename}
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

@app.post("/ask-nefertiti/", response_model=VideoResponse)
async def ask_nefertiti(request: QuestionRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    if not current_nefertiti_image or not os.path.exists(current_nefertiti_image):
        raise HTTPException(status_code=400, detail="Please upload a Nefertiti image first using /upload-image/")
    try:
        background_tasks.add_task(cleanup_old_files)
        response_text = get_response_text(request.question)
        logger.info(f"Question: {request.question}")
        logger.info(f"Response: {response_text}")
        audio_path = generate_speech(response_text, request.language)
        video_path = create_lip_sync_video(current_nefertiti_image, audio_path)
        processing_time = time.time() - start_time
        video_id = os.path.basename(video_path).replace("video_", "").replace(".mp4", "")
        return VideoResponse(video_id=video_id, message="Video generated successfully", processing_time=round(processing_time, 2))
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-video/{video_id}")
async def download_video(video_id: str):
    video_path = os.path.join(TEMP_DIR, f"video_{video_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"nefertiti_response_{video_id}.mp4")

@app.get("/available-questions/")
async def get_available_questions():
    questions = [key.title() + "?" for key in NEFERTITI_KNOWLEDGE.keys() if key != "default"]
    return {"questions": questions, "total": len(questions)}

@app.get("/")
async def root():
    return {"message": "Nefertiti AI API is running", "version": "1.0.0", "status": "active", "has_nefertiti_image": current_nefertiti_image is not None}
