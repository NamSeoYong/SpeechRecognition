from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import subprocess
import json
import torch
import librosa
import time
from argparse import ArgumentParser
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from scipy.io import wavfile
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class TranscriptRequest(BaseModel):
    model_dir: str
    noise_file: str
    denoise_file: str

class TranscriptResponse(BaseModel):
    transcription: str

def run_command(command):
    subprocess.run(command)

def denoise(model_dir, noise_file, denoise_file):
    command = [
        'python',
        f'{model_dir}/noise-reduction/denoise.py',
        f'--model={model_dir}/noise-reduction/models/tscn',
        f'--noisy={model_dir}/data/original/{noise_file}.wav',
        f'--denoise={model_dir}/data/denoise/{denoise_file}.wav'
    ]
    run_command(command)

def keywordspotting(model_dir, denoise_file):
    command = [
        'python',
        f'{model_dir}/Torch-KWT/window_inference.py',
        '--conf', f'{model_dir}/Torch-KWT/sample_configs/base_config.yaml',
        '--ckpt', f'{model_dir}/Torch-KWT/runs/exp-0.0.2/best.pth',
        '--inp', f'{model_dir}/data/denoise/{denoise_file}.wav',
        '--lmap', f'{model_dir}/Torch-KWT/label_map.json',
        '--wlen', '0.5',
        '--mode', 'max'
    ]
    run_command(command)

def read_kws_result(model_dir, denoise_file):
    file_path = f"{model_dir}/"

    with open(file_path+"preds_clip.json", encoding='utf-8') as f:
        data = json.load(f)
        start = data[f"{model_dir}/data/denoise/{denoise_file}.wav"][2]
        #print(start)

    return int(start)

def speech_to_text(model_dir, denoise_file, start):
    file_name = f'{model_dir}/data/denoise/{denoise_file}.wav'

    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

    # Read audio file
    data = wavfile.read(file_name)
    framerate = data[0]
    sounddata = data[1]
    time = np.arange(0, len(sounddata)) / framerate

    # Load audio using librosa
    input_audio, _ = librosa.load(file_name, sr=framerate)

    # Tokenize input audio
    input_values = processor(input_audio[start:], sampling_rate=framerate, return_tensors="pt").input_values

    # Perform speech-to-text inference
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe_audio(request: Request, transcript_request: TranscriptRequest):
    model_dir = transcript_request.model_dir
    noise_file = transcript_request.noise_file
    denoise_file = transcript_request.denoise_file

    denoise(model_dir, noise_file, denoise_file)
    keywordspotting(model_dir, denoise_file)
    start = read_kws_result(model_dir, denoise_file)
    transcription = speech_to_text(model_dir, denoise_file, start)

    return templates.TemplateResponse("result.html", {"request": request, "transcription": transcription})

@app.get("/result")
def show_result(request: Request, transcription: str):
    # Render the result page template with the transcription
    return templates.TemplateResponse("result.html", {"request": request, "transcription": transcription})