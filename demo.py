import subprocess, json, torch, librosa, time
from argparse import ArgumentParser
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from scipy.io import wavfile
import numpy as np

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
    
    print("result :", transcription, "\n")

def main(args):
    start_time = time.time()
    denoise(args.model_dir, args.noise_file, args.denoise_file)
    keywordspotting(args.model_dir, args.denoise_file)
    start = read_kws_result(args.model_dir, args.denoise_file)
    speech_to_text(args.model_dir, args.denoise_file, start)
    end = time.time()
    print(end - start_time, "sec\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model_file")
    parser.add_argument("--noise_file", type=str, required=True, help="Name of noisefile except .wav")
    parser.add_argument("--denoise_file", type=str, required=True, help="Name of noisefile except .wav")
    
    args = parser.parse_args()

    main(args)

