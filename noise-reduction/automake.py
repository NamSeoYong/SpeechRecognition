import os

# 디렉토리 내 파일들의 datapath를 가져옴
directory = "data"  # 디렉토리 경로 수정
files = os.listdir(directory)
datapaths = [os.path.join(directory, file) for file in files]

# 반복문으로 명령어 처리
for i, datapath in enumerate(datapaths):
    command = f"python denoise.py --model=models/tscn --noisy={datapath} --denoise=denoise_data/de_{i}.wav"
    os.system(command)

