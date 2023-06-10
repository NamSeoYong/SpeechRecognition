# SpeechRecognition
An announcement voice recognition service for the hearing-impaired people based on deep learning using Python


## Team Members
* Team Leader: Nam Seoyong (Division of Computer Science, HanYang University ERICA, Student ID : 2021075478)
* Team Member: Choi Sooyeon (Division of Computer Science, HanYang University ERICA, Student ID : 2021023118)
* Team Member: Lee Gyulim (Division of Computer Science, HanYang University ERICA, Student ID : 2021090646)


## Development Setting
* Ubuntu 20.04
* Python 3.8.16
* PyTorch 1.12.1+cu116
* CUDA 12.1


## Libraries & Tools
* tqdm
* librosa
* pandas
* numpy
* matplotlib
* pystoi
* scipy
* openpyxl


## How to Noise Reduction

❏ dataset.csv(input csv) 생성 방법입니다.
```
python3 dataset_maker.py \
--dataset_root {datapath} \
--csv_save_path {datapath}/dataset.csv
```


❏ dataset.csv 구조
|clean_path|noisy_path|script_path|train_val_test|
|:--:|:--:|:--:|:--:|
|share/clean_file_1.wav|share/noisy_file_1.wav|share/script_file_1.json|TR|
|share/clean_file_2.wav|share/noisy_file_2.wav|share/script_file_2.json|VA|
|...|...|...|...|
|share/clean_file_n.wav|share/noisy_file_n.wav|share/script_file_n.json|TE|

❏ 훈련 방법입니다.
```
python train.py \
--model=models/tscn \
--csv_file=share/dataset.csv \
--cme_epochs=50 \
--finetune_epochs=10 \
--csr_epochs=20 \
--batch_size=16 \
--multi_gpu=True 
```

❏ 소음 억제 방법입니다.
```
python denoise.py \
--model=models/tscn \
--noisy=sn1.wav \
--denoise=de1.wav \
--clean=sd1.wav
python denoise.py \
--model=models/tscn \
--csv_file=share/dataset.csv \
--output_dir=share/denoise
```

|comment|wav player|
|:--:|:--:|
|입력| https://user-images.githubusercontent.com/65753560/143393711-c9ec37a0-95ef-407f-8e72-444553c43bc0.mp4 |
|출력| https://user-images.githubusercontent.com/65753560/143393778-9dc9331c-915a-4555-b4f8-4197a575420f.mp4 |
|정답| https://user-images.githubusercontent.com/65753560/143393794-f40d689c-9892-49bc-81d4-c28a3a5aeb18.mp4 |

# NIA 2022 noise-reduction  
❏ NIA 2022 AI 학습용 데이터로 8:1:1 훈련, 검증, 실험 분할 학습 진행  
```
NIA 2022 noise-reduction 데이터 총 2082h -> train 1666h valid 208h test 208h  
```
※ 전체 데이터는 [AI - HUB](https://aihub.or.kr/)에서 받을 수 있습니다.  


❏ 훈련된 모델의 ESTOI 수치 결과입니다.  
||**TSCN**|
|:--:|:--:|
|**ESTOI**|0.85|
|**F1-score-error-rate**|0.18|
