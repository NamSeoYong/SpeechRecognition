# SpeechRecognition
An announcement voice recognition service for the hearing-impaired people based on deep learning using Python

A service that recognizes "stop" during subway announcements and recognizes "where, where, the door is on your left/right"

## Team Members
* Team Leader: Nam Seoyong (Division of Computer Science, HanYang University ERICA, Student ID : 2021075478)
* Team Member: Choi Sooyeon (Division of Computer Science, HanYang University ERICA, Student ID : 2021023118)
* Team Member: Lee Gyulim (Division of Computer Science, HanYang University ERICA, Student ID : 2021090646)

## Contents
0. [Folder Structure](#folder-structure)
1. [Develoment Setting](#development-setting)
2. [Libraries & Tools](#libraries--tools)
3. [Data-Augmentation](#data-augmentation)
4. [Noise-Reduction](#noise-reduction)
5. [Keyword-Spotting](#keyword-spotting)
6. [Run-Demo](#run-demo)

### Folder Structure

```
📦SpeechRecognition
 ┣ 📂noise-reduction
 ┃ ┣ 📂dataloader
 ┃ ┃ ┗ 📜DataLoader.py
 ┃ ┣ 📂models
 ┃ ┃ ┗ 📂tscn
 ┃ ┃ ┃ ┣ 📜loss_history.csv
 ┃ ┃ ┃ ┣ 📜TSCN_CME.pth
 ┃ ┃ ┃ ┗ 📜TSCN_CSR.pth
 ┃ ┣ 📂tscn
 ┃ ┃ ┣ 📜CME.py
 ┃ ┃ ┣ 📜CSR.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┗ 📜TSCN.py
 ┃ ┣ 📂utils
 ┃ ┃ ┗ 📜utils.py
 ┃ ┣ 📜dataset.csv
 ┃ ┣ 📜dataset_maker.py
 ┃ ┣ 📜denoise.py
 ┃ ┣ 📜README.md
 ┃ ┣ 📜report_denoise.py
 ┃ ┣ 📜requirements.txt
 ┃ ┣ 📜sd1.wav
 ┃ ┣ 📜sn1.wav
 ┃ ┗ 📜train.py
 ┣ 📂static
 ┣ 📂templates
 ┃ ┣ 📜index.html
 ┃ ┗ 📜result.html
 ┣ 📂Torch-KWT
 ┃ ┣ 📂docs
 ┃ ┃ ┗ 📜config_file_explained.md
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📜kwt.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂runs
 ┃ ┣ 📂sample_configs
 ┃ ┃ ┗ 📜base_config.yaml
 ┃ ┣ 📂utils
 ┃ ┃ ┣ 📜augment.py
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜loss.py
 ┃ ┃ ┣ 📜misc.py
 ┃ ┃ ┣ 📜opt.py
 ┃ ┃ ┣ 📜scheduler.py
 ┃ ┃ ┣ 📜trainer.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📜config_parser.py
 ┃ ┣ 📜download_gspeech_v2.sh
 ┃ ┣ 📜inference.py
 ┃ ┣ 📜kwt1_pretrained.ckpt
 ┃ ┣ 📜label_map.json
 ┃ ┣ 📜make_data_list.py
 ┃ ┣ 📜preds.json
 ┃ ┣ 📜preds_clip.json
 ┃ ┣ 📜README.md
 ┃ ┣ 📜requirements.txt
 ┃ ┣ 📜train.py
 ┃ ┗ 📜window_inference.py
 ┣ 📜demo.py
 ┣ 📜Data_Augmentation.ipynb
 ┣ 📜LICENSE.txt
 ┣ 📜main.py
 ┣ 📜preds_clip.json
 ┗ 📜README.md
```

### Development Setting
* Ubuntu 20.04
* Python 3.8.16
* PyTorch 1.12.1+cu116
* CUDA 12.1


### Libraries & Tools
* tqdm
* librosa
* pandas
* numpy
* matplotlib
* pystoi
* scipy
* openpyxl
* pyyaml >= 5.3.1
* audiomentations
* pydub
* einops
* etc...

### Data Augmentation
If you want to progress data augmentation then run [data_augmentation](Data_Augmentation.ipynb).
Only one file can do now(directory or multiple file to be implemented)

### Noise Reduction

#### Dataset
[Download Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=568)

#### How to make dataset for denoise
```
python3 noise-reduction/dataset_maker.py \
--dataset_root {datapath} \
--csv_save_path {datapath}/dataset.csv
```


##### Structure of dataset for denoise
|clean_path|noisy_path|script_path|train_val_test|
|:--:|:--:|:--:|:--:|
|share/clean_file_1.wav|share/noisy_file_1.wav|share/script_file_1.json|TR|
|share/clean_file_2.wav|share/noisy_file_2.wav|share/script_file_2.json|VA|
|...|...|...|...|
|share/clean_file_n.wav|share/noisy_file_n.wav|share/script_file_n.json|TE|

#### Training `denoise` model
```
python noise-reduction/train.py \
--model=models/tscn \
--csv_file=share/dataset.csv \
--cme_epochs=40 \
--finetune_epochs=10 \
--csr_epochs=40 \
--batch_size=8 \
--multi_gpu=True 
```

### Keyword Spotting

#### Dataset for `KWS`
You can download using with sh

```
sh Torch-KWT/download_gspeech_v2.sh <destination_path>
```
#### Training `KWS` model

Run:

```
python Torch-KWT/make_data_list.py -v <path/to/validation_list.txt> -t <path/to/testing_list.txt> -d <path/to/dataset/root> -o <output dir>
```

This will create the files `training_list.txt`, `validation_list.txt`, `testing_list.txt` and `label_map.json` at the specified output dir. 

Running `train.py` is fairly straightforward. Only a path to a config file is required. 

```
python Torch/KWTtrain.py --conf path/to/config.yaml
```

Refer to the [example config](Torch-KWT/sample_configs/base_config.yaml) to see how the config file looks like, and see the [config explanation](Torch-KWT/docs/config_file_explained.md) for a complete rundown of the various config parameters.

##### Pretrained Checkpoints

| Model Name | Test Accuracy | Link |
| ---------- | ------------- | ---- |
|    KWT-1   |     95.98*     | [kwt1-v01.pth](https://drive.google.com/uc?id=1y91PsZrnBXlmVmcDi26lDnpl4PoC5tXi&export=download) |

#### Results
You can use the model for inference,
- `inference.py`: For short ~1s clips, like the audios in the Speech Commands dataset
- `window_inference.py`: For running inference on longer audio clips, where multiple keywords may be present. Runs inference on the audio in a sliding window manner.

```
python inference.py --conf sample_configs/base_config.yaml \
                    --ckpt <path to pretrained_model.ckpt> \
                    --inp <path to audio.wav / path to audio folder> \
                    --out <output directory> \
                    --lmap label_map.json \
                    --device cpu \
                    --batch_size 8   # should be possible to use much larger batches if necessary, like 128, 256, 512 etc.

python window_inference.py --conf sample_configs/base_config.yaml \
                    --ckpt <path to pretrained_model.ckpt> \
                    --inp <path to audio.wav / path to audio folder> \
                    --out <output directory> \
                    --lmap label_map.json \
                    --device cpu \
                    --wlen 1 \
                    --stride 0.5 \
                    --thresh 0.85 \
                    --mode multi
```
There are three mode in window inference
- multi: saves all found predictions (default)
- max: saves the "most confident" prediction (outputs only a single 'clipwise; prediction for the whole clip)
- n_voting: saves the "most frequent" prediction (outputs only a single 'clipwise' prediction for the whole clip)

If you run window_inference.py with mode "max" then result is like this
```
{"/home/a/SpeechRecognition/data/denoise/b.wav": ["stop", 0.9001830816268921, 25600.0]}
```
In preds_cilp.json

### Run demo

#### How to run demo
Run `demo.py` with three arguments.
```
python demo.py --model_dir {this_file_dir} --noise_file {noise_file_name} --denoise_file {denoise_file_name}
```
Then, you can see result.

For example,
``` 
python demo.py --model ~/SpeechRecognition --noise_file handae --denosie_file de_handae
```
The result is
```
model_dir : /home/a/SpeechRecognition
noise_file : handae.wav
denoise_file : de_handae.wav
finish denoise 5.694255113601685 sec

finish kws 3.3709611892700195 sec

result : anyang university at anzan hanyang university at anzan the doors are on your left

4.017111778259277 sec
```

#### How to run demo with FastAPI
1. Install FastAPI
```
pip install fastapi uvicorn
```
2. Start FastAPI
```
uvicorn main:app --reload
```
3. Open website to `http://localhost:8080`

4. Enter 3 arguments same as demo