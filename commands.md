# Command Line Instructions



## 1. Process Data

### Single Audio File
**Command:**
```bash
python xtts_cli.py --audio_path /home/user/audio/example.wav --out_path /home/user/output --language en --whisper_model small
```

**Example:**
```bash
python xtts_cli.py --audio_path "C:\Users\oussa\Downloads\New folder\min joy sample.mp3" --out_path "C:\Users\oussa\Downloads\output" --language en --whisper_model small
```


### Audio Folder
**Command:**
```bash
python xtts_cli.py --audio_folder_path /home/user/audio_folder --out_path /home/user/output --language en --whisper_model small
```

## 2. Train Data

**Command:**
```bash
python train_data.py --train_csv "path/to/train.csv" --eval_csv "path/to/eval.csv" --custom_model "path/to/custom_model.pth" --version "v2.0.2" --num_epochs 10 --batch_size 4 --grad_acumm 2 --max_audio_length 15 --output_path "path/to/output"
```

**Example:**
```bash
python train_data.py --train_csv "C:\Users\oussa\Downloads\train.csv" --eval_csv "C:\Users\oussa\Downloads\eval.csv" --custom_model "C:\Users\oussa\Downloads\custom_model.pth" --version "v2.0.2" --num_epochs 10 --batch_size 4 --grad_acumm 2 --max_audio_length 15 --output_path "C:\Users\oussa\Downloads\output"
```

### Optimize Model
**Command:**
```bash
python train_data.py --optimize
```

## 3. Inference

**Command:**
```bash
python inference.py --out_path "path/to/output" --speaker_reference_audio "path/to/reference.wav" --tts_text "Your text here" --xtts_checkpoint "path/to/checkpoint.pth" --xtts_config "path/to/config.json" --xtts_vocab "path/to/vocab.json" --xtts_speaker "path/to/speaker.pth" --version "v2.0.2" --temperature 0.75 --length_penalty 1 --repetition_penalty 5 --top_k 50 --top_p 0 --sentence_split --use_config
```

**Example:**
```bash
python inference.py --tts_text "This is a sample voice using text to speech" --out_path "C:\Users\oussa\Downloads\output" --speaker_reference_audio "C:\Users\oussa\Downloads\reference.wav" --xtts_checkpoint "C:\Users\oussa\Downloads\checkpoint.pth" --xtts_config "C:\Users\oussa\Downloads\config.json" --xtts_vocab "C:\Users\oussa\Downloads\vocab.json" --xtts_speaker "C:\Users\oussa\Downloads\speaker.pth" --version "v2.0.2" --temperature 0.75 --length_penalty 1 --repetition_penalty 5 --top_k 50 --top_p 0 --sentence_split --use_config
```