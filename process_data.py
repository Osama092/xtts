import argparse
import os
import sys
import tempfile
from pathlib import Path

import shutil
import glob

import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list, find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Clear logs
def remove_log_file(file_path):
    log_file = Path(file_path)
    if log_file.exists() and log_file.is_file():
        log_file.unlink()

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)

    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting=True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file

def load_params_tts(out_path, version):
    out_path = Path(out_path)

    ready_model_path = out_path / "ready"

    vocab_path = ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path = ready_model_path / "speakers_xtts.pth"
    reference_path = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
            return "Params for TTS not found", "", "", ""

    return "Params for TTS loaded", model_path, config_path, vocab_path, speaker_path, reference_path

import os

def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path):
    print(f"Audio Path: {audio_path}")
    print(f"Output Path: {out_path}")
    print(f"Language: {language}")
    print(f"Whisper Model: {whisper_model}")

    clear_gpu_cache()

    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)

    if audio_folder_path:
        audio_files = list(list_audios(audio_folder_path))
    else:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_files = [audio_path]

    if not audio_files:
        return "No audio files found! Please provide files via command line or specify a folder path.", "", ""
    else:
        try:
            # Loading Whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "float32"
            asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
            train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path)
        except Exception as e:
            traceback.print_exc()
            return f"The data processing was interrupted due to an error!! Please check the console to verify the full error message! \n Error summary: {e}", "", ""

    if audio_total_size < 120:
        message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
        print(message)
        return message, "", ""

    print("Dataset Processed!")
    return "Dataset Processed!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved). Default: output/",
        default=str(Path.cwd() / "finetune_models"),
    )


    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to audio file(s).",
        default="",
    )
    parser.add_argument(
        "--audio_folder_path",
        type=str,
        help="Path to folder containing audio files.",
        default="",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language for dataset.",
        default="en",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        help="Whisper model to use.",
        default="large-v3",
    )

    args = parser.parse_args()

    if args.audio_path or args.audio_folder_path:
        preprocess_dataset(args.audio_path, args.audio_folder_path, args.language, args.whisper_model, args.out_path)
    else:
        print("Please provide either --audio_path or --audio_folder_path.")

#Single audio
#python xtts_cli.py --audio_path /home/user/audio/example.wav --out_path /home/user/output --language en --whisper_model large-v3
#Example 
#python process_data.py --audio_path "C:\Users\oussa\Downloads\New folder\min joy sample.mp3" --out_path "C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models" --language en --whisper_model small

#Audio foler
#python xtts_cli.py --audio_folder_path /home/user/audio_folder --out_path /home/user/output --language en --whisper_model large-v3
