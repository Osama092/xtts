import argparse
import os
import sys
import tempfile
from pathlib import Path

import os
import shutil
import glob

import gradio as gr
import librosa.display
import numpy as np

import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list,find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

#torch.cuda.set_per_process_memory_fraction(0.5, 0)
# Adjust the fraction as needed


# Clear logs
def remove_log_file(file_path):
     log_file = Path(file_path)

     if log_file.exists() and log_file.is_file():
         log_file.unlink()

# remove_log_file(str(Path.cwd() / "log.out"))

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab,xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab,speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting = True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature, # Add custom parameters here
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting = sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Speech generated successfully! File saved at: {out_path}")
    else:
        print("Failed to generate speech or the file is empty.")

    return "Speech generated !", out_path, speaker_audio_file




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning and optimization demo"""
    )

    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved). Default: output/",
        default=str(Path.cwd() / "finetune_models"),
    )

    parser.add_argument(
        "--speaker_reference_audio",
        type=str,
        help="The used audio in creating the data",
        default=str(Path.cwd() / "finetune_models" / "ready" / "reference.wav"),
    )

    parser.add_argument(
        "--tts_text",
        type=str,
        help="The text that getting ttsed",
        default="This model sounds really good and above all, it's reasonably fast.",
    )

    parser.add_argument(
        "--xtts_checkpoint",
        type=str,
        help="Path to the xtts_checkpoint file.",
        default=str(Path.cwd() / "finetune_models" / "ready" / "unoptimize_model.pth")
    )

    parser.add_argument(
        "--xtts_config",
        type=str,
        help="Path to the xtts_config file.",
        default=str(Path.cwd() / "finetune_models" / "ready" / "config.json")
    )

    parser.add_argument(
        "--xtts_vocab",
        type=str,
        help="Path to the xtts_vocab file.",
        default=str(Path.cwd() / "finetune_models" / "ready" / "vocab.json")
    )

    parser.add_argument(
        "--xtts_speaker",
        type=str,
        help="Path to the xtts_speaker file.",
        default=str(Path.cwd() / "finetune_models" / "ready" / "speakers_xtts.pth")
    )

    parser.add_argument(
        "--version",
        type=str,
        help="XTTS base version. Default: 'v2.0.2'",
        default="v2.0.2",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Number of temperature. Default: 0.75",
        default=0.75,
    )

    parser.add_argument(
        "--length_penalty",
        type=int,
        help="Number of length_penalty. Default: 1",
        default=1,
    )

    parser.add_argument(
        "--repetition_penalty",
        type=int,
        help="Number of epochs to repetition_penalty. Default: 5",
        default=5,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        help="Number of top_k. Default: 50",
        default=50,
    )

    parser.add_argument(
        "--top_p",
        type=int,
        help="Number of top_p. Default: 0",
        default=0,
    )

    parser.add_argument(
        "--sentence_split",
        action='store_true',
        help="Enable sentence splitting. Default: False"
    )

    parser.add_argument(
        "--use_config",
        action='store_true',
        help="Use config values. Default: False"
    )

    args = parser.parse_args()

    # Load the XTTS model
    result = load_model(
        xtts_checkpoint=args.xtts_checkpoint,
        xtts_config=args.xtts_config,
        xtts_vocab=args.xtts_vocab,
        xtts_speaker=args.xtts_speaker
    )
    print(result)  # Print the model loading result

    # Ensure the model was loaded successfully
    if XTTS_MODEL is None:
        print("Model loading failed. Exiting.")
        sys.exit(1)

    # Call the run_tts function
    lang = "en"  # Set your desired language here
    tts_text = args.tts_text
    speaker_audio_file = args.speaker_reference_audio
    temperature = args.temperature
    length_penalty = args.length_penalty
    repetition_penalty = args.repetition_penalty
    top_k = args.top_k
    top_p = args.top_p
    sentence_split = args.sentence_split
    use_config = args.use_config

    result, out_path, speaker_audio_file = run_tts(
        lang=lang,
        tts_text=tts_text,
        speaker_audio_file=speaker_audio_file,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        sentence_split=sentence_split,
        use_config=use_config
    )

    print(result)
    if out_path:
        print(f"Output file: {out_path}")
    if speaker_audio_file:
        print(f"Speaker audio file: {speaker_audio_file}")
