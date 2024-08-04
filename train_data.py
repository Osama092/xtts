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

    return "Speech generated !", out_path, speaker_audio_file


def load_params_tts(out_path,version):
    
    out_path = Path(out_path)

    # base_model_path = Path.cwd() / "models" / version 

    # if not base_model_path.exists():
    #     return "Base model not found !","","",""

    ready_model_path = out_path / "ready" 

    vocab_path =  ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path =  ready_model_path / "speakers_xtts.pth"
    reference_path  = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
          return "Params for TTS not found", "", "", ""         

    return "Params for TTS loaded", model_path, config_path, vocab_path,speaker_path, reference_path
    
def train_model(custom_model,version,language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()

    run_dir = Path(output_path) / "run"

    # Remove train dir
    if run_dir.exists():
        os.remove(run_dir)
    
    # Check if the dataset language matches the language you specified 
    lang_file_path = Path(output_path) / "dataset" / "lang.txt"

    # Check if lang.txt already exists and contains a different language
    current_language = None
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
            if current_language != language:
                print("The language that was prepared for the dataset does not match the specified language. Change the language to the one specified in the dataset")
                language = current_language
            
    if not train_csv or not eval_csv:
        return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
    try:
        # convert seconds to waveform frames
        max_audio_length = int(max_audio_length * 22050)
        speaker_xtts_path,config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(custom_model,version,language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
    except:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

    # copy original files to avoid parameters changes issues
    # os.system(f"cp {config_path} {exp_path}")
    # os.system(f"cp {vocab_file} {exp_path}")
    
    ready_dir = Path(output_path) / "ready"

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")

    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
    # os.remove(ft_xtts_checkpoint)

    ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")

    # Reference
    # Move reference audio to output folder and rename it
    speaker_reference_path = Path(speaker_wav)
    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_reference_path, speaker_reference_new_path)

    print("Model training done!")
    # clear_gpu_cache()
    return "Model training done!"

def optimize_model(out_path, clear_train_data):
    # print(out_path)
    out_path = Path(out_path)  # Ensure that out_path is a Path object.

    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"

    # Clear specified training data directories.
    if clear_train_data in {"run", "all"} and run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {run_dir}: {e}")

    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        try:
            shutil.rmtree(dataset_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {dataset_dir}: {e}")

    # Get full path to model
    model_path = ready_dir / "unoptimize_model.pth"

    if not model_path.is_file():
        return "Unoptimized model not found in ready folder", ""

    # Load the checkpoint and remove unnecessary parts.
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Make sure out_path is a Path object or convert it to Path
    os.remove(model_path)

        # Save the optimized model.
    optimized_model_file_name="model.pth"
    optimized_model=ready_dir/optimized_model_file_name

    torch.save(checkpoint, optimized_model)
    ft_xtts_checkpoint=str(optimized_model)

    clear_gpu_cache()

    return "Model optimized and saved"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning and optimization demo"""
    )

    parser.add_argument(
        "--custom_model",
        type=str,
        help="Path to the custom model.pth file (optional). If not provided, the base model will be used.",
        default="",
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        help="Path to the train CSV file.",
        default=str(Path.cwd() / "finetune_models" / "dataset" / "metadata_train.csv"),

    )
    
    parser.add_argument(
        "--eval_csv",
        type=str,
        help="Path to the eval CSV file.",
        default=str(Path.cwd() / "finetune_models" / "dataset" / "metadata_eval.csv"),

    )
    
    # Read the language from a specific file

    # Get the directory where the current script is located
    base_dir = Path(__file__).parent

    # Construct the relative path to 'lang.txt'
    lang_file_path = base_dir / 'finetune_models' / 'dataset' / 'lang.txt'

    # Open and read the file using the relative path
    with open(lang_file_path, 'r', encoding='utf-8') as lang_file:
        language = lang_file.read().strip()

    parser.add_argument(
        "--version",
        type=str,
        help="XTTS base version. Default: 'v2.0.2'",
        default="v2.0.2",
    )
    
    # No need to specify these as they will take default values if not provided
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 6",
        default=6,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 2",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output model. This path is fixed and cannot be changed by the user.",
        default=str(Path.cwd() / "finetune_models"),
    )
    
    parser.add_argument(
        "--optimize",
        action='store_true',
        help="If set, optimize the model instead of training it."
    )
    
    args = parser.parse_args()

    if args.optimize:
        # Run optimization
        result = optimize_model(args.output_path, "run")
    else:
        # Check if train_csv and eval_csv are provided
        if not args.train_csv or not args.eval_csv:
            print("Error: --train_csv and --eval_csv must be provided for training.")
        else:
            # Run training
            result = train_model(
                args.custom_model,  # This is optional
                args.version,
                language,  # Use the language read from the file
                args.train_csv,
                args.eval_csv,
                args.num_epochs,
                args.batch_size,
                args.grad_acumm,
                args.output_path,  # This is now unchangeable
                args.max_audio_length,
            )

    print(result)

#without custom model and defualt setting
#python train_data.py --train_csv "path\to\train.csv" --eval_csv "path\to\eval.csv"
#python train_data.py --train_csv "C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\dataset\metadata_train.csv" --eval_csv "C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\dataset\metadata_eval.csv"


#with custom model
#python train_data.py --train_csv "path\to\train.csv" --eval_csv "path\to\eval.csv" [--custom_model "path\to\custom_model.pth"]


#With changing parameters(num_epochs, batch_size,...)
# python train_data.py --train_csv "path\to\train.csv" --eval_csv "path\to\eval.csv" --num_epochs 10 --batch_size 4

#optimize model
#python train_data.py --optimize





