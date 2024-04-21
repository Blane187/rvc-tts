import argparse
import asyncio
import datetime
import logging
import os
import time
import traceback

import librosa
import torch
from fairseq import checkpoint_utils

import edge_tts
from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Define default values
DEFAULT_SPEED = 0
DEFAULT_F0_KEY_UP = 0
DEFAULT_F0_METHOD = "rmvpe"
DEFAULT_INDEX_RATE = 1
DEFAULT_PROTECT = 0.33

# Set up argument parser
parser = argparse.ArgumentParser(description="RVC Text-to-Speech CLI")
parser.add_argument("tts_text", help="Input text for text-to-speech conversion")
parser.add_argument(
    "--model_name", default="your_default_model_name", help="Name of the model to use"
)
# Add other arguments here...

def tts_cli(args):
    config = Config()
    # Load hubert model, rmvpe model, etc.
    # Your code for loading models...

    # Call tts function with provided arguments
    result_info, edge_output_filename, _ = tts(
        args.model_name,
        args.speed,
        args.tts_text,
        args.tts_voice,
        args.f0_key_up,
        args.f0_method,
        args.index_rate,
        args.protect,
    )
    print(result_info)
    if edge_output_filename:
        print(f"Edge voice output saved as: {edge_output_filename}")

if __name__ == "__main__":
    args = parser.parse_args()
    tts_cli(args)
