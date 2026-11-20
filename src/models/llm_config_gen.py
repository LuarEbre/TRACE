import datetime
import json
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
#LOCAL_BASE_DIR = "./trained_models/qwen2.5-3b-instruct"
#LOCAL_GGUF_PATH = "./trained_models/qwen2.5-3b-metadrive-q4_k_m.gguf"
#CONFIG_SAVE_DIR = "./last_configs"

# prompt for llm model 
SYSTEM_PROMPT = (
    "You are a MetaDrive Generator. Translate the user description into a valid JSON configuration."
    "Output ONLY raw JSON without markdown formatting or explanations. "
    "You will either use "
    "1) 'type': 'block_sequence' with 'config' as a string of block codes (S=Straight, C=Curve, O=Roundabout, T=T-Intersection, r=In-Ramp, R=Out-Ramp) when specific road elements are requested, OR "
    "2) 'type': 'block_num' with 'config' as an integer (e.g. 5) when a random or unspecified road layout is requested."
    "Default start_seed random between 1 and 1000 and num_scenarios to 1 unless specified otherwise."
)