import os
from modules import paths


def preload(parser):
    parser.add_argument("--hf-token-out", type=str, help="HF Token for HF Out")