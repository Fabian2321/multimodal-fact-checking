from .data_loader import FakedditDataset
from .model_handler import (
    load_clip, 
    process_batch_for_clip,
    load_blip_conditional,
    process_batch_for_blip_conditional
)
# You can add other core components here as your project grows
# For example, if you have a main pipeline function you want to expose:
# from .pipeline import main as run_pipeline

print("Multimodal fact-checking 'src' package initialized.")
