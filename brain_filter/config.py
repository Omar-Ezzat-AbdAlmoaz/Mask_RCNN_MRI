# brain_filter/config.py

import torch

# The path to a trained model
MODEL_PATH = "/path/to/best_autoencoder_brain_final.pth"  # ← Change da

# Final threshold 
THRESHOLD = 0.004

# The device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
