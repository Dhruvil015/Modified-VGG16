import torch

# Basic config
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT = "..\\Deep Learning\\VGG16-IP\\checkpoints\\checkpoint.pth.tar" # change path aoccrodingly
MODEL = "..\\Deep Learning\\VGG16-IP\\checkpoints\\model.pth.tar" # change path aoccrodingly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
LEARNING_RATE = 0.005
NUM_EPOCHS = 20
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_CHANNELS = 3
