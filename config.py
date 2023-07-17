# Training hyperparameters
NUM_CLASSES = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30

# Dataset
DATA_DIR = "data/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32