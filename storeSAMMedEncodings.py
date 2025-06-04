# iterate across a dataset and save the SAMMed encodings
import argparse
import os
import torchvision.transforms as transforms
from cheXpertDataset import CheXpertDataset

# Set up the argument parser
parser = argparse.ArgumentParser(description="Precompute and store SAMMed Encodings for a dataset")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katephd/Documents")
parser.add_argument("-s", "--start_idx", type=int, help="Index of the image to start encoding", default=51803)
parser.add_argument("-e", "--encoder", type=str, choices=["SAMMedEncoder", "CxrFoundationEncoder"],
                    help="Encoder to use for encoding", default="CxrFoundationEncoder")

args = parser.parse_args()

root_dir = args.root_dir
start_idx = args.start_idx
encoder = args.encoder

data_dir = os.path.join(root_dir, 'data/CheXpertSmall')
dataset = CheXpertDataset(data_dir, split='train', transform=transforms.ToTensor())

if encoder == "CxrFoundationEncoder":
    from cxrFoundationEncoder import CxrFoundationEncoder
    encoder = CxrFoundationEncoder(dataset, "CheXpert")
    encoder.encode(start_idx)
    
elif encoder == "SAMMedEncoder":
    from samMedEncoder import SamMedEncoder
    # Check if the SAMMed checkpoint path exists
    if not os.path.exists("SAMMedCheckpoint/pretrain_model/sam-med2d_b.pth"):
        raise FileNotFoundError("SAMMed checkpoint file not found. Please download it first.")
    
    # Set the checkpoint path
    checkpoint_path = os.path.join("SAMMedCheckpoint/pretrain_model/sam-med2d_b.pth")

    encoder = SamMedEncoder(dataset, "CheXpert")
    encoder.encode(start_idx, checkpoint_path)
