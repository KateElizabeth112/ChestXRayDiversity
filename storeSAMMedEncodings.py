# iterate across a dataset and save the SAMMed encodings
import argparse
from samMedEncoder import SamMedEncoder
import os
import torchvision.transforms as transforms
from cheXpertDataset import CheXpertDataset

# Set up the argument parser
parser = argparse.ArgumentParser(description="Precompute and store SAMMed Encodings for a dataset")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katephd/Documents")
#parser.add_argument("-d", "--dataset_name", type=str, help="Name of dataset.", default="breastmnist")
#parser.add_argument("-i", "--image_size", type=int, help="Size of the images", default=28)
parser.add_argument("-s", "--start_idx", type=int, help="Index of the image to start encoding", default=0)

args = parser.parse_args()

root_dir = args.root_dir
start_idx = args.start_idx

data_dir = os.path.join(root_dir, 'data/CheXpertSmall')
dataset = CheXpertDataset(data_dir, split='train', transform=transforms.ToTensor())

checkpoint_path = os.path.join("SAMMedCheckpoint/pretrain_model/sam-med2d_b.pth")

encoder = SamMedEncoder(dataset, "CheXpert")
encoder.encode(start_idx, checkpoint_path)
