import copy

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace

import pickle as pkl
import os


class SamMedEncoder:
    def __init__(self, data, dataset_name):
        # check that the dataset parameter is an instance of Dataset
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        # check that the dataset_name parameter is a string
        assert isinstance(dataset_name, str), "dataset_name is not a string"

        self.data = data
        self.dataset_name = dataset_name
        
        # create a directory to store encodings
        if not os.path.exists("SAMMedEncodings"):
            os.mkdir("SAMMedEncodings")

        self.representations_dir = os.path.join("SAMMedEncodings", f"{self.dataset_name}")
        if not os.path.exists(self.representations_dir):
            os.mkdir(self.representations_dir)

        # set up a data loader. batch size must be 1 for SAMMed Encoder
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)

    def encode(self, start_idx, checkpoint_path):
        # check the input parameters
        # check that the start_idx parameter is an integer
        assert isinstance(start_idx, int), "start_idx is not an integer"

        # check that the checkpoint_path parameter is a string
        assert isinstance(checkpoint_path, str), "checkpoint_path is not a string"

        # check that the checkpoint_path parameter is a valid path
        assert os.path.exists(checkpoint_path), "checkpoint_path does not exist"

        # check that the checkpoint_path parameter ends with .pth
        assert checkpoint_path[-4:] == ".pth", "checkpoint_path does not end with .pth"

        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = self.checkpoint_path
        model = sam_model_registry["vit_b"](args).to(device)
        predictor = SammedPredictor(model)

        for k, (image, _) in enumerate(self.data_loader):
            if k >= self.start_idx:
                print(f"Encoding {self.dataset_name} image {k}")
                image_np = np.moveaxis(image.squeeze(dim=0).numpy(), 0, 2)
                predictor.set_image(image_np)
                embedding = predictor.get_image_embedding()

                # check the embedding size
                print("Embedding size: ", embedding.shape)

                f = open(os.path.join(self.representations_dir, "img_{}.pkl".format(k)), "wb")
                pkl.dump(embedding.flatten().unsqueeze(0), f)
                f.close()

    def retrieve(self, indices, encodings_dir):
        # check that we have a directory where the encodings are stored
        assert os.path.exists(encodings_dir), "Encodings directory does not exist"

        # retrieve pre-computed embeddings based on a list of indicies
        for p in range(indices.shape[0]):
            f = open(os.path.join(encodings_dir, "img_{}.pkl".format(indices[p])), "rb")
            embedding = pkl.load(f)
            f.close()

            if p == 0:
                vectors = copy.deepcopy(embedding)
            else:
                vectors = torch.cat((vectors, embedding), dim=0)

        return vectors.detach().numpy()
