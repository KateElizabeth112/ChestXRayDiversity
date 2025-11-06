import copy

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.models import inception_v3
import torch.nn as nn
from vendi_score import data_utils
from PIL import Image as im

import pickle as pkl
import os


def get_inception(pretrained=True, pool=True):
    model = inception_v3(pretrained=pretrained, transform_input=True).eval()

    if pool:
        model.fc = nn.Identity()
    return model


def get_embeddings(
        images,
        model=None,
        transform=None,
        batch_size=64,
        device=torch.device("cpu"),
        pretrained=True
):
    if type(device) == str:
        device = torch.device(device)
    if model is None:
        model = get_inception(pretrained=pretrained, pool=True).to(device)
        transform = inception_transforms()
    if transform is None:
        transform = transforms.ToTensor()
    embeddings = []
    for batch in data_utils.to_batches(images, batch_size):
        x = torch.stack([transform(img) for img in batch], 0).to(device)
        with torch.no_grad():
            output = model(x)
        if type(output) == list:
            output = output[0]
        embeddings.append(output.squeeze().cpu().numpy())
    return np.concatenate(embeddings, 0)


def inception_transforms():
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        ]
    )




class InceptionEncoder:
    def __init__(self, data, dataset_name):
        # check that the dataset parameter is an instance of Dataset
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        # check that the dataset_name parameter is a string
        assert isinstance(dataset_name, str), "dataset_name is not a string"

        self.data = data
        self.dataset_name = dataset_name
        
        # create a directory to store encodings
        if not os.path.exists("InceptionEncodings"):
            os.mkdir("InceptionEncodings")

        self.representations_dir = os.path.join("InceptionEncodings", f"{self.dataset_name}")
        if not os.path.exists(self.representations_dir):
            os.mkdir(self.representations_dir)

        # set up a data loader. batch size must be 1 for SAMMed Encoder
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)

    def encode(self, start_idx):
        # check the input parameters
        # check that the start_idx parameter is an integer
        assert isinstance(start_idx, int), "start_idx is not an integer"

        # set up the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_inception(pretrained=True, pool=True).to(device)
        transform = inception_transforms()

        for k, (image, _) in enumerate(self.data_loader):
            if k >= start_idx:
                print(f"Encoding {self.dataset_name} image {k}")

                with torch.no_grad():
                    embedding = model(transform(im.fromarray(image.squeeze().numpy())).unsqueeze(0))

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
