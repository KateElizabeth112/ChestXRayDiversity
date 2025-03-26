# code to handle diversity scoring of a dataset of vectors
import numpy as np
import vendiScore
import torch
from torch.utils.data.dataset import Dataset
from copy import copy
from scipy.stats import entropy
from PIL import Image as im
from samMedEncoder import SamMedEncoder
import os


class DiversityScore:
    """
    Class for computing the diversity score for a dataset via a similarity matrix.
    """

    def __init__(self, data, indices, params):
        # check that the vectors parameter is a numpy array with two dimensions
        assert isinstance(params, dict), "params should be a dictionary"
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        self.params = params
        self.data = data
        self.indices = indices

        # check that we have a dataset_name parameter in the params dictionary
        assert "dataset_name" in params, "dataset_name is not in params dictionary"

        # check that the dataset_name parameter is a string
        assert isinstance(params["dataset_name"], str), "dataset_name is not a string"

        self.dataset_name = params["dataset_name"]

    def __len__(self):
        return self.vectors.shape[0]

    def cosineSimilarity(self, vectorsA, vectorsB):
        """
        Compute cosine similarity between multiple vectors. Sets a class attribute.

        Returns:
        numpy.ndarray: Cosine similarity matrix.
        """

        # Compute dot product of vectors
        dot_product = np.dot(vectorsA, vectorsB.T)

        # Compute norms of vectors
        normA = np.linalg.norm(vectorsA, axis=1, keepdims=True)
        normB = np.linalg.norm(vectorsB, axis=1, keepdims=True)

        # Compute cosine similarity matrix
        similarity_matrix = dot_product / (normA * normB.T)

        return similarity_matrix

    def vendiScore(self, embed="inception"):
        """
        Calculates the Vendi score directly from the cosine similarity matrix of pixel values.
        :return:
        float: The Vendi score for the dataset
        """
        if embed == "inception":
            data = [im.fromarray(self.data[i][0].squeeze().numpy()) for i in range(len(self.data))]
            vectors = vendiScore.getInceptionEmbeddings(data)
        elif embed == "sammed":
            encoder = SamMedEncoder(self.data, self.dataset_name)
            vectors = encoder.retrieve(self.indices, os.path.join("SAMMedEncodings", f"{self.dataset_name}"))

        similarity_matrix = self.cosineSimilarity(vectors, vectors)

        score = vendiScore.score_K(similarity_matrix)

        intdiv = vendiScore.intdiv_K(similarity_matrix)

        return score, intdiv

    def labelEntropy(self):
        """
        calculate the entropy of the labels in the dataset
        :return:
        """
        # get the dataset labels
        labels = np.array([self.data[i][1] for i in range(len(self.data))])

        if labels.shape[1] == 1:
            counts = np.zeros((2))
            counts[0] = np.sum(labels, axis=0)
            counts[1] = labels.shape[0] - np.sum(labels, axis=0)
        else:
            counts = np.sum(labels, axis=0)

        # calculate entropy from distribution over categorical labels
        label_entropy = entropy(counts)

        return label_entropy

    def scoreDiversity(self):
        """
        Runs all diversity scoring methods, returns a dictionary of results.
        :return:
        """
        # Store the results in a dictionary
        results = {}
        #results["label_entropy"] = self.labelEntropy()

        for embedding in ["inception", "sammed"]:
            vs, intdiv = self.vendiScore(embed=embedding)
            results["vs_{}".format(embedding)] = vs
            results["intdiv_{}".format(embedding)] = intdiv

        return results
