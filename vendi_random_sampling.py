# script to see if we can speed up Vendi score calculation with random sampling
import numpy as np
from cheXpertDataset import CheXpertDataset
from inceptionEncoder import InceptionEncoder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import vendiScore
import pandas as pd
import pickle as pkl
import os
import time

# set up some arguments using argparse
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the diversity scores for various stratifications from a ChestXRay dataset.")
parser.add_argument("-dn", "--dataset_name", type=str, help="Name of the dataset to use for diversity scoring", default="CheXpert")
parser.add_argument("-rd", "--root_dir", type=str, help="Root directory where the data and code is located",
                    default="/Users/katephd/Documents")
parser.add_argument("-N", "--dataset_size", type=int, help="Number of samples to draw for full dataset", default=1000)
parser.add_argument("-n", "--num_samples", type=int, help="Number of samples to draw for random sampling", default=50)
parser.add_argument("-R", "--num_repeats", type=int, help="Number of repetitions for random sampling", default=100)
parser.add_argument("-Rr", "--num_runs", type=int, help="Number of random sampling runs", default=100)

args = parser.parse_args()

root_dir = args.root_dir
dataset_name = args.dataset_name
N = int(args.dataset_size)  # number of samples to draw for full dataset
n = int(args.num_samples) # number of samples to draw for random sampling
number_of_runs = int(args.num_runs) # number of random sampling runs
number_of_reps = int(args.num_repeats)  # number of repetitions for random sampling

# set up data and results paths
data_dir = os.path.join(root_dir, "data")
results_dir = os.path.join(root_dir, "code", "ChestXRayDiversity", "results", "random_sampling")

def cosineSimilarity(vectorsA, vectorsB):
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


def sampleAndComputeVendi(data_dir, number_of_runs, N, n, number_of_reps, dataset_name, save=True):
    # load the data and sample the full dataset
    dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='train', transform=transforms.ToTensor())

    # open the train reduced csv file
    train_reduced_csv = os.path.join(data_dir, 'CheXpertSmall', 'train_reduced.csv')
    df = pd.read_csv(train_reduced_csv)

    # get the image IDs from the dataframe so we can sample from them
    image_ids = df["image_id"].values

    # prepare containers to store the results
    vs_full = np.zeros((number_of_runs,))
    time_full = np.zeros((number_of_runs,))
    vs_sub = np.zeros((number_of_runs, number_of_reps))
    time_sub = np.zeros((number_of_runs,))
    sampled_ids = []

    for i in range(number_of_runs):
        print(f"Starting run {i} of {number_of_runs} for N={N} and n={n}")

        idx = np.random.choice(range(image_ids.shape[0]), N, replace=False)
        ids = image_ids[idx].astype(int)

        # Check IDs are sampled correctly
        print(f"Sampled {ids.shape[0]} IDs for run {i}.")

        # save the sampled IDs for this run so we can use them later when we calculate FKEA Vendi scores
        sampled_ids.append(ids)

        # initia=lize the encoder with the dataset and the name of the dataset (for loading the correct encodings)
        encoder = InceptionEncoder(dataset, "CheXpert")

        # if N <=10,000 calculate Vendi score for full dataset and time the operation 
        if N <= 18000:
            vectors = encoder.retrieve(ids, os.path.join("InceptionEncodings", f"{dataset_name}"))
            start_time = time.time()
            similarity_matrix = cosineSimilarity(vectors, vectors)

            score = vendiScore.score_K(similarity_matrix)
            end_time = time.time()

            # store the results
            vs_full[i] = score
            time_full[i] = end_time - start_time
        else:
            print(f"Skipping full Vendi score calculation for N={N} due to computational constraints.")
            vs_full[i] = np.nan
            time_full[i] = np.nan

        # now do random sampling of the ids for stochastic Vendi score calculation and time the operation
        start_time = time.time()
        for j in range(number_of_reps):
            idx_sub = np.random.choice(ids, n, replace=False)
            vectors_sub = encoder.retrieve(idx_sub, os.path.join("InceptionEncodings", f"{dataset_name}"))

            # calculate Vendi score for subset
            similarity_matrix_sub = cosineSimilarity(vectors_sub, vectors_sub)

            score_sub = vendiScore.score_K(similarity_matrix_sub)

            # store the result
            vs_sub[i, j] = score_sub
        
        end_time = time.time()
        time_sub[i] = end_time - start_time

        print(f"Completed run {i + 1} of {number_of_runs}")
        print(f"Full Vendi score: {vs_full[i]:.4f} (time: {time_full[i]:.4f} seconds)")
        print(f"Subset Vendi score (mean over {number_of_reps} reps): {np.mean(vs_sub[i]):.4f} (time: {time_sub[i]:.4f} seconds)")

        # save the results
        if save:
            with open(os.path.join(results_dir, f"stochastic_vendi_{N}_{n}.pkl"), "wb") as f:
                pkl.dump({"vs_full": vs_full, "vs_sub": vs_sub, "time_full": time_full, "time_sub": time_sub, "sampled_ids": sampled_ids}, f)


def main(): 

    #for n in [10, 50, 100, 200]:
    number_of_runs = 100
    n=100

    #for N in [1000, 2000, 5000, 10000]:
    #    sampleAndComputeVendi(data_dir, number_of_runs, N, n, number_of_reps, dataset_name, save=False)

    sampleAndComputeVendi(data_dir, number_of_runs, 50000, n, number_of_reps, dataset_name, save=True)



if __name__ == "__main__":
    main()
