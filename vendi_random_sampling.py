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
parser.add_argument("-dd", "--data_dir", type=str, help="Root directory where the data is located",
                    default="/Users/katephd/Documents/data")
parser.add_argument("-N", "--dataset_size", type=int, help="Number of samples to draw for full dataset", default=1000)
parser.add_argument("-n", "--num_samples", type=int, help="Number of samples to draw for random sampling", default=50)
parser.add_argument("-R", "--num_repeats", type=int, help="Number of repetitions for random sampling", default=100)
parser.add_argument("-Rr", "--num_runs", type=int, help="Number of random sampling runs", default=100)

args = parser.parse_args()

data_dir = args.data_dir
dataset_name = args.dataset_name
N = int(args.dataset_size)  # number of samples to draw for full dataset
n = int(args.num_samples) # number of samples to draw for random sampling
number_of_runs = int(args.num_runs) # number of random sampling runs
number_of_reps = int(args.num_repeats)  # number of repetitions for random sampling


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


def sampleAndComputeVendi(root_dir, number_of_runs, N, n, number_of_reps, dataset_name):
    # load the data and sample the full dataset
    # load the data
    dataset = CheXpertDataset(os.path.join(root_dir, "CheXpertSmall"), split='train', transform=transforms.ToTensor())

    # open the train reduced csv file
    train_reduced_csv = os.path.join(root_dir, 'CheXpertSmall', 'train_reduced.csv')
    df = pd.read_csv(train_reduced_csv)

    # filter the dataframe  IDs to only include AP scans 
    condition1 = df["AP/PA"] == "AP"
    image_ids = df[condition1]["image_id"].values

    # prepare containers to store the results
    vs_full = np.zeros((number_of_runs,))
    time_full = np.zeros((number_of_runs,))
    vs_sub = np.zeros((number_of_runs, number_of_reps))
    time_sub = np.zeros((number_of_runs,))

    for i in range(number_of_runs):
        idx = np.random.choice(range(image_ids.shape[0]), N, replace=False)
        ids = image_ids[idx].astype(int)

        # get the embeddings for the sampled IDs
        encoder = InceptionEncoder(dataset, "CheXpert")
        start_time = time.time()
        vectors = encoder.retrieve(ids, os.path.join("InceptionEncodings", f"{dataset_name}"))

        # calculate Vendi score for full dataset and time the operation 
        similarity_matrix = cosineSimilarity(vectors, vectors)

        score = vendiScore.score_K(similarity_matrix)
        end_time = time.time()

        # store the results
        vs_full[i] = score
        time_full[i] = end_time - start_time

        # now do random sampling of the ids
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
        with open(f"vendi_random_sampling_{N}_{n}.pkl", "wb") as f:
            pkl.dump({"vs_full": vs_full, "vs_sub": vs_sub, "time_full": time_full, "time_sub": time_sub}, f)


def plotResults():
    # load the results
    with open("vendi_random_sampling_results.pkl", "rb") as f:
        results = pkl.load(f)

    vs_full = results["vs_full"]
    vs_sub = results["vs_sub"]

    # calculate the average and std of the subset scores
    vs_sub_mean = np.mean(vs_sub, axis=1)
    vs_sub_std = np.std(vs_sub, axis=1)

    # calculate the correlation between full and subset scores
    correlation = np.corrcoef(vs_full, vs_sub_mean)[0, 1]
    print(f"Correlation between full and subset Vendi scores: {correlation:.4f}")
    
    # plot the results as a scatter plot with error bars
    # left plot: full vs subset Vendi scores (without errror bars)
    # right plot: scatter plot of full Vendi scores vs variance of subset Vendi scores
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    #plt.errorbar(vs_full, vs_sub_mean, yerr=vs_sub_std, fmt='o', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.scatter(vs_full, vs_sub_mean)
    plt.xlabel("Full Vendi Score", fontsize=14)
    plt.ylabel("Subset Vendi Score", fontsize=14)
    plt.title("Full vs Subset Vendi Scores", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(vs_full, vs_sub_std)
    plt.xlabel("Full Vendi Score", fontsize=14)
    plt.ylabel("Subset Vendi Score Std Dev", fontsize=14)
    plt.title("Full Vendi Score vs Subset Score Variance", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()
    


def main(): 
    sampleAndComputeVendi(data_dir, number_of_runs, N, n, number_of_reps, dataset_name)
    #plotResults()



if __name__ == "__main__":
    main()
