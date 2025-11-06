# this is a quick test script to see whether kmeans clustering is fast enough to be used before Vendi score calculation
import numpy as np
from sklearn.cluster import KMeans
from cheXpertDataset import CheXpertDataset
import torchvision.transforms as transforms
from inceptionEncoder import InceptionEncoder
from PIL import Image as im
import vendiScore
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt


# Steps:
# 1. Randomly select data
# 2. Get the data into an array of shape (n_samples, n_features)
# 3. Run KMeans clustering
# 4. Check the time taken
# 5. Print the cluster centers and labels

# set up some paths
root_dir =  "/Users/katephd/Documents/data"

def runClustering(dataset_name, number_of_samples, number_of_clusters, number_of_reps=10):

    # load the data
    dataset = CheXpertDataset(os.path.join(root_dir, "CheXpertSmall"), split='train', transform=transforms.ToTensor())

    # open the train reduced csv file
    train_reduced_csv = os.path.join(root_dir, 'CheXpertSmall', 'train_reduced.csv')
    df = pd.read_csv(train_reduced_csv)

    # filter the dataframe  IDs to only include AP scans 
    condition1 = df["AP/PA"] == "AP"
    image_ids = df[condition1]["image_id"].values

    # create an array to store the computation times
    times = np.zeros((number_of_reps, len(number_of_samples), len(number_of_clusters)))

    # sample the image IDs and get the embeddings
    for n_samples in number_of_samples:
        for n_clusters in number_of_clusters:
            print(f"Running KMeans with {n_samples} samples and {n_clusters} clusters")

            for rep in range(number_of_reps):
                print(f"Repetition {rep + 1} of {number_of_reps}")

                 # check if the number of samples is greater than the number of image IDs
                if n_samples < image_ids.shape[0]:
                    # create n_samples random indices between 0 and the dataset size
                    idx = np.random.choice(range(image_ids.shape[0]), n_samples, replace=False)
                    ids = image_ids[idx].astype(int)
                else: 
                    n_samples = image_ids.shape[0]
                    idx = np.random.choice(range(image_ids.shape[0]), n_samples, replace=False)
                    ids = image_ids[idx].astype(int)
                    print(f"n_samples is greater than the number of available AP scans, using {n_samples} samples instead")

                # get the embeddings for the sampled IDs
                encoder = InceptionEncoder(dataset, "CheXpert")
                embeddings = encoder.retrieve(ids, os.path.join("InceptionEncodings", f"{dataset_name}"))

                print(f"embeddings array shape: {embeddings.shape}")

                # Run K means on the embeddings
                # time the operation
                import time
                start_time = time.time()
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
                kmeans.fit(embeddings)
                end_time = time.time()
                print(f"KMeans clustering took {end_time - start_time} seconds")

                # store the time taken
                times[rep, number_of_samples.index(n_samples), number_of_clusters.index(n_clusters)] = end_time - start_time

                # save the results
                print("Saving the results...")
                with open("kmeans_times.pkl", "wb") as f:
                    pkl.dump((number_of_reps, number_of_samples, number_of_clusters, times), f)


def plotResults():
    # plot the computation times for K means as a function of number of samples and number of clusters
    # open the results file
    with open("kmeans_times.pkl", "rb") as f:
        number_of_reps, number_of_samples, number_of_clusters, times = pkl.load(f)

    # calculate mean and std across repetitions
    mean_times = np.mean(times, axis=0)
    std_times = np.std(times, axis=0)

    # plot the results using seaborn
    # computation time vs number of samples for different number of clusters
    # set up the plot
    plt.figure(figsize=(8, 6))
    
    for i, n_clusters in enumerate(number_of_clusters):
        sns.lineplot(x=number_of_samples[:-2], y=mean_times[:-2, i], label=f"{n_clusters} clusters", marker='o')
        plt.fill_between(number_of_samples[:-2],
                         mean_times[:-2, i] - std_times[:-2, i],
                         mean_times[:-2, i] + std_times[:-2, i],
                         alpha=0.2)
    plt.xlabel("Number of samples")
    plt.ylabel("Computation time (seconds)")
    # use a log scale for both axes
    plt.xscale("log")
    plt.yscale("log")
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")
    plt.grid()
    plt.show()


def main():
    # set up some variables
    dataset_name = "CheXpert"
    number_of_samples = [1000, 5000, 10000, 20000, 40000, 50000]
    number_of_clusters = [5, 10, 20, 50]
    number_of_reps = 10

    #runClustering(dataset_name, number_of_samples, number_of_clusters, number_of_reps)
    plotResults()


if __name__ == "__main__":
    main()

