# A scratch file to test diversity scoring of the CheXpert dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from diversityScore import DiversityScore
from cheXpertDataset import CheXpertDataset
from torch.utils.data import Subset
import os
import numpy as np
import pandas as pd
import pickle as pkl
import mlflow
import argparse


def runExperiment(num_samples, num_repeats, demographic, values, dataset_name, root_dir):

    if dataset_name == "CheXpert":
        dataset = CheXpertDataset(os.path.join(root_dir, "CheXpertSmall"), split='train', transform=transforms.ToTensor())

        # open the train reduced csv file
        train_reduced_csv = os.path.join(root_dir, 'CheXpertSmall', 'train_reduced.csv')
        df = pd.read_csv(train_reduced_csv)
    else:
        raise ValueError("Dataset not supported")
    
    for value in values:
        for ns in num_samples:
            for i in range(num_repeats):
                print(f"scoring diversity for {value} {ns} samples, repeat {i}")

                # filter the dataframe  IDs to only include AP scans from females
                condition1 = df["AP/PA"] == "AP"

                # filter the dataframe to only include the demographic value
                if demographic == "Age":
                    lower, upper = map(int, value.split('-'))
                    condition2 = df[demographic] >= lower 
                    condition3 = df[demographic] < upper
                    image_ids = df[condition1 & condition2 & condition3]["image_id"].values
                else:
                    condition2 = df[demographic] == value
                    image_ids = df[condition1 & condition2]["image_id"].values

                # check if the number of samples is greater than the number of image IDs
                # if not, fill diversitys scores with Nan
                if ns < image_ids.shape[0]:
                    # create ns random indices between 0 and the dataset size
                    idx = np.random.choice(range(image_ids.shape[0]), ns, replace=False)
                    ids = image_ids[idx].astype(int)

                    # create a subset of the dataset for diversity scoring
                    ds = DiversityScore(Subset(dataset, ids), ids, {"dataset_name": dataset_name})

                    scores = ds.scoreDiversity()
                else:
                    scores = {"vs_inception": np.nan, "vs_sammed": np.nan}

                # store the results
                with mlflow.start_run():
                    print("Starting mlflow logging")
                    # Log the parameters
                    params = {  
                        "num_samples": ns,
                        "demographic": demographic,
                        "value": value,
                        "dataset_name": dataset_name,
                    }

                    mlflow.log_params(params)

                    # Log the diversity metrics for the training data
                    mlflow.log_metric("vs_inception", scores["vs_inception"])
                    mlflow.log_metric("vs_sammed", scores["vs_sammed"])
    

    # create containers to store the diversity scores which are initiated to NaN
    vendi_scores_inception = np.zeros((len(num_samples), num_repeats))
    vendi_scores_inception.fill(np.nan)
    vendi_scores_sammed = np.zeros((len(num_samples), num_repeats))
    vendi_scores_sammed.fill(np.nan)

    # create a flag that will be set when we can no longer run experiments
    SKIP_FLAG = False
    
    for ns in num_samples:
        if SKIP_FLAG:
            print("Skipping {} samples".format(ns))
            break

        for i in range(num_repeats):
            if SKIP_FLAG:
                print("Skipping {} repeats".format(i))
                break

            print("scoring mixed diversity for {} samples, repeat {}".format(ns, i))
            print("We will be using {} samples per demographic value".format(int(ns / len(values))))

            # create an empty list to store the IDs. We  will concatenate into a single array at the end
            ids_mixed = []

            # divide the number of samples by the number of demographic values
            ns_per_value = int(ns / len(values))

            # loop over the demographic values
            for value in values:

                if SKIP_FLAG:
                    print("Skipping {} value".format(value))
                    break

                # filter the dataframe  IDs to only include AP scans from females
                condition1 = df["AP/PA"] == "AP"

                if demographic == "Age":
                    lower, upper = map(int, value.split('-'))
                    condition2 = df[demographic] >= lower 
                    condition3 = df[demographic] < upper
                    image_ids = df[condition1 & condition2 & condition3]["image_id"].values
                else:
                    condition2 = df[demographic] == value
                    image_ids = df[condition1 & condition2]["image_id"].values

                # check if the number of samples is greater than the number of image IDs
                if ns_per_value > image_ids.shape[0]:
                    print("Not enough samples for demographic value {}. Skipping this repeat".format(value))
                    SKIP_FLAG = True
                    break

                # create ns_per_value random indices between 0 and the dataset size for that demographic value
                idx = np.random.choice(range(image_ids.shape[0]), int(ns_per_value), replace=False)
                ids = image_ids[idx].astype(int)

                # add the IDs to the list
                ids_mixed.append(ids)

            # convert the list to a numpy array
            ids_mixed = np.concatenate(ids_mixed, axis=0)

            # create a subset of the dataset for diversity scoring
            ds = DiversityScore(Subset(dataset, ids_mixed), ids_mixed, {"dataset_name": "CheXpert"})

            scores = ds.scoreDiversity()

            # store the results
            with mlflow.start_run():
                print("Starting mlflow logging")
                # Log the parameters
                params = {  
                    "num_samples": ns,
                    "demographic": demographic,
                    "value": values,
                    "dataset_name": dataset_name,
                }

                mlflow.log_params(params)

                # Log the diversity metrics for the training data
                mlflow.log_metric("vs_inception", scores["vs_inception"])
                mlflow.log_metric("vs_sammed", scores["vs_sammed"])


def plotResults(values, num_samples, encoder, results_dir):       
    # Now we can plot the re1sults
    plt.clf()

    # generate a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(values) + 1))

    # load the results
    for value, i in zip(values, range(len(values))):
        # load the results for each demographic value
        f = open(os.path.join(results_dir, f"results_{value}.pkl"), "rb")
        results = pkl.load(f)
        f.close()

        # calculate the average and standard deviation of the scores
        av_scores = np.mean(results[f"vendi_scores_{encoder}"], axis=1)
        std_scores = np.std(results[f"vendi_scores_{encoder}"], axis=1)

        # plot the results
        plt.plot(num_samples, av_scores, color=colors[i], label=value)
        plt.fill_between(num_samples, av_scores + std_scores, av_scores - std_scores, color=colors[i], alpha=0.2)


    f = open(os.path.join(results_dir, f"results_mixed.pkl"), "rb")
    results_mixed = pkl.load(f)
    f.close()
    
    # calculate the average and standard deviation of the scores
    av_scores_mixed = np.mean(results_mixed[f"vendi_scores_{encoder}"], axis=1)
    std_scores_mixed = np.std(results_mixed[f"vendi_scores_{encoder}"], axis=1)

    # plot the results
    plt.plot(num_samples, av_scores_mixed, color=colors[i+1], label="Mixed")
    plt.fill_between(num_samples, av_scores_mixed + std_scores_mixed, av_scores_mixed - std_scores_mixed, color=colors[i+1], alpha=0.2)

    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Diversity score")
    plt.title(f"{encoder} diversity score")
    plt.show()

def main():
    root_dir = '/Users/katephd/Documents/data/'

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("ChestXRayDiversity")

    num_repeats = 3
    #num_samples = [50, 100, 200, 500, 1000]
    num_samples = [50, 200, 1000]
    #demographic = "Sex"
    #values = ["Male", "Female"]
    #demographic = "Age"
    #values = ["20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]
    demographic = "Atelectasis"
    values = [1, 0]
    dataset_name = "CheXpert"

    # run the experiment
    runExperiment(num_samples, num_repeats, demographic, values, dataset_name, root_dir, save=True)

    # plot the results
    #plotResults(values, num_samples, "inception", ".")
    #plotResults(values, num_samples, "sammed", ".")


if __name__ == "__main__":
    main()
