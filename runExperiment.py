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


def runExperiment(num_samples, num_repeats, demographic, values, dataset_name, root_dir, save=False):

    if dataset_name == "CheXpert":
        dataset = CheXpertDataset(os.path.join(root_dir, "CheXpertSmall"), split='train', transform=transforms.ToTensor())

        # open the train reduced csv file
        train_reduced_csv = os.path.join(root_dir, 'CheXpertSmall', 'train_reduced.csv')
        df = pd.read_csv(train_reduced_csv)
    else:
        raise ValueError("Dataset not supported")
    
    for value in values:
        # create containers to store the diversity scores which are initiated to NaN
        vendi_scores_inception = np.zeros((len(num_samples), num_repeats))
        vendi_scores_inception.fill(np.nan)
        vendi_scores_sammed = np.zeros((len(num_samples), num_repeats))
        vendi_scores_sammed.fill(np.nan)

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
                vendi_scores_inception[num_samples.index(ns), i] = scores["vs_inception"]
                vendi_scores_sammed[num_samples.index(ns), i] = scores["vs_sammed"]

                if save:
                    # save the results as a pickle file
                    f = open(f"results_{value}.pkl", "wb")
                    pkl.dump({"vendi_scores_inception": vendi_scores_inception, "vendi_scores_sammed": vendi_scores_sammed}, f)
                    f.close()
    

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
            vendi_scores_inception[num_samples.index(ns), i] = scores["vs_inception"]
            vendi_scores_sammed[num_samples.index(ns), i] = scores["vs_sammed"]
            
            # save the results as a pickle file
            if save:
                f = open(f"results_mixed.pkl", "wb")
                pkl.dump({"vendi_scores_inception": vendi_scores_inception, "vendi_scores_sammed": vendi_scores_sammed}, f)
                f.close()

def plotResults(values, num_samples, encoder, results_dir):       
    # Now we can plot the re1sults
    plt.clf()

    # load the results
    for value in values:
        f = open(os.path.join(results_dir, f"results_{value}.pkl"), "rb")
        results = pkl.load(f)
        f.close()

        scores = np.mean(results[f"vendi_scores_{encoder}"], axis=1)
        plt.plot(num_samples, scores, label=value)


    f = open(os.path.join(results_dir, f"results_mixed.pkl"), "rb")
    results_mixed = pkl.load(f)
    f.close()

    scores_mixed = np.mean(results_mixed[f"vendi_scores_{encoder}"], axis=1)

    plt.plot(num_samples, scores, label="Mixed")

    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Diversity score")
    plt.title(f"{encoder} diversity score")
    plt.show()

def main():
    root_dir = '/Users/katephd/Documents/data/'

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
    plotResults(values, num_samples, "inception", ".")
    plotResults(values, num_samples, "sammed", ".")


if __name__ == "__main__":
    main()
