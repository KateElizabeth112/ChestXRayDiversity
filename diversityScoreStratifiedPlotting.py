import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import argparse
import seaborn as sns

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the diversity scores for various stratifications from a ChestXRay dataset.")
parser.add_argument("-d", "--demographic", type=str, help="Demographic to stratify the dataset by", default="Sex")
parser.add_argument("-n", "--num_samples", type=list, nargs='+', help="Number of samples to use for diversity scoring", default=[50, 100, 200, 500, 1000])
parser.add_argument("-s", "--num_repeats", type=str, help="Number of repeats to use for diversity scoring", default="3")
parser.add_argument("-f", "--dataset_name", type=str, help="Name of the dataset to use for diversity scoring", default="CheXpert")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katephd/Documents")

args = parser.parse_args()

num_repeats = int(args.num_repeats)
num_samples = args.num_samples
demographic = args.demographic

if demographic == "Age":
    values = ["20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]
    results_csv = "runs_age.csv"
elif demographic == "Sex":
    values = ["Male", "Female"]
    results_csv = "runs_sex.csv"


results_path = os.path.join(args.root_dir, "code/ChestXRayDiversity/results", results_csv)


def plotResults(demographic, values, num_samples, encoder, results_file):       
    # Plot the results of the diversity scores by demographic value and number of samples.
    # set up a plot using seaborn
    sns.set_theme(style="whitegrid")
    plt.clf()


    # generate a color map
    colors = plt.cm.plasma(np.linspace(0, 1, len(values) + 1))

    # load the results csv for the given demographic
    df = pd.read_csv(results_file)

    # append the values list with mixed values
    values.append(str(values))

    for value, i in zip(values, range(len(values))):
        # create a container to store the average and std scores for a given demographic value across different number of samples
        av_scores = []
        std_scores = []

        # For each demographic value, filter the results csv by the following conditions
        condition1 = df["demographic"] == demographic
        condition2 = df["value"] == value

        # iterate over the number of samples and collect the diversity scores
        for ns in num_samples:
            condition3 = df["num_samples"] == ns
            scores = df[condition1 & condition2 & condition3][f"vs_{encoder}"].values

            # calculate the average and standard deviation of the scores and store them
            av_scores.append(np.nanmean(scores))
            std_scores.append(np.nanstd(scores))
            
        # convert to numpy arrays
        av_scores = np.array(av_scores)
        std_scores = np.array(std_scores)

        # plot the results
        if i == len(values) - 1:
            plot_label = "Mixed"
        else:
            plot_label = value

        plt.plot(num_samples, av_scores, color=colors[i], label=plot_label)
        plt.fill_between(num_samples, av_scores + std_scores, av_scores - std_scores, color=colors[i], alpha=0.2)

    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Vendi Score")
    plt.show()

def main():

    # plot the results
    plotResults(demographic, values, num_samples, "inception", results_path)


if __name__ == "__main__":
    main()
