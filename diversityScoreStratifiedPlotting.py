import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import argparse
import seaborn as sns

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the diversity scores for various stratifications from a ChestXRay dataset.")
parser.add_argument("-d", "--demographic", type=str, help="Demographic to stratify the dataset by", default="AP/PA")
parser.add_argument("-n", "--num_samples", type=list, nargs='+', help="Number of samples to use for diversity scoring", default=[50, 100, 200, 500, 1000])
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located", default="/Users/katephd/Documents")

args = parser.parse_args()

num_samples = args.num_samples
demographic = args.demographic

if demographic == "Age":
    values = ['19-39', '40-59', '60-79', '80-100']
    results_csv = "runs_age.csv"
elif demographic == "Sex":
    values = ["Male", "Female"]
    results_csv = "runs_sex.csv"
elif demographic == "AP/PA":
    values = ["AP", "PA"]
    results_csv = "runs.csv"
elif demographic == "Disease":
    results_csv = "runs_disease.csv"
    values = ["1", "[1]"]


results_path = os.path.join(args.root_dir, "code/ChestXRayDiversity/results", results_csv)


def plotResults(demographic, values, num_samples, encoder, results_file, save_fig=False):       
    # Plot the results of the diversity scores by demographic value and number of samples.
    # set up a plot using seaborn
    sns.set_theme(style="whitegrid")
    plt.clf()

    # load the results csv for the given demographic
    df = pd.read_csv(results_file)

    # run an alternative loop to plot disease types
    if demographic == "Disease":
        disease_types = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                    'Edema', 'Atelectasis', 
                    'Pneumothorax', 'Pleural Effusion',  
                    'Support Devices']

        # generate a color map
        #colors = plt.cm.coolwarm(np.linspace(0, 1, len(disease_types) + 1))
        #colors = iter([plt.cm.tab20(i) for i in range(20)])
        
        # cycle over the disease types and plot the results for positive findings only
        for dt in disease_types:
            av_scores = []
            std_scores = []

            # first check if this demographic value exists in the results csv
            if dt in df["demographic"].values:
                print(f"Plotting results for disease type: {dt}")

                # For each disease type, filter the results csv by the following conditions
                condition1 = df["demographic"] == dt
                condition2 = df["value"] == values[0]

                # iterate over the number of samples and collect the diversity scores
                for ns in num_samples:
                    print(f"Processing number of samples: {ns}")
                    condition3 = df["num_samples"] == ns
                    scores = df[condition1 & condition2 & condition3][f"vs_{encoder}"].values

                    # detect if no scores are found for this disease type and number of samples
                    if len(scores) == 0:
                        print("No scores found for this disease type and number of samples")
                        av_scores.append(np.nan)
                        std_scores.append(np.nan)
                    else:
                        # calculate the average and standard deviation of the scores and store them
                        av_scores.append(np.nanmean(scores))
                        std_scores.append(np.nanstd(scores))
                    
                # convert to numpy arrays
                av_scores = np.array(av_scores)
                std_scores = np.array(std_scores)

                # plot the results
                plt.plot(num_samples, av_scores, color=plt.cm.tab10(disease_types.index(dt)), label=dt)
                plt.fill_between(num_samples, av_scores + std_scores, av_scores - std_scores, color=plt.cm.tab10(disease_types.index(dt)), alpha=0.2)

    else:
        # append the values list with mixed values
        values.append(str(values))

        # generate a color map
        colors = plt.cm.plasma(np.linspace(0, 1, len(values) + 1))

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

            plt.plot(num_samples, av_scores, color=plt.cm.tab10(i), label=plot_label)
            plt.fill_between(num_samples, av_scores + std_scores, av_scores - std_scores, color=plt.cm.tab10(i), alpha=0.2)

    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Vendi Score")
    # set y xis range
    plt.ylim(3.5, 6)
    if save_fig:
        plt.savefig(f"diversity_score_{demographic}.png")
    plt.show()

def main():

    # plot the results
    plotResults(demographic, values, num_samples, "inception", results_path)


if __name__ == "__main__":
    main()
