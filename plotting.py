import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plotResultsByValue(demographic, values, num_samples, encoder, results_dir):       
    # Check that we can find the results directory
    results_dir = os.path.join(results_dir, "results")
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
    
    # Check that we can find the results csv file
    results_csv = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"Results csv file {results_csv} does not exist")
        return
    
    # Load the results csv file
    results = pd.read_csv(results_csv) 

    # Check that the encoder column exists
    if f"vs_{encoder}" not in results.columns:
        print(f"Encoder column vs_{encoder} does not exist in results csv file")
        return

    # Now we can plot the re1sults
    plt.clf()

    # generate a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(values) + 1))

    # process the results for each demographic value and number of samples
    for value, i in zip(values, range(len(values))):
        av_scores = []
        std_scores = []
        for ns in num_samples:
            condition1 = (results["demographic"] == demographic)
            condition2 = (results["value"] == str(value))
            condition3 = (results["num_samples"] == ns)

            # calculate the average and standard deviation of the scores
            av_scores.append(np.nanmean(results[condition1 & condition2 & condition3][f"vs_{encoder}"].values))
            std_scores.append(np.nanstd(results[condition1 & condition2 & condition3][f"vs_{encoder}"].values))

        # convert to numpy arrays
        av_scores = np.array(av_scores)
        std_scores = np.array(std_scores)

        # plot the results
        plt.plot(num_samples, av_scores, color=colors[i], label=value)
        plt.fill_between(num_samples, av_scores + std_scores, av_scores - std_scores, color=colors[i], alpha=0.2)

    
    # calculate the average and standard deviation of the scores
    #av_scores_mixed = np.mean(results_mixed[f"vendi_scores_{encoder}"], axis=1)
    #std_scores_mixed = np.std(results_mixed[f"vendi_scores_{encoder}"], axis=1)

    # plot the results
    #plt.plot(num_samples, av_scores_mixed, color=colors[i+1], label="Mixed")
    #plt.fill_between(num_samples, av_scores_mixed + std_scores_mixed, av_scores_mixed - std_scores_mixed, color=colors[i+1], alpha=0.2)

    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Diversity score")
    plt.title(f"{encoder} diversity score")
    plt.show()


def plotResultsByDemographic(demographics, value, num_samples, encoder, results_dir):
    # Check that we can find the results directory
    results_dir = os.path.join(results_dir, "results")
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
    
    # Check that we can find the results csv file
    results_csv = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"Results csv file {results_csv} does not exist")
        return
    
    # check that demographics is a list
    if not isinstance(demographics, list):
        print(f"Demographics should be a list, but got {type(demographics)}")
        return
    
    # check that num_samples is a list
    if not isinstance(num_samples, list):
        print(f"Num_samples should be a list, but got {type(num_samples)}")
        return
    
    # Load the results csv file
    results = pd.read_csv(results_csv) 

    # Check that the encoder column exists
    if f"vs_{encoder}" not in results.columns:
        print(f"Encoder column vs_{encoder} does not exist in results csv file")
        return

    # Now we can plot the re1sults
    plt.clf()

    # generate a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(demographics) + 1))

    # process the results for each demographic value and number of samples
    for demographic, i in zip(demographics, range(len(demographics))):
        av_scores = []
        std_scores = []
        for ns in num_samples:
            condition1 = (results["demographic"] == demographic)
            condition2 = (results["value"] == str(value))
            condition3 = (results["num_samples"] == ns)

            # calculate the average and standard deviation of the scores
            av_scores.append(np.nanmean(results[condition1 & condition2 & condition3][f"vs_{encoder}"].values))
            std_scores.append(np.nanstd(results[condition1 & condition2 & condition3][f"vs_{encoder}"].values))

        # convert to numpy arrays
        av_scores = np.array(av_scores)
        std_scores = np.array(std_scores)

        # plot the results
        plt.plot(num_samples, av_scores, color=colors[i], label=demographic)
        plt.fill_between(num_samples, av_scores + std_scores, av_scores - std_scores, color=colors[i], alpha=0.2)

    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Diversity score")
    plt.title(f"{encoder} diversity score")
    plt.show()




def main():
    root_dir = '/Users/katephd/Documents/code/ChestXRayDiversity'

    demographic = "Atelectasis"
    values = [1, 0, -1]
    value = 1
    num_samples = [50, 100, 200, 500, 1000]

    demographics = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema"]

    plotResultsByDemographic(demographics, value, num_samples, "inception", root_dir)

if __name__ == "__main__":
    main()
