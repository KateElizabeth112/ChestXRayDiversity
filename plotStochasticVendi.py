# printig and plotting functions for stochastic Vendi experiments
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats

# set up path to results
results_dir = os.path.join("results", "random_sampling")

# define the plot colours
lblu = "#4878d0"
lred = "#d65f5f"

def processResults(dataset_sizes, sample_size):
     # Calculate the correlation between the stochastic Vendi scores and the true Vendi scores
    stochasticVSTimeMean = []
    fkeaVSTimeMean = []
    trueVSTimeMean = []

    stochasticVSTimeStd = []
    fkeaVSTimeStd = []
    trueVSTimeStd = []

    mseStochastic = []
    mseStochastic_corrected = []
    mseFKEA = []

    pearsonStochastic = []
    pearsonFKEA = []

    spearmanStochastic = []
    spearmanPStochastic = []
    spearmanFKEA = []
    spearmanPFKEA = []

    kendallStochastic = []
    kendallPStochastic = []
    kendallFKEA = []
    kendallPFKEA = []

    vsFKEA = []
    vsStochastic = []
    vsTrue = []


    n = sample_size

    for N in dataset_sizes:
        print(f"Processing results for N={N} and n={n}...")
        with open(os.path.join(results_dir, f"stochastic_vendi_{N}_{n}.pkl"), "rb") as f:
            resultsStochastic = pkl.load(f)
        
        # calculate the average time saved by using random sampling
        trueVSTimeMean.append(np.mean(resultsStochastic["time_full"]))
        stochasticVSTimeMean.append(np.mean(resultsStochastic["time_sub"]))

        trueVSTimeStd.append(np.std(resultsStochastic["time_full"]))
        stochasticVSTimeStd.append(np.std(resultsStochastic["time_sub"]))

        # get VS results
        VSTrue = resultsStochastic["vs_full"]
        VSStochastic = np.mean(resultsStochastic["vs_sub"], axis=1)

        # store the VS results for later plotting
        vsTrue.append(VSTrue)
        vsStochastic.append(VSStochastic)

        # get MSE between true and stochastic VS
        mseStochastic.append(np.mean((VSTrue - VSStochastic) ** 2))

        # load the FKEA results
        with open(os.path.join(results_dir, f"fkea_scores_{N}_{n}.pkl"), "rb") as f:
            resultsFKEA = pkl.load(f)
        
        fkeaTime = resultsFKEA["scoring_time"]
        fkeaVSTimeMean.append(np.mean(fkeaTime))
        fkeaVSTimeStd.append(np.std(fkeaTime))

        # get FKEA VS results
        VSFKEA = np.array(resultsFKEA["fkea_scores"])

        # store the FKEA VS results for later plotting
        vsFKEA.append(VSFKEA)

        # Calculate the pearson correlation between the estimates Vendi scores and the true Vendi scores
        pearsonStochastic.append(np.corrcoef(VSTrue, VSStochastic)[0, 1])
        pearsonFKEA.append(np.corrcoef(VSTrue, VSFKEA)[0, 1])

        # calculate the spearman correlation between the estimates Vendi scores and the true Vendi scores
        corr, p_val = scipy.stats.spearmanr(VSTrue, VSStochastic)
        spearmanStochastic.append(corr)
        spearmanPStochastic.append(p_val)
        
        corr, p_val = scipy.stats.spearmanr(VSTrue, VSFKEA)
        spearmanFKEA.append(corr)
        spearmanPFKEA.append(p_val)

        # calculate the kendall correlation between the estimates Vendi scores and the true Vendi scores
        corr, p_val = scipy.stats.kendalltau(VSTrue, VSStochastic)
        kendallStochastic.append(corr)
        kendallPStochastic.append(p_val)

        corr, p_val = scipy.stats.kendalltau(VSTrue, VSFKEA)
        kendallFKEA.append(corr)
        kendallPFKEA.append(p_val)

        # calculate the correction factor to transform the line of best fit to y=x for stochastic Vendi scores
        A = np.vstack([VSStochastic, np.ones(len(VSStochastic))]).T
        m, c = np.linalg.lstsq(A, VSTrue, rcond=None)[0]
        correction_factor_stochastic = m

        # apply the correction factor to the stochastic Vendi scores        
        VSStochastic_corrected = VSStochastic * correction_factor_stochastic

        # get the MSE between the corrected stochastic Vendi scores and the true Vendi scores
        mseStochastic_corrected.append(np.mean((VSTrue - VSStochastic_corrected) ** 2))

        if len(VSFKEA) != len(VSTrue):
            print(f"Warning: Length of FKEA scores ({len(VSFKEA)}) does not match length of true Vendi scores ({len(VSTrue)}). Skipping MSE calculation for FKEA.")
            mseFKEA.append(np.nan)
        else:
            # get MSE between true and FKEA VS
            mseFKEA.append(np.mean((VSTrue - VSFKEA) ** 2))

    # save the results to a dictionary and return it
    results = {
        "stochasticVSTimeMean": stochasticVSTimeMean,
        "fkeaVSTimeMean": fkeaVSTimeMean,
        "trueVSTimeMean": trueVSTimeMean,
        "stochasticVSTimeStd": stochasticVSTimeStd,
        "fkeaVSTimeStd": fkeaVSTimeStd,
        "trueVSTimeStd": trueVSTimeStd,
        "mseStochastic": mseStochastic,
        "mseStochastic_corrected": mseStochastic_corrected,
        "mseFKEA": mseFKEA,
        "pearsonStochastic": pearsonStochastic,
        "pearsonFKEA": pearsonFKEA,
        "spearmanStochastic": spearmanStochastic,
        "spearmanPStochastic": spearmanPStochastic,
        "spearmanFKEA": spearmanFKEA,
        "spearmanPFKEA": spearmanPFKEA,
        "kendallStochastic": kendallStochastic,
        "kendallPStochastic": kendallPStochastic,
        "kendallFKEA": kendallFKEA,
        "kendallPFKEA": kendallPFKEA,
        "vsTrue": vsTrue,
        "vsStochastic": vsStochastic,
        "vsFKEA": vsFKEA
    }

    return results


def plotCorrelations(results, dataset_sizes, diversity_measure="FKEA"):
    vsTrue = results["vsTrue"]

    if diversity_measure == "FKEA":
        vsEstimates = results["vsFKEA"]
        scatter_color = lred
    elif diversity_measure == "Stochastic Vendi":
        vsEstimates = results["vsStochastic"]
        scatter_color = lblu
    else:
        raise ValueError("Invalid diversity measure. Must be either 'FKEA' or 'Stochastic Vendi'.")

    # create a subplot for correlations with two columns and number of rows is half of dataset sizes (rounded up) using Seaborn
    # all plots sshould have the same z and y limits which are determined by the min and max of the true Vendi scores across all dataset sizes
    y_min = min([min(vs) for vs in vsEstimates])
    y_max = max([max(vs) for vs in vsEstimates])
    x_min = min([min(vs) for vs in vsTrue])
    x_max = max([max(vs) for vs in vsTrue])

    num_plots = len(dataset_sizes)
    num_cols = 3
    num_rows = (num_plots + 1) // num_cols
    plt.figure(figsize=(12, num_rows * 4))

    # set the color palette of seaborn to use the same colour for all the scatter plots
    for i, N in enumerate(dataset_sizes):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.scatterplot(x=vsTrue[i], y=vsEstimates[i], color=scatter_color)
        plt.xlabel("True Vendi Score", fontsize=14)
        plt.ylabel(f"{diversity_measure} Score", fontsize=14)
        plt.title(f"N={N}", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        #plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid()
    plt.tight_layout()
    plt.show()


def plotComputationTime(results, dataset_sizes, show=True):
    # plot the results for time taken using seaborn (number of samples N on x axis, time on y axis)
    # get the time results from the results dictionary

    stochasticVSTimeMean = np.array(results["stochasticVSTimeMean"])
    fkeaVSTimeMean = np.array(results["fkeaVSTimeMean"])
    trueVSTimeMean = np.array(results["trueVSTimeMean"])
    stochasticVSTimeStd = np.array(results["stochasticVSTimeStd"])
    fkeaVSTimeStd = np.array(results["fkeaVSTimeStd"])
    trueVSTimeStd = np.array(results["trueVSTimeStd"])

    # create an array for the predictions of true Vendi score for larger values of N by using the pattern for smaller values of N and extrapolating it to larger values of N
    # use a polynomial regression to extrapolate the time taken for larger values of N based on the time taken for smaller values of N
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly = PolynomialFeatures(degree=3)
    X = np.array(dataset_sizes[:-2]).reshape(-1, 1)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, trueVSTimeMean[:-2])
    X_pred = np.array([25000, 50000]).reshape(-1, 1)
    X_pred_poly = poly.transform(X_pred)
    trueVSTimePred = model.predict(X_pred_poly)

    # Append Nan values to the predictions at the beginning of the predicted Vendi times
    trueVSTimePred = np.insert(trueVSTimePred, 0, [np.nan, np.nan, np.nan, np.nan, trueVSTimeMean[-3]])

    # plot the results for time taken using seaborn (number of samples N on x axis, time on y axis)
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=dataset_sizes, y=stochasticVSTimeMean, label="Stochastic Vendi Score", marker='o')
    plt.fill_between(dataset_sizes, stochasticVSTimeMean - stochasticVSTimeStd, stochasticVSTimeMean + stochasticVSTimeStd, alpha=0.2)
    sns.lineplot(x=dataset_sizes, y=fkeaVSTimeMean, label="FKEA Score", marker='o', color='red')
    plt.fill_between(dataset_sizes, fkeaVSTimeMean - fkeaVSTimeStd, fkeaVSTimeMean + fkeaVSTimeStd, alpha=0.2, color='red')
    sns.lineplot(x=dataset_sizes, y=trueVSTimeMean, label="True Vendi Score", marker='o', color='black',)
    sns.lineplot(x=dataset_sizes, y=trueVSTimePred, label="True Vendi Score Prediction", marker='o', color='black', linestyle='--')
    plt.fill_between(dataset_sizes, trueVSTimeMean - trueVSTimeStd, trueVSTimeMean + trueVSTimeStd, alpha=0.2, color='black')
    plt.xlabel("Dataset Size (N)", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    # set teh y axis to log scale 
    plt.yscale("log")
    if show:
        plt.show()


def plotMSE(results, dataset_sizes, show=True):
    # plot the MSE between stochastic VS, corrected stochastic VS and FKEA VS against number of samples N
    mseStochastic = np.array(results["mseStochastic"])
    mseStochastic_corrected = np.array(results["mseStochastic_corrected"])
    mseFKEA = np.array(results["mseFKEA"])

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=dataset_sizes, y=mseStochastic, label="Stochastic Vendi Score", marker='o')
    sns.lineplot(x=dataset_sizes, y=mseFKEA, label="FKEA Benchmark", marker='o')
    sns.lineplot(x=dataset_sizes, y=mseStochastic_corrected, label="Corrected Stochastic Vendi Score", marker='o', color='green', linestyle='--')
    plt.xlabel("Dataset Size (N)", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.yscale("log")
    if show:
        plt.show()


def printCorrelationCoefficients(results, dataset_sizes):
    # print out the correlation coefficients for each dataset size N in a format suitable for LaTeX
    # use a bold font if the significant p-value is less than 0.05

    # get the correlation coefficients and p-values from the results dictionary
    spearmanStochastic = results["spearmanStochastic"]
    spearmanPStochastic = results["spearmanPStochastic"]
    spearmanFKEA = results["spearmanFKEA"]
    spearmanPFKEA = results["spearmanPFKEA"]
    kendallStochastic = results["kendallStochastic"]
    kendallPStochastic = results["kendallPStochastic"]
    kendallFKEA = results["kendallFKEA"]
    kendallPFKEA = results["kendallPFKEA"]

    for i, N in enumerate(dataset_sizes):
        if kendallPStochastic[i] < 0.05:
            print(r"& \bf{" + f"{kendallStochastic[i]:.2f}" + r"} ", end ="") 
        else:
            print(f"{kendallStochastic[i]:.2f}", end ="")
        
        if kendallPFKEA[i] < 0.05:
            print(r"& \bf{" + f"{kendallFKEA[i]:.2 f}" + r"} ", end="")
        else:
            print(f"& {kendallFKEA[i]:.2f}", end="")

    print(r"\\", end="\n")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    for i, N in enumerate(dataset_sizes):
        if spearmanPStochastic[i] < 0.05:
            print(r"& \bf{" + f"{spearmanStochastic[i]:.2f}" + r"} ", end ="") 
        else:
            print(f"& {spearmanStochastic[i]:.2f}", end ="")

        if spearmanPFKEA[i] < 0.05:
            print(r"& \bf{" + f"{spearmanFKEA[i]:.2f}" + r"} ", end="")
        else:
            print(f"& {spearmanFKEA[i]:.2f}", end ="")

    print(r"\\", end="\n")
    

def main():
    dataset_sizes = [1000, 2000, 5000, 10000, 15000, 25000, 50000]
    results = processResults(dataset_sizes, sample_size=100)

    printCorrelationCoefficients(results, dataset_sizes[:-2])
    plotMSE(results, dataset_sizes, show=True)
    plotComputationTime(results, dataset_sizes, show=True)
    plotCorrelations(results, dataset_sizes[:-2], diversity_measure="Stochastic Vendi")
    plotCorrelations(results, dataset_sizes[:-2], diversity_measure="FKEA")

if __name__ == "__main__":
    main()