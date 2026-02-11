# plotting functions for random sampling Vendi experiments
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import os


num_samples = [10, 50, 100, 200]
ds_size = [1000, 2000, 5000, 10000]

# loop over the results and calculate: 
# 1. the average and std of the samples scores for each run
# 2. the correlation between full dataset Vendi score and the mean of the subset scores
# 3. the correction factor to transform the line of best fit to y=x
# 4. the MSE between the corrected subset scores and the full dataset scores
# 5. the average time saved by using random sampling

# set up path to results
results_dir = os.path.join("results", "random_sampling")

def plotRandomSamplingResults():
    # create some storage variables for the results
    mse_results = np.zeros((len(ds_size), len(num_samples)))
    correlation_results = np.zeros((len(ds_size), len(num_samples)))
    correction_factors = np.zeros((len(ds_size), len(num_samples)))
    time_saved = np.zeros((len(ds_size), len(num_samples)))

    for n in num_samples:
        for N in ds_size:
            # load the results
            with open(os.path.join(results_dir, f"vendi_random_sampling_{N}_{n}.pkl"), "rb") as f:
                results = pkl.load(f)

            vs_full = results["vs_full"]
            vs_sub = results["vs_sub"]

            # calculate the average and std of the subset scores
            vs_sub_mean = np.mean(vs_sub, axis=1)
            vs_sub_std = np.std(vs_sub, axis=1)

            # calculate the correlation between full and subset scores
            correlation = np.corrcoef(vs_full, vs_sub_mean)[0, 1]
            print(f"N={N}, n={n}: Correlation between full and subset Vendi scores: {correlation:.4f}")

            # calculate the correction factor to transform the line of best fit to y=x
            A = np.vstack([vs_sub_mean, np.ones(len(vs_sub_mean))]).T
            m, c = np.linalg.lstsq(A, vs_full, rcond=None)[0]
            correction_factor = m
            print(f"N={N}, n={n}: Correction factor: {correction_factor:.4f}")

            # calculate the MSE between the corrected subset scores and the full dataset scores
            vs_sub_corrected = vs_sub_mean * correction_factor
            mse = np.mean((vs_full - vs_sub_corrected) ** 2)
            print(f"N={N}, n={n}: MSE between corrected subset scores and full dataset scores: {mse:.6f}")

            # calculate the average time saved by using random sampling
            time_full = results["time_full"]
            time_sub = results["time_sub"]
            avg_time_full = np.mean(time_full)
            avg_time_sub = np.mean(time_sub)
            time_saved_value = avg_time_full - avg_time_sub
            print(f"N={N}, n={n}: Average time saved by using random sampling: {time_saved_value:.4f} seconds")

            # store the results
            mse_results[ds_size.index(N), num_samples.index(n)] = mse
            correlation_results[ds_size.index(N), num_samples.index(n)] = correlation
            correction_factors[ds_size.index(N), num_samples.index(n)] = correction_factor
            time_saved[ds_size.index(N), num_samples.index(n)] = time_saved_value


    # plot the results using seaborn
    # first plot the correlation against number of samples for different dataset sizes
    plt.figure(figsize=(8, 6))
    for i, N in enumerate(ds_size):
        sns.lineplot(x=num_samples, y=correlation_results[i, :], label=f"N={N}", marker='o')
    plt.xlabel("Number of samples (n)", fontsize=14)
    plt.ylabel("Correlation coefficient", fontsize=14)
    plt.title("Correlation between full and subset Vendi scores", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()  

    # now plot the MSE against number of samples for different dataset sizes
    plt.figure(figsize=(8, 6))
    for i, N in enumerate(ds_size):
        sns.lineplot(x=num_samples, y=mse_results[i, :], label=f"N={N}", marker='o')
    plt.xlabel("Number of samples (n)", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.title("MSE between corrected subset and full Vendi scores", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale("log")
    plt.grid()
    plt.show()

    # finally plot the correction factors against number of samples for different dataset sizes
    plt.figure(figsize=(8, 6))
    for i, N in enumerate(ds_size):
        sns.lineplot(x=num_samples, y=correction_factors[i, :], label=f"N={N}", marker='o')
    plt.xlabel("Number of samples (n)", fontsize=14)
    plt.ylabel("Correction Factor", fontsize=14)
    plt.title("Correction factors for subset Vendi scores", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()

    # also plot the average time saved by using random sampling aganst number of samples for different dataset sizes
    plt.figure(figsize=(8, 6))
    for i, N in enumerate(ds_size):
        sns.lineplot(x=num_samples, y=time_saved[i, :], label=f"N={N}", marker='o')
    plt.xlabel("Number of samples (n)", fontsize=14)
    plt.ylabel("Average Time Saved (seconds)", fontsize=14) 
    plt.title("Average time saved by using random sampling", fontsize=16)
    # draw a black  solid line at y=0
    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()


def plotRandomSamplingResultsByN():
    # create some storage variables for the results
    time_n = np.zeros((len(num_samples), len(ds_size)))
    time_N = np.zeros((len(num_samples), len(ds_size)))

    for n in num_samples:
        for N in ds_size:
            # load the results
            with open(os.path.join(results_dir, f"vendi_random_sampling_{N}_{n}.pkl"), "rb") as f:
                results = pkl.load(f)

            # calculate the average time saved by using random sampling
            time_full = results["time_full"]
            time_sub = results["time_sub"]
            avg_time_full = np.mean(time_full)
            avg_time_sub = np.mean(time_sub)
            
            time_n[num_samples.index(n), ds_size.index(N)] = avg_time_sub
            time_N[num_samples.index(n), ds_size.index(N)] = avg_time_full

    # plot the results using seaborn
    # plot time taken for full dataset against number of samples for different dataset sizes
    # plot also the full dataset time
    plt.figure(figsize=(8, 6))
    for i, n in enumerate(num_samples):
        sns.lineplot(x=ds_size, y=time_n[i, :], label=f"n={n}", marker='o')
    sns.lineplot(x=ds_size, y=np.mean(time_N, axis=0), label=f"Full dataset (average)", marker='o', color='black', linestyle='--')
    plt.xlabel("Dataset size (N)", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()  


def plotScatterResults():
    # load the results
    with open(os.path.join(results_dir, "vendi_random_sampling_results.pkl"), "rb") as f:
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
    #plotRandomSamplingResults()
    #plotScatterResults()
    plotRandomSamplingResultsByN()

if __name__ == "__main__":
    main()