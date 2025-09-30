# script to prepare some plots to illustrate computational complexity of the Vendi score as well as the impact of dataset size and average similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import vendiScore
import time
import pickle as pkl


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


def plotOffDiagonalSimilarity():
    # function to generate a plot of the Vendi score as a function of off-diagonal similarity for different dataset sizes
    # define a range of dataset sizes and off-diagonal similarities
    n_samples = [10, 50, 100, 200, 500]
    sim_x = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    # prepare a container for the results
    vendi_score = np.zeros((len(n_samples), len(sim_x)))
    int_div = np.zeros((len(n_samples), len(sim_x)))

    for n in n_samples:
        for sim in sim_x:
            # generate a similarirty matrix where all off-diagonal entries have average similarity 'sim'
            sim_matrix = np.full((n, n), sim)
            np.fill_diagonal(sim_matrix, 1.0)  # set diagonal to 1.0 (self-similarity)

            # compute the Vendi score
            score = vendiScore.score_K(sim_matrix)
            vendi_score[n_samples.index(n), sim_x.index(sim)] = score
            print(f"Computed Vendi score for n={n}, sim={sim}: {score}")

            # compute Internal Diversity
            intdiv = vendiScore.intdiv_K(sim_matrix)
            int_div[n_samples.index(n), sim_x.index(sim)] = intdiv

    # plot the results
    # we will plot the Vendi score and Internal Diversity as a function of similarity for different dataset sizes using seaborn
    # there will be two separate plots (not subplots)
    plt.figure(figsize=(8, 4))
    for i, n in enumerate(n_samples):
        plt.plot(sim_x, vendi_score[i, :], label=f'n={n}')

    # set x and y labels with larger font size
    plt.xlabel('Off-Diagonal Similarity', fontsize=14)
    plt.ylabel('Vendi Score', fontsize=14)

    # set x and y label font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # set the x and y ticks to start at 0
    plt.xlim(0, 1)
    plt.ylim(0, 350)

    # set the legend to show with larger font size
    plt.legend(fontsize=12)
    plt.grid()  
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")
    plt.show()

    """
    plt.figure(figsize=(12, 6))
    for i, n in enumerate(n_samples):
        plt.plot(sim_x, int_div[i, :], label=f'n={n}')
    plt.xlabel('Similarity')
    plt.ylabel('Internal Diversity')
    plt.legend()
    plt.grid()  
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")
    plt.tight_layout()
    plt.show()
    """


def getComputationalComplexityVS(n_samples):
    # function to generate a plot of the computational complexity of the Vendi score as a function of dataset size up to 100000 samples

    # generate a symmetric similarity matrix with random values between 0 and 1
    times = []
    for n in n_samples:
        sim_matrix = np.random.rand(n, n)
        sim_matrix = (sim_matrix + sim_matrix.T) / 2  # make it symmetric
        np.fill_diagonal(sim_matrix, 1.0)  # set diagonal to 1.0 (self-similarity)
        import time
        start_time = time.time()
        score = vendiScore.score_K(sim_matrix)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"Computed Vendi score for n={n} in {elapsed_time:.4f} seconds")

    # save the results to a file so we can use them later
    import pickle as pkl
    with open("computational_complexity_vs.pkl", "wb") as f:
        pkl.dump((n_samples, times), f)


    """
    # plot the results
    plt.figure(figsize=(8, 4))
    plt.plot(n_samples, times, marker='o')
    plt.xlabel('Dataset Size (n)', fontsize=14)
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()  
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")
    plt.show()
    """

def getComputationalComplexitySim(n_samples, n_features=[10, 50, 100, 200, 500]):
    # Function to plot computational complexity for the cosine similarity kernel as a function of dataset size and number of features
    # create a container to store the results
    times = np.zeros((len(n_samples), len(n_features)))

    # create a random dataset of vectors with n_samples and n_features and compute cosine similarities to form similarity matrix
    for n in n_samples:
        for f in n_features:
            X = np.random.rand(n, f)
            
            # time the operation
            start_time = time.time()
            sim_matrix = cosineSimilarity(X, X)
            end_time = time.time()
            elapsed_time = end_time - start_time

            # store the results
            times[n_samples.index(n), n_features.index(f)] = elapsed_time

            # print out the result
            print(f"Computed similarity matrix for n={n} and f={f} in {elapsed_time:.4f} seconds")

    # save the results to a file so we can use them later
    with open("computational_complexity_sim.pkl", "wb") as f:
        pkl.dump((n_samples, n_features, times), f)

    """
    # plot the results using seaborn
    plt.figure(figsize=(8, 4))
    for i, f in enumerate(n_features):
        plt.plot(n_samples, times[:, i], label=f'f={f}', marker='o')
    plt.xlabel('Dataset Size (n)', fontsize=14)
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid()  
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")
    plt.show()
    """
    
def plotComputationalComplexity():
    # function to plot the computational complexity results from getComputationalComplexitySim and getComputationalComplexityVS as adjacent subplots
    # load the results from the files
    with open("computational_complexity_vs.pkl", "rb") as f:
        n_samples_vs, times_vs = pkl.load(f)
    with open("computational_complexity_sim.pkl", "rb") as f:
        n_samples_sim, n_features_sim, times_sim = pkl.load(f)

    # create the subplots using seaborn 
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, f in enumerate(n_features_sim):
        axs[0].plot(n_samples_sim, times_sim[:, i], label=f'f={f}', marker='o')
    axs[0].set_xlabel('Dataset Size (n)', fontsize=14)
    axs[0].set_ylabel('Computation Time (seconds)', fontsize=14)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend(fontsize=12)
    axs[0].grid()  
    axs[0].set_title('Cosine Similarity', fontsize=16)     
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")

    axs[1].plot(n_samples_vs, times_vs, marker='o')
    axs[1].set_xlabel('Dataset Size (n)', fontsize=14)
    axs[1].set_ylabel('Computation Time (seconds)', fontsize=14)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].grid()  
    axs[1].set_title('Vendi Score', fontsize=16)
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")

    # set the y axis of the two plots to be the same
    y_min = min(min(times_vs), np.min(times_sim))
    y_max = max(max(times_vs), np.max(times_sim))
    axs[0].set_ylim(y_min, y_max)
    axs[1].set_ylim(y_min, y_max)   
    plt.show()    

def main():
    n_samples = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]

    #plotOffDiagonalSimilarity()
    getComputationalComplexityVS(n_samples)
    getComputationalComplexitySim(n_samples, n_features=[10, 50, 100, 200, 500])
    #plotComputationalComplexity()

if __name__ == "__main__":
    main()