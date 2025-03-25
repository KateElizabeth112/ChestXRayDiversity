import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

root_dir = '/Users/katephd/Documents/data/CheXpertSmall'
train_dir = os.path.join(root_dir, 'train')
valid_dir = os.path.join(root_dir, 'valid')

train_csv = os.path.join(root_dir, 'train.csv')
valid_csv = os.path.join(root_dir, 'valid.csv')


def createReducedTrainCSV(train_dir, train_csv):
    # list the files in the train directory
    patients = os.listdir(train_dir)
    patients = [p for p in patients if p != '.DS_Store']

    # open the train.csv file
    train = pd.read_csv(train_csv)

    # create an empty pandas dataframe to store information about patients with a view1 frontal image
    train_reduced = pd.DataFrame(columns=train.columns)

    # Add some extra columns to the train_reduced dataframe for x and y dimensions of the image
    train_reduced["img_shape_x"] = None
    train_reduced["img_shape_y"] = None

    # cycle over the patient's folders and check they have a study1 view1 frontal image
    for p in patients:
        patient_dir = os.path.join(train_dir, p)
        
        # check whether we have study1 view1 frontal
        study1_view1 = os.path.join(patient_dir, 'study1', 'view1_frontal.jpg')
        if not os.path.exists(study1_view1):
            print('Missing study1 view1 for patient {}'.format(p))
        else:
            # open the jpg image and save it as a numpy array
            img = Image.open(study1_view1)
            img_np = np.array(img)
            img_np = img_np / 255.0
            img_np = img_np.astype(np.float32)

            # save the numpy array as a .npy file
            np.save(os.path.join(root_dir, 'train_npy', f"img_{p[7:]}.npy"), img_np)

            # copy the information from train.csv to the train_reduced.csv  
            patient_info = train[train['Path'] == os.path.join("CheXpert-v1.0-small/train", p, 'study1', 'view1_frontal.jpg')]

            # add the x and y dimensions of the image
            patient_info["img_shape_x"] = img_np.shape[0]
            patient_info["img_shape_y"] = img_np.shape[1]

            # update the train_reduced dataframe
            train_reduced = train_reduced.append(patient_info)

    # save the train_reduced.csv
    train_reduced.to_csv(os.path.join(root_dir, 'train_reduced.csv'), index=False)
    print('Saved train_reduced.csv')


def plotSexDistribution():
    # open the train_reduced.csv 
    train_reduced = pd.read_csv(os.path.join(root_dir, 'train_reduced.csv'))

    # plot the distribution of patient sex from the train_reduced.csv as a bar chart
    sex = train_reduced["Sex"].values


def plotAgeDistribution():
    # open the train_reduced.csv 
    train_reduced = pd.read_csv(os.path.join(root_dir, 'train_reduced.csv'))

    # plot the distribution of patient age from the train_reduced.csv as a histogram
    age = train_reduced["Age"].values
    plt.hist(age, bins=20)
    plt.title('Age distribution')
    plt.xlabel('Age')
    plt.ylabel('Number of patients')
    plt.show()


def plotImageShapeDistribution():
    # make a sublot of the distribution of image shapes from the train_reduced.csv
    train_reduced = pd.read_csv(os.path.join(root_dir, 'train_reduced.csv'))

    # get the x and y dimensions of the images
    x = train_reduced["img_shape_x"].values
    y = train_reduced["img_shape_y"].values

    # plot the distribution of x and y dimensions as histograms
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(x, bins=20)
    axs[0].set_title('Image shape x distribution')
    axs[0].set_xlabel('Image shape x')
    axs[0].set_ylabel('Number of patients')
    axs[1].hist(y, bins=20)
    axs[1].set_title('Image shape y distribution')
    axs[1].set_xlabel('Image shape y')
    axs[1].set_ylabel('Number of patients')
    plt.show()



def main():
    #createReducedTrainCSV(train_dir, train_csv)
    #plotImageShapeDistribution()
    #plotSexDistribution()
    plotAgeDistribution()


if __name__ == '__main__':
    main()