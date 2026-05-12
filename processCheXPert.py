# proces the CheXpert dataset to create a reduced training CSV file and save frontal study 1 images as numpy arrays
# ignore lateral images and those from studies other than study 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image


def processImages(data_dir, split="train"):
    # process the CheXpert dataset to create a reduced training CSV file and save frontal study 1 images as numpy arrays

    # set up paths depending on the split
    if split == "train":
        image_dir = os.path.join(data_dir, 'train')
        csv_path = os.path.join(data_dir, 'train.csv')
        npy_dir = os.path.join(data_dir, 'train_npy')
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
    elif split == "valid":
        image_dir = os.path.join(data_dir, 'valid')
        csv_path = os.path.join(data_dir, 'valid.csv')
        npy_dir = os.path.join(data_dir, 'valid_npy')
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
    else:
        raise ValueError("Invalid split. Must be 'train' or 'valid'.")
    
    # list the files in the train directory
    patients = os.listdir(image_dir)
    patients = [p for p in patients if p != '.DS_Store']

    # order the patients list by patient ID
    patients.sort()

    # open the train.csv file
    df = pd.read_csv(csv_path)

    # create an empty pandas dataframe to store information about patients with a view1 frontal image
    df_reduced = pd.DataFrame(columns=df.columns)

    # Add some extra columns to the reduced dataframe for x and y dimensions of the image and the patient ID and image ID
    df_reduced["img_shape_x"] = None
    df_reduced["img_shape_y"] = None
    df_reduced["patient_id"] = None
    df_reduced["image_id"] = None

    # Start a counter which will increment whenever we save a new file
    image_counter = 0

    # cycle over the patient's folders and check they have a study1 view1 frontal image
    for patient in patients:
        patient_dir = os.path.join(image_dir, patient)
        
        # check whether we have study1 view1 frontal
        study1_view1 = os.path.join(patient_dir, 'study1', 'view1_frontal.jpg')

        if not os.path.exists(study1_view1):
            print('Missing study1 view1 for patient {}'.format(patient))
        else:
            print('Processing patient {}'.format(patient))

            # open the jpg image and save it as a numpy array
            img = Image.open(study1_view1)
            img_np = np.array(img)
            img_np = img_np / 255.0
            img_np = img_np.astype(np.float32)

            # save the numpy array as a .npy file
            np.save(os.path.join(npy_dir, f"img_{format(image_counter, '05d')}.npy"), img_np)

            
            patient_path = os.path.join("CheXpert-v1.0-small/" + split + "/", patient, 'study1', 'view1_frontal.jpg')

            # check that the patient information is available in the metadata csv file by checking patienbt path is in the 'Path' column of the metadata csv file
            if patient_path not in df['Path'].values:
                print('Patient path {} not found in metadata csv file'.format(patient_path))
                continue

            # copy the information to a new dataframe for the reduced dataset
            patient_info = df[df['Path'] == patient_path]

            # add the x and y dimensions of the image
            patient_info["img_shape_x"] = img_np.shape[0]
            patient_info["img_shape_y"] = img_np.shape[1]
            patient_info["patient_id"] = patient
            patient_info["image_id"] = format(image_counter, '05d')

            # update the df_reduced dataframe with the patient information on a new row
            # check the index of the last row in the df_reduced dataframe and add 1 to it to get the index for the new row
            if df_reduced.shape[0] == 0:
                new_index = 0
            else:                
                new_index = df_reduced.index[-1] + 1

            df_reduced.loc[new_index] = patient_info.values[0]

            # increment the image counter
            image_counter += 1

    # save the reduced metadata csv 
    df_reduced.to_csv(os.path.join(data_dir, f'{split}_reduced.csv'), index=False)

    print('Saved new metadata CSV file with {} entries'.format(df_reduced.shape[0]))


def plotSexDistribution(data_dir, split="train"):
    # open the train_reduced.csv 
    df = pd.read_csv(os.path.join(data_dir, f'{split}_reduced.csv'))

    # plot the distribution of patient sex from the metadata csv as a bar chart
    sex = df["Sex"].values


def plotAgeDistribution(data_dir, split="train"):
    # open the train_reduced.csv 
    df = pd.read_csv(os.path.join(data_dir, f'{split}_reduced.csv'))

    # plot the distribution of patient age from the metadata csv as a histogram
    age = df["Age"].values
    plt.hist(age, bins=20)
    plt.title('Age distribution')
    plt.xlabel('Age')
    plt.ylabel('Number of patients')
    plt.show()


def plotImageShapeDistribution(data_dir, split="train"):
    # make a sublot of the distribution of image shapes from the train_reduced.csv
    df = pd.read_csv(os.path.join(data_dir, f'{split}_reduced.csv'))

    # get the x and y dimensions of the images
    x = df["img_shape_x"].values
    y = df["img_shape_y"].values

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
    data_dir = '/Users/katephd/Documents/data/CheXpertSmall'

    processImages(data_dir, split="valid")
    #plotImageShapeDistribution()
    #plotSexDistribution()
    #plotAgeDistribution()


if __name__ == '__main__':
    main()