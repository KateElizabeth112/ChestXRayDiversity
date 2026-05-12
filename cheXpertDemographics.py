# a script to calculate and plot demographics for the reduced  CheXpert dataset containing only study1 view1 frontal images
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# set up the directory structure
root_dir = '/Users/katephd/Documents/data/CheXpertSmall'
reduced_csv = os.path.join(root_dir, 'train_reduced.csv')
plot_dir = os.path.join(root_dir, 'dataset_demographics')

disease_cols = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
                'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 
                'Support Devices']

def printBasicStatistics(save_plots=False):
    # load the reduced csv file
    df = pd.read_csv(reduced_csv)

    # print basic statistics about the demographics
    print("Basic Statistics for CheXpert Reduced Dataset\n")

    # print the total number of patients/images
    total_images = df.shape[0]
    print(f"Total number of images: {total_images}")

    # print how many AP/PA images there are
    ap_counts = df['AP/PA'].value_counts()
    print("\nAP/PA Counts:")
    print(ap_counts)

    # print how many males and females there are
    sex_counts = df['Sex'].value_counts()
    print("\nSex Counts:")
    print(sex_counts)

    # print the number of positive findings for each disease type
    print("\nDisease Positive Findings Counts:")
    for col in disease_cols:
        positive_count = df[col].value_counts().get(1, 0)
        print(f"{col}: {positive_count}")

    # print the number of patients in each age group
    bins = [19, 40, 60, 80, 100]
    labels = ['19-39', '40-59', '60-79', '80+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    agegroup_counts = df['AgeGroup'].value_counts().sort_index()
    print("\nAge Group Counts:")
    print(agegroup_counts)


def ageDistributionPlot(save_plots=False):
    # load the reduced csv file
    df = pd.read_csv(reduced_csv)

    # remove the sex unknown entries
    df = df[df['Sex'].isin(['Male', 'Female'])] 

    # plot the counts in each age category split by sex
    bins = [19, 40, 60, 80, 100]
    labels = ['19-39', '40-59', '60-79', '80+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='AgeGroup', hue='Sex')
    plt.title('Age Distribution by Sex', size=16)
    plt.xlabel('Age Group', size=14)
    plt.ylabel('Count', size=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'agegroup_distribution_by_sex.png'))
    else:
        plt.show()
    plt.close()

    # plot the counts in each age category split by AP/PA
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='AgeGroup', hue='AP/PA')
    plt.title('Age Distribution by AP/PA', size=16)
    plt.xlabel('Age Group', size=14)
    plt.ylabel('Count', size=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'agegroup_distribution_by_ap_pa.png'))
    else:
        plt.show()
    plt.close() 

    # plot the counts in each age category split by disease types marked as a positive finding 1 in a single plot
    # the bars should be next to eachother for each disease type and not overlapping
    # use seaborn catplot
    df_melted = df.melt(id_vars=['AgeGroup'], value_vars=disease_cols, var_name='Disease', value_name='Presence')
    df_positive = df_melted[df_melted['Presence'] == 1]
    plt.clf()
    g = sns.catplot(data=df_positive, x='AgeGroup', hue='Disease', kind='count', height=6, aspect=1.5)
    # set the fontsize for the axes labels
    g.set_xticklabels(size=14)
    g.set_yticklabels(size=14)
    g.set_ylabels('Count', size=14)
    g.set_xlabels
    plt.title('Age Distribution by Disease Type', size=16)
    #plt.tight_layout()
    plt.set_cmap('tab10')
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'agegroup_distribution_by_disease.png'))
    else:
        plt.show()
    plt.close()

def diseaseDistributionPlot(save_plots=False):
    # plot a bar chart of the disease distribution in the reduced csv file
    # no findings, enlarged cardiomediastinum, cardiomegaly, lung opacity, lung lesion, edema, consolidation, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, fracture, support devices
    df = pd.read_csv(reduced_csv)

    # remove the sex unknown entries
    df = df[df['Sex'].isin(['Male', 'Female'])] 

    disease_counts = {}
    for col in disease_cols:
        disease_counts[col] = df[col].value_counts().get(1, 0)  # count only positive cases
    # plot the disease counts as a bar chart
    plt.clf()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(disease_counts.keys()), y=list(disease_counts.values()))
    plt.xticks(rotation=45, ha='right', size=14)
    plt.title('Disease Distribution', size=16)
    plt.xlabel('Disease', size=14)
    plt.ylabel('Count', size=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'disease_distribution.png'))
    else:
        plt.show()
    plt.close()

    # split the bars by sex
    plt.clf()
    plt.figure(figsize=(12, 6))
    disease_sex_counts = {}
    for col in disease_cols:
        disease_sex_counts[col] = df[df[col] == 1]['Sex'].value_counts()
    disease_sex_df = pd.DataFrame(disease_sex_counts).fillna(0)
    disease_sex_df = disease_sex_df.reset_index().melt(id_vars='index', var_name='Disease', value_name='Count')
    sns.barplot(data=disease_sex_df, x='Disease', y='Count', hue='index')
    plt.xticks(rotation=45, ha='right', size=14)
    plt.title('Disease Distribution by Sex', size=16)
    plt.xlabel('Disease', size=14)
    plt.ylabel('Count', size=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'disease_distribution_by_sex.png'))
    else:
        plt.show()
    plt.close()

    # split the bars by age group
    plt.clf()
    plt.figure(figsize=(12, 6))
    bins = [19, 40, 60, 80, 100]
    labels = ['19-39', '40-59', '60-79', '80+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    disease_agegroup_counts = {}
    for col in disease_cols:
        disease_agegroup_counts[col] = df[df[col] == 1]['AgeGroup'].value_counts().sort_index()
    disease_agegroup_df = pd.DataFrame(disease_agegroup_counts).fillna(0)
    disease_agegroup_df = disease_agegroup_df.reset_index().melt(id_vars='index', var_name='Disease', value_name='Count')
    sns.barplot(data=disease_agegroup_df, x='Disease', y='Count', hue='index')
    plt.xticks(rotation=45, ha='right', size=14)
    plt.title('Disease Distribution by Age Group', size=16)
    plt.xlabel('Disease', size=14)
    plt.ylabel('Count', size=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'disease_distribution_by_agegroup.png'))
    else:
        plt.show()
    plt.close()


def selectPleuralEffusion():
    # select for prescence and absence of pleural effusion and print the counts for each
    df = pd.read_csv(reduced_csv)
    pleural_effusion_counts = df['Pleural Effusion'].value_counts()
    print("Pleural Effusion Counts:")
    print(pleural_effusion_counts)

    # now look at the distribution of sex (in terms of counts and proportions) for the presence and absence of pleural effusion
    pleural_effusion_sex_counts = df.groupby('Pleural Effusion')['Sex'].value_counts()
    print("\nSex Distribution by Pleural Effusion Presence:")
    print(pleural_effusion_sex_counts)

    # look at the distribution of age (in terms of counts and proportions) for the presence and absence of pleural effusion
    bins = [19, 40, 60, 80, 100]
    labels = ['19-39', '40-59', '60-79', '80+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    pleural_effusion_agegroup_counts = df.groupby('Pleural Effusion')['AgeGroup'].value_counts()
    print("\nAge Group Distribution by Pleural Effusion Presence:")
    print(pleural_effusion_agegroup_counts)

    # look at the distribution of sex across the age groups for the presence and absence of pleural effusion
    pleural_effusion_agegroup_sex_counts = df.groupby(['Pleural Effusion', 'AgeGroup'])['Sex'].value_counts()
    print("\nSex Distribution by Age Group and Pleural Effusion Presence:")
    print(pleural_effusion_agegroup_sex_counts)
                    

def main():
    #diseaseDistributionPlot(save_plots=True)
    #printBasicStatistics()
    #ageDistributionPlot(save_plots=True)
    selectPleuralEffusion()


if __name__ == "__main__":
    main()
