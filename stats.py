# get histogram of visage distrubution
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(input_path, age_min, age_max):

    dictionary_age = []

    for d in range(age_min, age_max + 1):
        file_list = os.listdir(os.path.join(input_path, str(d)))

        # count_len = len([x for x in file_list if not x.startswith('.')])

        for f in file_list:

            dictionary_age.append(
                {
                    'age': d,
                    'file': f
                }
            )

    print(dictionary_age)

    df = pd.DataFrame(dictionary_age)

    df.hist(column='age')
    plt.show()


def plot_histograms():

    dictionary_age_visage = []
    dictionary_age_instagram = []

    visage_input_path = os.path.expanduser('~/Documents/images/dataset/visage_base/')
    instagram_input_path = os.path.expanduser('~/Documents/images/ed/instagram_filtered/')

    for d in range(1, 19):
        file_list_visage = os.listdir(os.path.join(visage_input_path, str(d)))

        # count_len = len([x for x in file_list if not x.startswith('.')])

        for f in file_list_visage:
            dictionary_age_visage.append(
                {
                    'age': d,
                    'file': f
                }
            )

    df_visage = pd.DataFrame(dictionary_age_visage)

    for d in range(8, 19):
        file_list_instagram = os.listdir(os.path.join(instagram_input_path, str(d)))

        # count_len = len([x for x in file_list if not x.startswith('.')])

        for f in file_list_instagram:
            dictionary_age_instagram.append(
                {
                    'age': d,
                    'file': f
                }
            )

    df_instagram = pd.DataFrame(dictionary_age_instagram)

    df_merged = pd.concat([df_visage, df_instagram], ignore_index=True)

    # weights
    #df_weights = np.ones_like(df_visage['age']) / len(df_merged)
    #df2_weights = np.ones_like(df_instagram['age']) / len(df_merged)
    #df_merged_weights = np.ones_like(df_merged['age']) / len(df_merged)

    df_weights = None
    df2_weights = None
    df_merged_weights = None

    # plt_range = (df_merged.values.min(), df_merged.values.max())
    #bins = [x for x in range(1, 18)]

    bins = None
    val_min = min(df_merged['age'])
    val_max = max(df_merged['age'])
    plt_range = (val_min, val_max)
    plt_range_selfie = (min(df_instagram['age']), max(df_instagram['age']))

    fig, ax = plt.subplots()
    ax.hist(df_visage['age'], bins=bins, weights=df_weights, color='green', histtype='step',
            label='VisAGe', range=plt_range, hatch="--"
            )

    ax.hist(df_instagram['age'], bins=bins, weights=df2_weights, color='blue', histtype='step',
            label='Selfie-FV', range=plt_range_selfie, hatch="|"
            )

    ax.hist(df_merged['age'], bins=bins, weights=df_merged_weights, color='red',   histtype='step',
            label='Combined', range=plt_range, hatch="\\"
            )

    # plt.axhline(500, color='blue', linestyle='dashed', linewidth=2)
    # plt.legend(('1','2','3', '4'))

    ax.margins(0.05)
    ax.set_ylim(bottom=0)

    # ax.set_xlim(0, val_max + 1)
    plt.legend(loc='upper right')
    plt.title('Histogram')
    plt.xlabel('age')
    plt.ylabel('number of images')
    plt.xticks(np.arange(plt_range[0], plt_range[1] + 1, step=1))

    plt.show()


def plot_visage_gender_age_distribution():

    df = pd.read_excel('input/visage_gender_distro.xlsx', index_col=0)

    print(df)

    df.plot(kind="bar")
    plt.xlabel("Age")
    plt.ylabel("Occurrences")
    plt.show()


# plot visage gender and age distribution
plot_visage_gender_age_distribution()


# plot mixed dataset histogram

# visage_input_path = os.path.expanduser('~/Documents/images/dataset/visage_base/')
# instagram_input_path = os.path.expanduser('~/Documents/images/ed/instagram_filtered/')

# plot_histograms()

