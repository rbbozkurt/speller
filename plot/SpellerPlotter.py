import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SpellerConstant


def plot_letter_sensors(target_letters: [], data_frame: pd.DataFrame(), labels: {}, data_details):
    for letter in target_letters:
        temp_df = data_frame.loc[data_frame.target == labels[letter]]
        for axis in SpellerConstant.SENSOR_AXES:
            fig, ax = plt.subplots()
            for row_ind in temp_df.index:
                temp_y_values = temp_df[axis][row_ind]
                temp_x_values = np.arange(temp_df['len'][row_ind])
                ax.plot(temp_x_values, temp_y_values)
            ax.set(title='{} - {} - {}'.format(letter, axis, data_details),
                   xlabel='Timesteps',
                   ylabel='Values')
            ax.set_ylim(-20, 20)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax.plot()
            fig.savefig('figures/{}_{}_{}_{}.pdf'.format("sensor_data", letter, axis, data_details.replace(' ', '')))
            plt.close(fig)


def plot_conf_matrix(cm, ylabel, xlabel, title):
    sns.heatmap(cm,
                annot=True)
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel(xlabel, fontsize=13)
    plt.title(title, fontsize=17)
    plt.savefig('figures/{}.pdf'.format(title))
    plt.close()
