"""
Collection of functions for plotting time series.

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


def process_string(str_in: str) -> str:
    """
    Process an input string to conver it to lowercase, remove blank spaces and punctuation signs.

    Parameters:
    str_in (str): Raw string.    

    Returns:
    str_out (str): Processed string.
    """

    punctuation_dict = {'á': 'a',
                        'é': 'e',
                        'í': 'i',
                        'ó': 'o',
                        'ú': 'u',
                        '^': '',
                        '.': '',
                        ' ': '_'}

    str_out = (str_in.lower().map(punctuation_dict))

    return str_out


def plot_linechart(df: pd.DataFrame, is_saved: bool = True) -> None:
    """
    Plots a line chart for each feature in dataset.

    Parameters:
    df (pandas.DataFrame): Input dataset.
    is_saved (bool): Indicates whether the plot should be saved on local disk.

    Returns:
    None
    """
    
    columns = list(df.columns)

    for column in columns:

        plt.subplots(figsize=(7,5))
        
        sns.lineplot(data=df,
                    x=df.index,
                    y=df[column],             
                    legend=False
                    )
        
        xlabel=df.index.name.title()
        ylabel=df[column].name.title()

        y_ticks = plt.gca().get_yticks()
        plt.gca().set_yticklabels([f'{x:,.0f}' for x in y_ticks])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} by {xlabel}')            
        # plt.xticks([])

        plot_title = process_string(ylabel)

        if is_saved:
            plt.savefig(f'../reports/figures/fig_{plot_title}_over_time.png', bbox_inches = 'tight')


def plot_predictions(y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame, model_name: str) -> None:
        """
        Plots a line chart with the training, testing and predictions sets for a given model.

        Parameters:
        y_train (pandas.DataFrame): Training dataset
        y_test (pandas.DataFrame): Testing dataset
        y_pred (pandas.DataFrame): Predictions from the model
        model_name (str): Name of the model for the plot title.

        Returns:
        None
        """

        HISTORICAL_TIME_SERIES_COLOR = sns.color_palette('Blues_r')[0]
        PREDICTIONS_COLOR = '#C70039'

        chart_title = 'Predictions from ' +  model_name + ' vs. WALMEX Historical Net Sales'

        # Plots
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(y_train, ax=ax, zorder=2)
        sns.lineplot(y_test, ax=ax, color='green', linestyle='dashed', zorder=1)
        sns.lineplot(y_pred, ax=ax, color = PREDICTIONS_COLOR, lw=2, zorder=3)

        # Adding shadow to predictions
        first_pred_x = y_test.index[0]
        last_pred_x = y_test.index[-1]
        ax.axvspan(first_pred_x, last_pred_x, color='#808080', alpha=0.2)

        # Adding legend to plot
        legend_lines = [Line2D([0], [0], color=HISTORICAL_TIME_SERIES_COLOR, lw=2),
                        Line2D([0], [0], color='green', lw=2,  linestyle='dashed'),
                        Line2D([0], [0], color=PREDICTIONS_COLOR, lw=2)]
        plt.legend(legend_lines, 
                   ['Training', 'Testing', 'Predictions'], 
                   loc='upper left', 
                   facecolor='white',
                   frameon = True)

        # Adding labels
        plt.title(chart_title)
        plt.ylabel('Net Sales (mdp)')
        plt.xlabel('Time')

        # Adjusting Y ticks to Currency format
        ticks = ax.get_yticks()
        new_labels = [f'${int(i):,.0f}' for i in ticks]
        ax.set_yticklabels(new_labels)

        plot_title = process_string(chart_title)

        plt.savefig(f'../reports/figures/fig_{plot_title}.png',  bbox_inches = 'tight')
        plt.show()