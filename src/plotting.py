"""
Collection of functions for plotting time series.

"""

import pandas as pd
import matplotlib.pyplot as plt
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

