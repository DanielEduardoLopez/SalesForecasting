"""
Collection of functions for plotting time series.

"""

# Libraries importation

from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


# Functions

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


def plot_multivariate_predictions_in_single_chart(train: pd.DataFrame, test: pd.DataFrame, predictions: pd.DataFrame, chart_title: str, plot_size: Tuple[int, int] = (8,7)) -> None:
        """
        Plots the training, testing and predictions sets for multivariate time series in a single visual.

        Parameters:
        train (pandas.DataFrame): Training set for the multiple time series in their original scale.
        test (pandas.DataFrame): Testing set for the multiple time series in their original scale.
        predictions (pandas.DataFrame): Prediction set for the multiple time series in their original scale.
        chart_title (str): Title for saving the plot
        plot_size (Tuple[int, int]):  Tuple for the figure size. (8,7) by default.

        Returns:
        None
        """ 

        # Colors
        time_series_color=sns.color_palette('Blues_r')[0]
        test_color = 'green'
        pred_color = '#C70039'

        # Data
        cols = train.columns.values

        # Fig overall size
        plt.figure(figsize=plot_size)

        # Loop
        i = 1
        for col in cols:    
            # Plots
            plt.subplot(len(cols), 1, i)        
            ax = sns.lineplot(train[col], color=time_series_color, lw=1, zorder=2)
            sns.lineplot(test[col], color=test_color, lw=1, linestyle='dashed', zorder=1)
            sns.lineplot(predictions[col], color=pred_color, lw=2, zorder=3)

            # Adding shadow to predictions
            first_pred_x = predictions.index[0]
            last_pred_x = predictions.index[-1]
            ax.axvspan(first_pred_x, last_pred_x, color='#808080', alpha=0.2)    

            # Removing grid
            plt.grid(False)

            # Title
            plt.title(col, y=0.8, loc='left')    

            # Removing labels and ticks
            plt.ylabel('')
            plt.xlabel('')
            #plt.yticks([])
            #plt.xticks([])
            i += 1

        # Adding a legend
        legend_x = 1.12 
        legend_y = 1.15 * len(cols)
        legend_lines = [Line2D([0], [0], color=time_series_color, lw=2),
                        Line2D([0], [0], color=test_color, lw=2, linestyle='dashed'),
                        Line2D([0], [0], color=pred_color, lw=2)]
        plt.legend(legend_lines, 
                   ['Training', 'Testing', 'Predictions'], 
                   loc='upper center', 
                   bbox_to_anchor=(legend_x, legend_y),
                   facecolor='#ececec',
                   frameon = True)


        # Adjusting ticks in plot
        # max_xticks = 12
        # xloc = plt.MaxNLocator(max_xticks)
        # ax.xaxis.set_major_locator(xloc)
        # plt.xticks(rotation=45)

        # Adding a x label
        plt.xlabel('Date')
        
        plot_title = process_string("Predictions from " + chart_title + " vs. Historical Time Series")
            
        plt.savefig(f'../reports/figures/fig_{plot_title}.png',  bbox_inches = 'tight')        
        plt.show()


def plot_multivariate_predictions_in_multiple_charts(train: pd.DataFrame, test: pd.DataFrame, predictions: pd.DataFrame, plot_size: Tuple[int, int] = (8,5)) -> None:
        """
        Plots the time series along with its correspondent predictions in an individual chart each.
        Train, test and prediction parameters must have the same columns.      

        Parameters:        
        train (pandas.DataFrame): Pandas dataframe with the training portion of the time series.
        test (pandas.DataFrame): Pandas dataframe with the testing portion of the time series.
        predictions (pandas.DataFrame): Pandas dataframe with the predictions for the time series.
        plot_size (Tuple[int, int]): Tuple for the figure size. (8,5) by default.

        Returns:
        None

        """
        columns = list(train.columns)
        train_color=sns.color_palette('Blues_r')[0]
        test_color='green'
        pred_color = '#C70039'

        for col in columns:

            # Drawing plots  
            fig, ax = plt.subplots(figsize=plot_size)
            sns.lineplot(train[col], ax=ax, color=train_color, lw=1)
            sns.lineplot(test[col], ax=ax, color=test_color, lw=1, linestyle='dashed')
            sns.lineplot(predictions[col], ax=ax, color=pred_color, lw=2)

            # Adding shadow to predictions
            first_pred_x = test.index[0]
            last_pred_x = test.index[-1]
            ax.axvspan(first_pred_x, last_pred_x, color='#808080', alpha=0.2)    

            # Adding labels to plot
            label = str(col).title().replace('_',' ')
            chart_title = f'Predicted {label} Over Time'
            plt.title(chart_title)
            plt.ylabel(f'{label}')
            plt.xlabel('Time')      

            # Adding legend to plot
            legend_lines = [Line2D([0], [0], color=train_color, lw=2),
                            Line2D([0], [0], color=test_color, lw=2, linestyle='dashed'),                            
                            Line2D([0], [0], color=pred_color, lw=2)]
            plt.legend(legend_lines, ['Training', 'Testing', 'Predictions'], loc='upper left', facecolor='white', frameon=True)

            plot_title = process_string(chart_title)
            
            plt.savefig(f'../reports/figures/fig_{plot_title}.png',  bbox_inches = 'tight')
            plt.show()


def plot_predictions_from_regression_model(predictions: pd.DataFrame, time_series: pd.DataFrame, chart_title: str = 'Predictions from Regression Model') -> None:
        """
        Plots the predictions from a regression model compared to a historical time series.

        Parameters:
        predictions (pandas.DataFrame): Predictions from the regression model with a DateTime index.
        time_series (pandas.DataFrame): Historical time series with a DateTime index.
        chart_title (str): Title for the chart.

        Returns:
        None
        """

        # Colors
        time_series_color = sns.color_palette('Blues_r')[0]
        pred_color = '#C70039'

        # If predictions and time_series are DataFrames, select the first column
        if isinstance(predictions, pd.DataFrame):
                predictions = predictions.iloc[:, 0]
        if isinstance(time_series, pd.DataFrame):
                time_series = time_series.iloc[:, 0]

        # Plots
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x=time_series.index, y=time_series.values, ax=ax, color=time_series_color, zorder=1)
        sns.scatterplot(x=predictions.index, y=predictions.values, ax=ax, color = pred_color, zorder=2)

        # Labels
        plt.title(chart_title)
        plt.xlabel("Date")
        plt.ylabel("Net Sales (mdp)")

        # Adding legend to plot
        legend_lines = [Line2D([0], [0], color=time_series_color, lw=2),
                        Line2D([0], [0], color=pred_color, lw=2, linestyle='None', marker="o")]
        plt.legend(legend_lines, ['Time Series', 'Predictions'], loc='upper left', facecolor='white', frameon = True)

        # Adjusting Y ticks to Currency format
        ticks = ax.get_yticks()
        new_labels = [f'${int(i):,.0f}' for i in ticks]
        ax.set_yticklabels(new_labels)

        plot_title = process_string(chart_title)

        plt.savefig(f'../reports/figures/fig_{plot_title}.png',  bbox_inches = 'tight')        
        plt.show()


def plot_sales_forecast(time_series: pd.DataFrame, predictions: pd.DataFrame, chart_title: str) -> None:
        """
        Plots a time series and its forecast.

        Parameters:
        time_series (pandas.DataFrame): Historical time series data.
        predictions (pandas.DataFrame): Forecasts for the time series.
        chart_title (str): Title to be displayed on the chart.

        Returns:
        None
        """

         # Colors
        time_series_color = sns.color_palette('Blues_r')[0]
        pred_color = '#C70039'

        # If predictions and time_series are DataFrames, select the first column
        if isinstance(predictions, pd.DataFrame):
                predictions = predictions.iloc[:, 0]
        if isinstance(time_series, pd.DataFrame):
                time_series = time_series.iloc[:, 0]
        
        # Adjusting plots continuity
        last_value = time_series.iloc[-1]
        last_index = time_series.index[-1]
        last_observation = pd.Series(last_value, index=[last_index])
        predictions = pd.concat([predictions,last_observation]).sort_index()

        # Plots
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x=time_series.index, y=time_series.values, ax=ax, color=time_series_color, zorder=1)
        sns.lineplot(x=predictions.index, y=predictions.values, ax=ax, color = pred_color, zorder=2)

        # Adding shadow to predictions
        first_pred_x = predictions.index[0]
        last_pred_x = predictions.index[-1]
        ax.axvspan(first_pred_x, last_pred_x, color='#808080', alpha=0.2)

        # Labels
        plt.title(chart_title)
        plt.xlabel("Date")
        plt.ylabel("Net Sales (mdp)")

        # Adding legend to plot
        legend_lines = [Line2D([0], [0], color=time_series_color, lw=2),
                        Line2D([0], [0], color=pred_color, lw=2)]
        plt.legend(legend_lines, ['Historical', 'Forecast'], loc='upper left', facecolor='white', frameon=True)

        # Adjusting Y ticks to Currency format
        ticks = ax.get_yticks()
        new_labels = [f'${int(i):,.0f}' for i in ticks]
        ax.set_yticklabels(new_labels)

        plot_title = process_string(chart_title)

        plt.savefig(f'../reports/figures/fig_{plot_title}.png',  bbox_inches = 'tight')
        plt.show()