"""
Collection of functions for assessing data quality, such as missing values and outliers.

"""

import pandas as pd
import numpy as np
from typing import Tuple, Literal


def get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the count and rate of missing values for each attribute in the dataset.

    Parameters:
    df (pandas.DataFrame): Input dataset.

    Returns
    missing_values_summary_table (pandas.DataFrame): Table with the count and rate of missing values for each attribute.

    """        
    missing_values_count = df.isnull().sum()

    missing_values_rate = (missing_values_count / len(df)) * 100

    missing_values_summary_table = pd.concat([pd.Series(missing_values_count.index), 
                                                pd.Series(missing_values_count.values), 
                                                pd.Series(missing_values_rate.values)], axis=1)
    
    missing_values_summary_table.columns = ['Attribute', 'Missing Values Count', 'Missing Values Rate (%)']

    missing_values_summary_table = missing_values_summary_table[missing_values_summary_table.iloc[:,1] != 0].sort_values(
                                        'Missing Values Rate (%)', ascending=False).round(2)
    
    print ("Dataset has " + str(df.shape[1]) + " attributes.\n"      
        "There are " + str(missing_values_summary_table.shape[0]) + " attributes that have missing values.")
    
    return missing_values_summary_table


def count_outliers(series: pd.Series, outlier_type: Literal['mild', 'extreme'] = 'mild') -> Tuple[int, float]:
    """ 
    Returns the number of suspected mild or extreme outliers and its rate based on the interquartile range (IQR) criterion.

    Parameters:
    series (pandas.Series): Vector of numerical data.
    outlier_type (Literal['mild', 'extreme']): Threshold applied to the outliers detection:
        - 'mild': Mild outliers, lower than Q1 - 1.5 * IQR, and greater than Q3 + 1.5 * IQR. (default)
        - 'extreme': Extreme outliers, lower than Q1 - 3 * IQR, and greater than Q3 + 3 * IQR.

    Returns:
    outliers_count (int): Number of suspected extreme outliers in the input data.
    outliers_percentage (float): Rate of suspected extreme outliers in the input data.

    """
    if outlier_type == 'mild':
        iqr_multiplier = 1.5
    elif outlier_type == 'extreme':
        iqr_multiplier = 3

    data_without_missing_values = series.dropna()

    q3, q1 = np.percentile(data_without_missing_values, [75, 25])

    iqr = q3 - q1

    lower_fence = q1 - iqr_multiplier * iqr
    upper_fence = q3 + iqr_multiplier * iqr

    mask = np.bitwise_and(data_without_missing_values > lower_fence, data_without_missing_values < upper_fence)

    data_count = data_without_missing_values.shape[0]

    outliers_count = data_count - data_without_missing_values.loc[mask].shape[0]

    outliers_percentage = round((outliers_count / data_count) * 100, 2)
    
    return outliers_count, outliers_percentage


def get_outliers_summary(df: pd.DataFrame, outlier_type: Literal['mild', 'extreme'] = 'mild') -> pd.DataFrame:
    """ 
    Calculates the count and rate of outliers for each numeric column in the dataset, based on the interquartile range (IQR).

    Parameters:
    df (pandas.DataFrame): Input dataset.
    outlier_type (Literal['mild', 'extreme']): Threshold applied to the outliers detection:
        - 'mild': Mild outliers, lower than Q1 - 1.5 * IQR, and greater than Q3 + 1.5 * IQR. (default)
        - 'extreme': Extreme outliers, lower than Q1 - 3 * IQR, and greater than Q3 + 3 * IQR.

    Returns:
    outliers_summary_table (pandas.DataFrame): Table with the number of suspected outliers and its corresponding rate for each column.

    """
  
    outliers_count_list = []

    outliers_percentage_list = []

    dataset_columns = list(df.select_dtypes(include='number').columns)

    for dataset_column in dataset_columns:

        outliers_count, outliers_percentage = count_outliers(df[dataset_column], outlier_type=outlier_type)

        outliers_count_list.append(outliers_count)
        outliers_percentage_list.append(outliers_percentage)
    
    outliers_dict = {'Attribute': dataset_columns, 
                    'Outliers Count':outliers_count_list, 
                    'Outliers Rate (%)': outliers_percentage_list}
    
    outliers_summary_table = pd.DataFrame(outliers_dict)    

    outliers_summary_table = (outliers_summary_table.loc[outliers_summary_table['Outliers Count']>0,:]
                            .sort_values(by='Outliers Count', ascending=False))

    print ("Dataset has " + str(df.shape[1]) + " columns.\n"      
    "There are " + str(outliers_summary_table.shape[0]) + " attributes that have suspected {outlier_type} outliers.")

    return outliers_summary_table