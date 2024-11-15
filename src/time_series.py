"""
Collection of functions for assessing, testing and modeling time series.

"""

# Libraries importation

from typing import Literal, Tuple, List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



# Functions

def test_stationarity(df: pd.DataFrame, alpha: Literal[10, 5, 1] = 5) -> None:
    """
    Tests for stationarity by applying the Augmented Dickey-Fuller (ADF) test to each of the features
    in the input dataset, at the provided significance level (alpha = 10%, 5%, or 1%).

    Parameters:
    df (pandas.DataFrame): Dataset with the variables to be tested for stationarity.
    alpha (Literal[10, 5, 1]): Specified significance level for the ADF test:
        - 1: Significance level of 1%.
        - 5: Significance level of 5%. (default)
        - 10: Significance level of 10%.

    Returns:
    None
    """

    alpha = float(alpha) / 100

    for col in df.columns:
            
        stationarity_test = adfuller(df[col], autolag='AIC')
        print(f'{col}:')
        print(f"ADF statistic: {stationarity_test[0]:.03f}")
        print(f"P-value: {stationarity_test[1]:.03f}")

        if stationarity_test[1] <= alpha:
            print("The series is stationary.\n")
        else:
            print("The series is not stationary.\n")


def test_cointegration(time_series: pd.DataFrame, det_order: Literal[-1, 0, 1] = 1, k_ar_diff: int = 1) -> pd.DataFrame:
        """
        Performs the Johansen's Cointegration test and returns the results.

        Parameters:
        time_series (pandas.DataFrame): Time series data for the cointegration test
        det_order (Literal[-1, 0, 1]): The order of the deterministic terms:
            - -1: No constant or trend.
            - 0: Constant term only.
            - 1: Constant and trend terms. (default)
        k_ar_diff (int): The number of lags.

        Returns:        
        results_table (pandas.DataFrame): Results from the Johansen's Cointegration test.

        """

        coint_test_result = coint_johansen(endog=time_series, det_order=det_order, k_ar_diff=k_ar_diff)

        trace_stats = coint_test_result.trace_stat
        trace_stats_crit_vals = coint_test_result.trace_stat_crit_vals

        rank_list_int = list(range(0, time_series.shape[1]))
        rank_list_str = [str(x) for x in rank_list_int]
        rank_list_prefix = ['r = ']
        rank_list_prefix = rank_list_prefix + (['r <= ',] * (len(rank_list_str) - 1))

        rank_list = []

        for i in range(len(rank_list_int)):
                rank_list.append(rank_list_prefix[i] + rank_list_str[i])

        result_dict = {'Rank':rank_list, 
                        'CL 90%':trace_stats_crit_vals[:,0], 
                        'CL 95%':trace_stats_crit_vals[:,1], 
                        'CL 99%':trace_stats_crit_vals[:,2], 
                        'Trace Statistic': trace_stats}
                       
        results_table = pd.DataFrame(result_dict)  

        return results_table


def calculate_scores(predictions: pd.DataFrame, actuals: pd.DataFrame) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculates the RMSE, the MAE and the Coefficient of Determination (r-squared) for a given set of predictions according to the provided actuals.

        Parameters:
        predictions (pandas.DataFrame): Time series predictions for testing period.
        actuals (pandas.DataFrame): Time series actual values for testing period.

        Returns:
        rmse (List[float]): Root Mean Squared Error of the predictions.
        mae (List[float]): Mean Absolute Error of the predictions.
        coeff_det (List[float]): Coefficient of determination (r^2) of the predictions.
        """

        rmse = []
        mae = []
        coeff_det = []

        if len(actuals) == len(predictions):
                        
                for i in range(0, actuals.shape[1]):

                        print(f'{actuals.columns[i]}')

                        rmse_value = np.sqrt(mean_squared_error(actuals.iloc[:,i].values, predictions.iloc[:,i].values))
                        mae_value = mean_absolute_error(actuals.iloc[:,i].values, predictions.iloc[:,i].values)
                        coeff_det_value = r2_score(actuals.iloc[:,i].values, predictions.iloc[:,i].values)

                        print(f'RMSE: {rmse_value:.3f}')
                        print(f'MAE: {mae_value:.3f}')
                        print(f'Coefficient of Determination: {coeff_det_value:.3f}\n')

                        rmse.append(rmse_value)
                        mae.append(mae_value)
                        coeff_det.append(coeff_det_value)

                return rmse, mae, coeff_det
                

        else:

                print('Number of features is different between the testing set and the predictions set.')


