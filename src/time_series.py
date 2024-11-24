"""
Collection of functions for assessing, testing and modeling time series.

"""

# Libraries importation

from typing import Literal, Tuple, List
from itertools import product
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from darts.models.forecasting.varima import VARIMA
from darts import TimeSeries


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


def test_granger_causality(df: pd.DataFrame, lag: int, alpha: float) -> pd.DataFrame:
        """
        Performs the Granger causality tests for each pair of stationary time series, in both directions y2 --> y1 and y1 --> y2.

        Parameters:
        df (pandas.DataFrame): Stationary time series data.
        lag (int): Lag for whose value the test is computed. 
        alpha (float): Desired significance level for the hypothesis test.

        Returns:
        summary (pandas.DataFrame): Summary results from the Granger causality tests.

        """
                
        # Creating empty lists
        y2 = []
        y1 = []
        interpretation = []
        ssr_ftest_pvalues = []
        ssr_chi2test_pvalues = []
        lrtest_pvalues = []
        params_ftest_pvalues = []
        gc = []

        # Iterating over series biderectionally
        arrays = list(combinations_with_replacement(df.columns, 2))

        for array in arrays:

                if array[1] == array[0]:
                        continue
                
                # y2 --> y1
                print(f'{array[1]} --> {array[0]}')
                result = grangercausalitytests(df[[array[0],array[1]]], maxlag=[lag])
                ssr_ftest_pvalue = result[lag][0]['ssr_ftest'][1]
                ssr_chi2test_pvalue = result[lag][0]['ssr_chi2test'][1]
                lrtest_pvalue = result[lag][0]['lrtest'][1]
                params_ftest_pvalue = result[lag][0]['params_ftest'][1]

                y2.append(array[1])
                y1.append(array[0])

                ssr_ftest_pvalues.append(round(ssr_ftest_pvalue,4))
                ssr_chi2test_pvalues.append(round(ssr_chi2test_pvalue, 4))
                lrtest_pvalues.append(round(lrtest_pvalue, 4))
                params_ftest_pvalues.append(round(params_ftest_pvalue, 4))

                # Test interpretation
                if ssr_ftest_pvalue <= alpha and ssr_chi2test_pvalue <= alpha and lrtest_pvalue <= alpha and params_ftest_pvalue <= alpha:
                        result = f'{array[1]} Granger-causes {array[0]}.'
                        print(f'\n{result}\n\n')
                        interpretation.append(result)
                        gc.append('True')
                
                elif ssr_ftest_pvalue > alpha and ssr_chi2test_pvalue > alpha and lrtest_pvalue > alpha and params_ftest_pvalue > alpha:
                        result = f'{array[1]} does not Granger-causes {array[0]}.'
                        print(f'\n{result}\n\n')
                        interpretation.append(result)
                        gc.append('False')

                else:
                        result = f'Inconsistent results among the tests.'
                        print(f'\n{result}\n\n')
                        interpretation.append(result)
                        gc.append('Uncertain')
                

                # y1 --> y2
                print(f'{array[0]} --> {array[1]}')
                result = grangercausalitytests(df[[array[1],array[0]]], maxlag=[lag])
                ssr_ftest_pvalue = result[lag][0]['ssr_ftest'][1]
                ssr_chi2test_pvalue = result[lag][0]['ssr_chi2test'][1]
                lrtest_pvalue = result[lag][0]['lrtest'][1]
                params_ftest_pvalue = result[lag][0]['params_ftest'][1]

                y2.append(array[0])
                y1.append(array[1])

                ssr_ftest_pvalues.append(round(ssr_ftest_pvalue, 4))
                ssr_chi2test_pvalues.append(round(ssr_chi2test_pvalue, 4))
                lrtest_pvalues.append(round(lrtest_pvalue, 4))
                params_ftest_pvalues.append(round(params_ftest_pvalue, 4))

                # Test interpretation
                if ssr_ftest_pvalue <= alpha and ssr_chi2test_pvalue <= alpha and lrtest_pvalue <= alpha and params_ftest_pvalue <= alpha:
                        result = f'{array[0]} Granger-causes {array[1]}.'
                        print(f'\n{result}\n\n')
                        interpretation.append(result)
                        gc.append('True')
                
                elif ssr_ftest_pvalue > alpha and ssr_chi2test_pvalue > alpha and lrtest_pvalue > alpha and params_ftest_pvalue > alpha:
                        result = f'{array[0]} does not Granger-causes {array[1]}.'
                        print(f'\n{result}\n\n')
                        interpretation.append(result)
                        gc.append('False')

                else:
                        result = f'Inconsistent results among the tests.'
                        print(f'\n{result}\n\n')
                        interpretation.append(result)
                        gc.append('Uncertain')

        # Building dataframe with summary results
        results_dict = {'y2':y2, 'y1':y1, 'Granger Causality':gc, 
                        'Test Interpretation':interpretation,
                        'SSR F-test p-values': ssr_ftest_pvalues,
                        'SSR Chi2-test p-values':ssr_chi2test_pvalues,
                        'LR-test p-values': lrtest_pvalues,
                        'Params F-test p-values': params_ftest_pvalues
                        }

        summary = pd.DataFrame(results_dict)

        return summary


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


def optimize_arma_model(p: range, q: range, time_series: pd.Series) -> ARIMA:
        """
        Optimize an autoregressive moving average (ARMA) model given a set of p and q values, by minimizing the Akaike Information Criterion (AIC).

        Parameters:
        p (range): Range for order p in the autoregressive portion of the ARMA model.
        q (range): Range for order q in the moving average portion of the ARMA model.
        time_series (pandas.Series): Time series data for fitting the ARMA model.

        Returns:
        arma_model (statsmodels.tsa.arima.model.arima): An ARIMA object fitted according to the combination of p and q that minimizes 
        the Akaike Information Criterion (AIC).        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                arma_model = ARIMA(time_series, order = (order[0], 0, order[1])).fit()
                order_results.append(order)
                aic_results.append(arma_model.aic)
        
        # Converting lists to dataframes
        results = pd.DataFrame({'(p,q)': order_results,
                                'AIC': aic_results                                
                                })        
        # Storing results from the best model
        lowest_aic = results.AIC.min()
        best_model = results.loc[results['AIC'] == lowest_aic, ['(p,q)']].values[0][0]

        # Printing results
        print(f'The best model is (p = {best_model[0]}, q = {best_model[1]}), with an AIC of {lowest_aic:.02f}.\n')         
        print(results)     

        # Fitting best model again
        arma_model = ARIMA(time_series, order = (best_model[0], 0, best_model[1])).fit()

        return arma_model


def optimize_arima_model(p: range, d: int, q: range, time_series: pd.Series) -> ARIMA:
        """
        Optimize an autoregressive integrated moving average (ARIMA) model based on the Akaike Information Criterion (AIC), 
        given a set of p and q values; while keeping the d order constant. 

        Parameters:
        p (range): Range for order p in the autoregressive portion of the ARIMA model.
        d (int): Integration order.
        q (range): Range for order q in the moving average portion of the ARIMA model.
        time_series (pandas.Series): Time series data for fitting the ARIMA model.

        Returns:
        arima_model (statsmodels.tsa.arima.model.ARIMA): An ARIMA object fitted according to the combination of p and q that minimizes 
        the Akaike Information Criterion (AIC).
        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                arima_model = ARIMA(time_series, order = (order[0], d, order[1])).fit()
                order_results.append((order[0], d, order[1]))
                aic_results.append(arima_model.aic)
        
        # Converting lists to dataframes
        results = pd.DataFrame({'(p,d,q)': order_results,
                                'AIC': aic_results                                
                                })        
        # Storing results from the best model
        lowest_aic = results.AIC.min()
        best_model = results.loc[results['AIC'] == lowest_aic, ['(p,d,q)']].values[0][0]

        # Printing results
        print(f'The best model is (p = {best_model[0]}, d = {d}, q = {best_model[2]}), with an AIC of {lowest_aic:.02f}.\n')         
        print(results)     

        # Fitting best model again
        arima_model = ARIMA(time_series, order = (best_model[0], best_model[1], best_model[2])).fit()

        return arima_model


def optimize_sarima_model(p: range, d: int, q: range, P: range, D: int, Q: range, m: int, time_series: pd.Series) -> SARIMAX:
        """
        Optimize a Seasonal Autoregressive Integrated Moving Average (SARIMA) model based on the Akaike Information Criterion (AIC), 
        given a set of p, q, P, and Q values; while keeping the d and D orders, and the frequency m constant.        

        Parameters:
        p (range): Range for order p in the autoregressive portion of the SARIMA model.
        d (int): Integration order.
        q (range): Range for order q in the moving average portion of the SARIMA model.
        P (range): Range for order P in the seasonal autoregressive portion of the SARIMA model.
        D (int): Seasonal integration order.
        Q (range): Range for order P in the seasonal moving average portion of the SARIMA model.
        m (int): Number of observations per seasonal cycle.
        time_series (pandas.Series): Time series data for fitting the SARIMA model.

        Returns:
        sarima_model (statsmodels.tsa.statespace.sarimax.SARIMAX): An SARIMAX object fitted according to the combination of p, q, P, 
        and Q values that minimizes the Akaike Information Criterion (AIC).
        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q, P, Q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                sarima_model = SARIMAX(endog=time_series, 
                                       order = (order[0], d, order[1]),
                                       seasonal_order=(order[2], D, order[3], m),
                                       ).fit(disp=False)
                order_results.append((order[0], d, order[1], order[2], D, order[3], m))
                aic_results.append(sarima_model.aic)
        
        # Converting lists to dataframes
        results = pd.DataFrame({'(p,d,q)(P,D,Q)m': order_results,
                                'AIC': aic_results                                
                                })        
        # Storing results from the best model
        lowest_aic = results.AIC.min()
        best_model = results.loc[results['AIC'] == lowest_aic, ['(p,d,q)(P,D,Q)m']].values[0][0]

        # Printing results
        print(f'The best model is (p = {best_model[0]}, d = {d}, q = {best_model[2]})(P = {best_model[3]}, D = {D}, Q = {best_model[5]})(m = {m}), with an AIC of {lowest_aic:.02f}.\n')         
        print(results)     

        # Fitting best model again
        sarima_model = SARIMAX(endog=time_series, 
                                order = (best_model[0], d, best_model[2]),
                                seasonal_order=(best_model[3], D, best_model[5], m),
                                ).fit(disp=False)

        return sarima_model


def optimize_sarimax_model(p: range, d: int, q: range, P: range, D: int, Q: range, m: int, endog: pd.Series, exog: pd.DataFrame) -> SARIMAX:
        """
        Optimize a Seasonal Autoregressive Integrated Moving Average with Exogeneous Variables (SARIMAX) model based on the Akaike Information Criterion (AIC), 
        given a set of p, q, P, and Q values; while keeping the d and D orders, and the frequency m constant.

        Parameters:
        p (range): Range for order p in the autoregressive portion of the SARIMA model.
        d (int): Integration order.
        q (range): Range for order q in the moving average portion of the SARIMA model.
        P (range): Range for order P in the seasonal autoregressive portion of the SARIMA model.
        D (int): Seasonal integration order.
        Q (range): Range for order P in the seasonal moving average portion of the SARIMA model.
        m (int): Number of observations per seasonal cycle.
        endog (pandas.Series): Time series of the endogenous variable for fitting the SARIMAX model.
        exog (pandas.DataFrame): Time series of the exogenous variables for fitting the SARIMAX model.

        Returns:
        sarimax_model (statsmodels.tsa.statespace.sarimax.SARIMAX): An SARIMAX model object fitted according to the combination of p, q, P, 
        and Q values that minimizes the Akaike Information Criterion (AIC).
        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q, P, Q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                sarimax_model = SARIMAX(endog=endog, 
                                        exog=exog,
                                       order = (order[0], d, order[1]),
                                       seasonal_order=(order[2], D, order[3], m),
                                       ).fit(disp=False)
                order_results.append((order[0], d, order[1], order[2], D, order[3], m))
                aic_results.append(sarimax_model.aic)
        
        # Converting lists to dataframes
        results = pd.DataFrame({'(p,d,q)(P,D,Q)m': order_results,
                                'AIC': aic_results                                
                                })        
        # Storing results from the best model
        lowest_aic = results.AIC.min()
        best_model = results.loc[results['AIC'] == lowest_aic, ['(p,d,q)(P,D,Q)m']].values[0][0]

        # Printing results
        print(f'The best model is (p = {best_model[0]}, d = {d}, q = {best_model[2]})(P = {best_model[3]}, D = {D}, Q = {best_model[5]})(m = {m}), with an AIC of {lowest_aic:.02f}.\n')         
        print(results)     

        # Fitting best model again
        sarimax_model = SARIMAX(endog=endog, 
                                exog=exog, 
                                order = (best_model[0], d, best_model[2]),
                                seasonal_order=(best_model[3], D, best_model[5], m),
                                ).fit(disp=False)

        return sarimax_model


def optimize_prophet_model(time_series: pd.DataFrame, changepoint_prior_scale: List[float], seasonality_prior_scale: List[float], metric: str) -> Prophet:
        """
        Optimize an univariate time series model with Prophet, given a set of changepoint_prior_scale and 
        seasonality_prior_scale values.

        Parameters:
        changepoint_prior_scale (List[float]): Values for the changepoint_prior_scale hyperparameter in Prophet.
        seasonality_prior_scale (List[float]): Values for the seasonality_prior_scale hyperparameter in Prophet.        
        series (pandas.DataFrame): Time series data in two columns: ds for dates in a date datatype, and y for the series.
        metric (str): Selected performance metric for optimization: One of 'mse', 'rmse', 'mae', 'mdape', or 'coverage'.
        
        Returns:
        m (prophet.Prophet): An Prophet model object optimized according to the combination of tested hyperparameters, by using the 
        indicated metric.
        
        """
        # Obtaining the combinations of hyperparameters
        params = list(product(changepoint_prior_scale, seasonality_prior_scale))

        # Creating emtpy lists to store performance results        
        metric_results = []

        # Defining cutoff dates
        start_cutoff_percentage = 0.5 # 50% of the data will be used for fitting the model
        start_cutoff_index = int(round(len(time_series) * start_cutoff_percentage, 0))
        start_cutoff = time_series.iloc[start_cutoff_index].values[0] 
        end_cutoff = time_series.iloc[-4].values[0] # The last fourth value is taken as the series is reported in a quarterly basis

        cutoffs = pd.date_range(start=start_cutoff, end=end_cutoff, freq='12M')

        # Fitting models
        for param in params:
                m = Prophet(changepoint_prior_scale=param[0], seasonality_prior_scale=param[1])
                m.add_country_holidays(country_name='MX')
                m.fit(time_series)
        
                df_cv = cross_validation(model=m, horizon='365 days', cutoffs=cutoffs)
                df_p = performance_metrics(df_cv, rolling_window=1)
                metric_results.append(df_p[metric].values[0])

        # Converting list to dataframe
        results = pd.DataFrame({'Hyperparameters': params,
                                'Metric': metric_results                                
                                })     
           
        # Storing results from the best model
        best_params = params[np.argmin(metric_results)]

        # Printing results
        print(f'\nThe best model hyperparameters are changepoint_prior_scale = {best_params[0]}, and seasonality_prior_scale = {best_params[1]}.\n')  
        print(results)

        # Fitting best model again
        m = Prophet(changepoint_prior_scale=best_params[0], seasonality_prior_scale=best_params[1])
        m.add_country_holidays(country_name='MX')
        m.fit(time_series);        

        return m


def optimize_var_model(p: List[int], time_series: pd.DataFrame) -> VAR:
        """
        Optimize a Vector Autoregression (VAR) model given a set of lags (p) by minimizing the lowest Akaike Information Criterion (AIC).

        Parameters:
        p (List[int]): Lag values.
        time_series (pandas.DataFrame): Time series data.

        Returns:
        var_model (statsmodels.tsa.vector_ar.var_model.VAR): VAR model object optimized according to the AIC criterion.

        """

        # Creating empty lists to store results

        aic_results = []

        for lag in p:
                var_model = VAR(endog=time_series).fit(maxlags=lag)
                aic_results.append(var_model.aic)
        
        # Converting lists to dataframes
        results = pd.DataFrame({'(p)': p,
                                'AIC': aic_results                                
                                })        
        # Storing results from the best model
        lowest_aic = results.AIC.min()
        best_model = results.loc[results['AIC'] == lowest_aic, ['(p)']].values[0][0]

        # Printing results
        print(f'The best model is (p = {best_model}), with an AIC of {lowest_aic:.02f}.\n')         
        print(results)     

        # Fitting best model again
        var_model = VAR(endog=time_series).fit(maxlags=best_model)

        return var_model


def optimize_varma_model(p: List[int], q: List[int], time_series: pd.DataFrame) -> VARMAX:
        """
        Optimize a Vector Autoregressive Moving Average (VARMA) model by minimizing the Akaike Information Criterion (AIC),
        based on the provided combinations of p and q.

        Parameters:
        p (List[int]): Orders for the autoregressive process of the model.
        q (List[int]): Orders for the moving average process of the model.
        series (pandas.DataFrame): Data for the time series for fitting the model.

        Returns:
        varma_model (statsmodels.tsa.statespace.varmax.VARMAX): A VARMAX object fitted according to the combination of p and q that minimizes 
        the Akaike Information Criterion (AIC).

        """
        
        # Obtaining the combinations of p and q
        order_list = list(product(p, q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                varma_model = VARMAX(endog=time_series, order=(order[0], order[1])).fit()
                order_results.append(order)
                aic_results.append(varma_model.aic)
        
        # Converting lists to dataframes
        results = pd.DataFrame({'(p,q)': order_results,
                                'AIC': aic_results                                
                                })        
        
        # Storing results from the best model
        lowest_aic = results.AIC.min()
        best_model = results.loc[results['AIC'] == lowest_aic, ['(p,q)']].values[0][0]

        # Printing results
        print(f'The best model is (p = {best_model[0]}, q = {best_model[1]}), with an AIC of {lowest_aic:.02f}.\n')         
        print(results)     

        # Fitting best model again
        varma_model = VARMAX(endog=time_series, order=(best_model[0], best_model[1])).fit()

        return varma_model


def fit_varima_model(p: int, d: int, q: int, time_series: pd.DataFrame) -> VARIMA:
        """
        Fits a VARIMA model given a set of p, d, and q values.

        Parameters:
        p (int): Order of the autoregressive process in the model.
        d (int): Integration order in the model.
        p (int): Order of the moving average process in the model.
        series (pandas.DataFrame): Data for the time series for fitting the model.

        Returns:
        varima_model (darts.models.forecasting.varima.VARIMA): A VARIMA object fitted according to the combination of p, d and q.


        """
        # Converting pandas.DataFrame to Darts.TimeSeries
        time_series = TimeSeries.from_dataframe(time_series)

        # Fitting VARIMA model
        varima_model = VARIMA(p=p, d=d, q=q, trend="n") # No trend for models with integration
        varima_model.fit(time_series)

        return varima_model