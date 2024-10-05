<p align="center">
	<img src="Images/Header.png?raw=true" width=80% height=80%>
</p>

# Sales Forecasting for Walmart in Mexico
#### Daniel Eduardo López

<font size="-1"><a href="https://www.linkedin.com/in/daniel-eduardo-lopez">LinkedIn</a> | <a href="https://github.com/DanielEduardoLopez">GitHub </a></font>

**11 Sep 2024**

____
### **Contents**

1. [Introduction](#intro)<br>
2. [General Objective](#objective)<br>
3. [Research Question](#question)<br>
4. [Hypothesis](#hypothesis)<br>
5. [Methodology](#methodology)<br>
6. [Results](#results)<br>
	6.1 [Data Collection](#collection)<br>
	6.2 [Data Exploration](#exploration)<br>
	6.3 [Data Preparation](#preparation)<br>
  	6.4 [Exploratory Data Analysis](#eda)<br>
	6.5 [Data Modeling](#modeling)<br>
 		&emsp; &nbsp;6.5.1 [Moving Average (MA) Model](#ma_model)<br>
   		&emsp; &nbsp;6.5.2 [Autoregressive (AR) Model](#ar_model)<br>
     		&emsp; &nbsp;6.5.3 [Autoregressive (AR) Model with Additive Decomposition](#ar_model_add_decomp)<br>
       		&emsp; &nbsp;6.5.4 [Autoregressive Moving Average (ARMA) Model](#arma_model)<br>
	 	&emsp; &nbsp;6.5.5 [Autoregressive Integrated Moving Average (ARIMA) Model](#arima_model)<br>
   		&emsp; &nbsp;6.5.6 [Seasonal Autoregressive Integrated Moving Average (SARIMA) Model](#sarima_model)<br>
     		&emsp; &nbsp;6.5.7 [Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) Model](#sarimax_model)<br>
       		&emsp; &nbsp;6.5.8 [Simple Exponential Smoothing (SES) Model](#ses_model)<br>
	 	&emsp; &nbsp;6.5.9 [Holt-Winters (HW) Model](#hw_model)<br>
   		&emsp; &nbsp;6.5.10 [Prophet Univariate Time Series Model](#prophet_model)<br>
     		&emsp; &nbsp;6.5.11 [Vector Autoregressive (VAR) Model](#var_model)<br>
       		&emsp; &nbsp;6.5.12 [Vector Autoregressive Moving Average (VARMA) Model](#varma_model)<br>
	 	&emsp; &nbsp;6.5.13 [Vector Autoregressive Integrated Moving Average (VARIMA) Model](#varima_model)<br>
   		&emsp; &nbsp;6.5.14 [Random Forest (RF) Regression Model](#regression_model_rf)<br>
     		&emsp; &nbsp;6.5.15 [Support Vector Regression (SVR) Model](#regression_model_svm)<br>
	6.6 [Evaluation](#evaluation)<br>
7. [Deployment](#deployment)<br>
8. [Conclusions](#conclusions)<br>
9. [Bibliography](#bibliography)<br>
10. [Description of Files in Repository](#files)<br>

____
<a class="anchor" id="intro"></a>
### **1. Introduction**
Walmart of Mexico (or WALMEX) is one of the most important retail companies within the region, with 3,903 stores in Mexico and Central America, an equity of 199,086,037 MXN, and a yearly revenue of 880,121,761 MXN, according to the figures from December 2023. According to WALMEX last financial report, its goal is to double its sales in a period of 10 years [(Wal-Mart de México S.A.B. de C.V., 2024)](#walmex).

Time series are "a set of data points ordered in time" [(Peixeiro, 2022)](#peixeiro), which can be analyzed to calculate forecasts and get valuable insights [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

Univariate time series is the most used approach when analyzing time series [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni), by means of models such as Moving Average (MA), Autoregressive Moving Average (ARMA), Autoregressive Integrated Moving Average (ARIMA), or Simple Exponential Smoothing; which solely depend on the time and the variable under study.

On the other hand, it is also possible to forecast time series using regression-based modeling, in which other variables or features are used to predict the response variable [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni). This approach could have the advantage of quantifying the impact of the external economic indicators in the performance of an organization.

In the case of Mexico, it is possible to collect public data from different government offices such as INEGI or BANXICO, or from international sources such as the S&P500, and to assess how they correlate to revenue.

In this context, it is desirable to explore both approaches to predict WALMEX net sales over the next years. Thus, the purpose of the present project is to forecast WALMEX net sales and, then, use that information to predict whether WALMEX will be able to achieve its long-term goal of doubling its sales within the next ten years.

___
<a class="anchor" id="objective"></a>
## **2. General Objective**

To predict whether Walmart of Mexico will double its sales within the next ten years.

___
<a class="anchor" id="question"></a>
## **3. Research Question**

Will Walmart of Mexico be able to double its sales within the next ten years? If so, when?

___
<a class="anchor" id="hypothesis"></a>
## **4. Hypothesis**

Walmart de México will manage to double its sales within the next ten years.


___
<a class="anchor" id="methodology"></a>
## **5. Methodology**

The methodology of the present study is based on the CRISP-DM [(Chapman et al., 2000)](#chapman) framework and Rollin’s *Foundational Methodology for Data Science* [(Rollins, 2015)](#rollins):

1. **Analytical approach**: Building and evaluation of **univariate and multivariate time series** and **regression models**.
2. **Data requirements**: Data about WALMEX's net sales, WALMEX stock value, IPC index (performance of the largest and most liquid stocks listed on the Mexican Stock Exchange), S&P500 index, MXN/USD exchange rate, bonds interest rate (CETES28), money market representative interest rates (28 day TIIE), inflation, and gross domestic product (GDP) of Mexico.
3. **Data collection**: Data from a period of the last 10 years (from 01 Feb 2014 to 01 Feb 2024) was collected from <a href="https://finance.yahoo.com/">Yahoo Finance</a>, <a href="https://www.walmex.mx/en/financial-information/annual.html">Walmex's investors website</a>, <a href="https://www.inegi.org.mx/">INEGI</a>, and <a href="https://www.banxico.org.mx/">Banxico</a>.
4. **Data exploration**: Data was explored with Python 3 and its libraries Numpy, Pandas, Matplotlib and Seaborn.
5. **Data preparation**: Data was cleaned and prepared with Python 3 and its libraries Numpy and Pandas.
6. **Exploratory Data Analysis**: Statistical measures, distributions, time trends, relationships, and correlations were assessed using Python 3 and its libraries Pandas, Matplotlib, and Seaborn.
7. **Data modeling**: Ten univariate time series models were built and assessed in Python 3 and its libraries Statsmodels and Prophet to predict the net sales of WALMEX: 
    - **Moving Average (MA) model**, 
    - **Autoregressive (AR) model**, 
    - a series of **Autoregressive (AR) models** with **Additive Decomposition**, 
    - **Autoregressive Moving Average (ARMA) model**, 
    - **Autoregressive Integrated Moving Average (ARIMA) model**, 
    - **Seasonal Autoregressive Integrated Moving Average (SARIMA) model**, 
    - **Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) model**, 
    - **Simple Exponential Smoothing (SES) model**, 
    - **Holt-Winters (HW) model**, and
    - **Prophet Univariate Time Series Modeling**.

    Then, three vector models were created and trained in Python 3 and its libraries Statsmodels and Darts to predict the values of the selected macroeconomic indicators as a multivariate time series:
    - **Vector Autoregressive (VAR) model**, 
    - **Vector Autoregressive Moving Average (VARMA) model**, and 
    - **Vector Autoregressive Integrated Moving Average (VARIMA) model**  

    After that, two regression models were built using **Random Forests** and **Support Vector Machines** in Python 3 and its library Scikit-learn to predict WALMEX total sales based on the predictions for the best performing multivariate time series model. 
    
    All the models were fit using a training set with 80% of the data, and assessed using a testing set with the remaining 20% of the data. The scores **Root Mean Squared Error (RMSE)**, the **Mean Absolute Error (MAE)**, and **Coefficient of Determination** $(r^{2})$ were used for model assessment.

8. **Evaluation**: The different models were ranked in terms of the **Root Mean Squared Error (RMSE)**, the **Mean Absolute Error (MAE)**, and **Coefficient of Determination** $(r^{2})$. The best univariate, multivariate, and regression models were selected and retrained with all the historical data, and they were used to predict whether Walmart of Mexico would be able to double its sales within the next ten years.
9. **Deployment**: The best forecasting model was deployed using Streamlit and Plotly.

___
<a class="anchor" id="results"></a>
## **6. Results**

### **6.1 Data Collection** <a class="anchor" id="collection"></a>

Firstly, WALMEX sales figures were retrieved from the <a href="https://www.walmex.mx/en/financial-information/annual.html">Walmex's investors website</a>. As the financial data was disclosed in a quaterly basis in 40 PDF files hosted on a variety of inconsistent links, for sake of efficiency, it was decided to collect the data manually and consolidate it in an Excel file. 

The amount of files was sizeable for manual handling, and the complexity of developing a script for scraping and parsing each file was too high to just retrieve the account of *Net sales*. Thus, it was decided to proceed in a manual fashion. Its important to note that net sales figures for WALMEX are in millions of Mexican pesos (mdp, for its acronym in Spanish).

Then, the stock close values of WALMEX, and the index values from the IPC and the S&P500 were retrieved from the API of *Yahoo! Finance* through the library **yfinance**. The currency of the WALMEX stock value is Mexican pesos (MXN).

Later, the GDP and inflation data were retrieved from <a href='https://www.inegi.org.mx/servicios/api_indicadores.html'>INEGI's Query Constructor</a>, saving the response JSON files into disk, and then loading them into the notebook. It's important to note that GDP values are in millions of Mexican pesos at 2018 prices.

Finally, the MXN/USD exchange rates, the bonds interest rates (CETES 28), and the money market representative interest rates (28 day TIIE) data were retrieved from <a href="https://www.banxico.org.mx/">Banxico's website</a> in form of CSV files.



### **6.2 Data Exploration** <a class="anchor" id="exploration"></a>

The collected datasets were explored to describe its attributes, number of records, shapes, data types, their quality in terms of its percentage of missing values and suspected extreme outliers; as well as to perform an initial exploration of statistical measures, time trends, distributions and relationships among variables.

It was identified that several attributes in the datasets exhibited missing values, the vast majority in a small extent. So, the method **dropna()** was sufficient to handle those missing values. However, in the case of the attributes *OBS_NOTE* in the gdp dataset, or *Sq.mt. Mexico* and *Sq.mt. Central America* in the sales dataset, whose missing values rates (%) are higher than 30%, so they were removed during the preparation step.

Likewise, the datasets were assessed to identify any extreme outliers according to the interquartile range criterion ($quartile \pm IQR * 3$) [(NIST/SEMATECH, 2012)](#nist). It was found that the datasets were free from extreme outliers.

Furthermore, the data was initialy analyzed to calculate simple statistical measures, identify time trends, explore distributions and relationships for each data source; as well as exploring time patterns. 

It was found that the stock value of WALMEX and S&P500 had a strong positive trend; whereas the Mexican IPC had a weak one. Mexican GDP showed a positive trend over the last 10 years with a sharply temporary decrease during the Covid pandemic. The exchange rate showed a weak positive trend over the last ten years. Moreover, both bonds and money market interest rates showed an arbitrary trend, as those rates are set by the Central Bank of Mexico according to their contractive inflation policy.

The net sales of WALMEX, S&P500, and GDP showed distributions skewed to the right; whereas the stock value of the IPC exhibited a normal distribution. On the other hand, the exchange rates dataset showed two distributions: the first distribution was skewed to the right, and the second one resembled a normal distribution. No distribution was noticeable in the bonds rates and money market interest rates datasets.

Finally, the stock value of WALMEX and the S&P500 exhibited a strong positive correlation; whereas a weak positive correlation was observed between the stock value of WALMEX and the Mexican IPC. A very weak positive correlation was seen between the S&P500 and the IPC. A strong positive correlation was also observed for all the variables net sales, units, and commercial area in Mexico and Central America. And, a positive correlation was found between net sales and GDP. Moreover, as expectable, a strong positive correlation was found for the bonds rates and the money market interest rates, and no relationship was observed among the exchange rates and the bonds or interest rates.

### **6.3 Data Preparation** <a class="anchor" id="preparation"></a>

After the data was explored, it was wrangled to build the appropriate dataset for modeling based on the purposes of the present study, and the quality of the data. 

In this context, the following transformations were performed:

* Select appropriate data and drop unnecessary attributes.
* Rename attributes.
* Alter data types.
* Handle missing values.
* Construct derived attributes.
* Group values to the same level of granularity.
* Join the diferent datasets.
* Split predictors and response variable.


### **6.4 Exploratory Data Analysis** <a class="anchor" id="eda"></a>

The prepared dataset was explored to calculate simple statistical measures, identify time trends, explore distributions and relationships from the unified dataset as well as assessing stationarity, autocorrelation, decomposition and cointegration among attributes.

A high-level view of the time series for all the features is shown below:

<p align="center">
	<img src="Images/fig_variables_over_time.png?raw=true" width=70% height=60%>
</p>

Moreover, it could be seen that the variables **units**, **S&P500**, **WALMEX stock**, and **net sales** have a positive relationship; whereas the **GDP** and **interest rates** showed a moderate positive relationship:

<p align="center">
	<img src="Images/fig_variables_distributions_and_relationships.png?raw=true" width=90% height=90%>
</p>

Furthermore, it could be seen that the correlation among **net sales**, **units**, **WALMEX stock**, and **S&P500** was strong; whereas **GDP**, **interest rates** and **IPC** were moderately correlated. In contrast, **exchange rates**, **GDP** and **IPC** were weakly correlated:

<p align="center">
	<img src="Images/fig_correlation_matrix.png?raw=true" width=70% height=60%>
</p>

Then, the stationarity of the net sales time series was assessed by testing for unit roots with the **Augmented Dickey-Fuller (ADF) test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: Non\text{-}stationary$$

$$H_{1}: Stationary$$

$$\alpha = 0.95$$

```bash
ADF statistic:  1.3627370980379012
P-value:  0.9969381097148852
```

As the $p\text{-}value > 0.05$ then the null hypothesis cannot be rejected. So, the time series is not stationary. 

In this context, the net sales data was transformed using **differencing** to stabilize the trend and seasonality [(Peixeiro, 2022)](#peixeiro):

<p align="center">
	<img src="Images/fig_first-order_differenced_net_sales.png?raw=true" width=70% height=60%>
</p>

```bash
ADF statistic:  -1.6511761047174953
P-value:  0.45642712357493526
```

Even though the ADF statistic is now negative, the $p\text{-}value > 0.05$, so the null hypothesis cannot be rejected. So, the time series is still not stationary after the first-order differencing. 

Thus, the net sales were transformed again with a second-order differencing, and the ADF test was applied once more.

<p align="center">
	<img src="Images/fig_second-order_differenced_net_sales.png?raw=true" width=70% height=60%>
</p>

```bash
ADF statistic:  -4.941259634510584
P-value:  2.89677222031049e-05
```

Now, the ADF statistic is strongly negative and the $p\text{-}value < 0.05$, so the null hypothesis is rejected. The second-order differenced time series is stationary.

Moreover, as the VAR model requires all the time series to be stationary as well [(Peixeiro, 2022)](#peixeiro), the stationarity of the other macroeconomic variables was also tested with the Augmented Dickey-Fuller (ADF) test:

$$H_{0}: Non\text{-}stationary$$

$$H_{1}: Stationary$$

$$\alpha = 0.95$$

```bash
units:
ADF statistic: 0.403
P-value: 0.982
The series is not stationary.

gdp:
ADF statistic: -3.197
P-value: 0.020
The series is stationary.

walmex:
ADF statistic: -1.143
P-value: 0.697
The series is not stationary.

sp500:
ADF statistic: -0.191
P-value: 0.940
The series is not stationary.

ipc:
ADF statistic: -2.044
P-value: 0.268
The series is not stationary.

exchange_rates:
ADF statistic: -2.308
P-value: 0.169
The series is not stationary.

interest_rates:
ADF statistic: -3.185
P-value: 0.021
The series is stationary.

```

Thus, only the *gdp* and *interest_rates* series are stationary, while the other were not. So, for simplicity, first-order differencing transformations were applied to all the series in order to simplify the detransformation when doing predictions with the VAR model.

```bash
units:
ADF statistic: -5.445
P-value: 0.000
The series is stationary.

gdp:
ADF statistic: -6.037
P-value: 0.000
The series is stationary.

walmex:
ADF statistic: -4.932
P-value: 0.000
The series is stationary.

sp500:
ADF statistic: -4.322
P-value: 0.000
The series is stationary.

ipc:
ADF statistic: -5.091
P-value: 0.000
The series is stationary.

exchange_rates:
ADF statistic: -4.562
P-value: 0.000
The series is stationary.

interest_rates:
ADF statistic: -3.253
P-value: 0.017
The series is stationary.
```

Thus, with the first-order differencing was enough to render all the macroeconomic indicators series as stationaries.

Later, the autocorrelation of the net sales time series was assessed by means of the Autorcorrelation Function (ACF) plot, to explore the significant coefficients/lags in the series:

<p align="center">
	<img src="Images/fig_net_sales_autocorrelation.png?raw=true" width=70% height=60%>
</p>

Thus, according to the ACF plot, the lags 2, 3, 4, and 5 are significant, which means that the time series is actually dependent on its past values.

<p align="center">
	<img src="Images/fig_first-order_differenced_net_sales_autocorrelation.png?raw=true" width=70% height=60%>
</p>

For the first-order differenced net sales autocorrelation plot, as there are no consecutive significant lags after lag 2, it is possible to state that the first-order differenced net sales is autocorrelated only at lags 1 and 2.

<p align="center">
	<img src="Images/fig_second-order_differenced_net_sales_autocorrelation.png?raw=true" width=70% height=60%>
</p>

Likewise, as there are no consecutive significant lags after lag 2, it is possible to state that the second-order differenced net sales is autocorrelated only at lags 1 and 2, too. However, it is clear that the overall trend of the ACF follows a sinusoidal-like pattern. This is evidence that the time series is autorregressive [(Pexerio, 2022)](#peixeiro).

As an important conclusion so far, as the ADF test of the first-order differenced net sales showed that the time series was not stationary, and the ACF plot of the first-order differenced net sales showed an autocorrelation at lag 2, then, **the net sales time series is not a random walk**. This means that statistical learning techniques can be successfully applied to estimate forecasts on the data [(Peixeiro, 2022)](#peixeiro).

As part of the explotatory data analysis, the WALMEX net sales series was decomposed using the *seasonal_decompose* method from the **statsmodels** library.

Firstly, an **additive decomposition** was performed:

<p align="center">
	<img src="Images/fig_additive_decomposition_for_walmex_net_sales_training_set.png?raw=true" width=70% height=60%>
</p>

From the plot above, it is clear that the WALMEX net sales adjusted very well to a additive model, as the trend is a straight line with a positive slope, and the seasonality could be isolated very well from the series exhibiting both regular frequency (with a cycle number of 4) and amplitude.

Then, a **multiplicative decomposition** was performed as well on the series:

<p align="center">
	<img src="Images/fig_multiplicative_decomposition_for_walmex_net_sales.png?raw=true" width=70% height=60%>
</p>

As the residuals in the multiplicative decomposition show an uniform value, this is an indication that the decomposition failed to properly capture all the information from the series. So, this decomposition approach is not appropriate.

Finally, to check whether the variables shared a common stochastic trend, or more formally, whether there is a linear combination of these variables that is stable (stationary) [(Kotzé, 2024)](#kotze); **Johansen's Cointegration test** was performed [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

The Johansen test checks the cointegration rank $r$ of a given set of vectors. The null hypothesis of $r=0$ implies no cointegration relationship; whereas a rank of $r\leq1$ means at least one cointegration relationship, a rank of $r\leq2$ means at least two cointegration relationships, etc. [(Wang, 2018)](#wang):

$$H_{0}: Non\text{-}cointegration$$

$$H_{1}: Cointegration$$

$$\alpha = 0.95$$

The test was performed using the most restrictive deterministic term order that assumes a linear trend (```det_order=1```), and one lag (```k_ar_diff=1```):

| Rank | CL 90% | CL 95% | CL 99% | Trace Statistic |
| :----:| :----: | :----: | :----: | :----: | 
| r = 0 | 133.7852 | 139.2780 | 150.0778 | 195.598921 |
| r <= 1 | 102.4674 | 107.3429 | 116.9829 | 135.936953 |
| r <= 2 | 75.1027 | 79.3422 | 87.7748 | 86.363668 |
| r <= 3 | 51.6492 | 55.2459 | 62.5202 | 49.849412 |
| r <= 4 | 32.0645 | 35.0116 | 41.0815 | 27.757403 |
| r <= 5 | 16.1619 | 18.3985 | 23.1485 | 13.290473 |
| r <= 6 | 2.7055 | 3.8415 | 6.6349 | 4.501931 |

Thus, it is possible to easily reject the null hypothesis of $r=0$, as the trace statistic is above the critical values at the 90%, 95%, and 99% confidence levels. However, the test results above suggested that there is up to two cointegration relationships a the 95% confidence level. So, it is not possible to state that all the variables are cointegrated.

To find the two cointegration relationships, from the relationships and correlations sections, it was clear that the variables *units*, *SP&500*, and *WALMEX* were strongly correlated. So, the cointegration test was performed using only those series.

Rank	| CL 90%	| CL 95%	| CL 99%	| Trace Statistic
| :----:| :----: | :----: | :----: | :----: | 
r = 0	| 32.0645	| 35.0116	| 41.0815	| 31.360790
r <= 1	| 16.1619	| 18.3985	| 23.1485	| 14.621989
r <= 2	| 2.7055	| 3.8415	| 6.6349	| 4.138262

From the results above, it is possible to reject the null hypothesis of $r=0$, as the trace statistic for $r<=2$ is above the critical value at the 95% confidence level.

Therefore, the Vector models (VAR, VARMA, VARIMA) were later built with the above-mentioned series.


### **6.5 Modeling** <a class="anchor" id="modeling"></a>

Firstly, ten univariate time series models were built to predict the net sales of WALMEX: 

- **Moving Average (MA) model**, 
- **Autoregressive (AR) model**, 
- a series of **Autoregressive (AR) models** with **Additive Decomposition**, 
- **Autoregressive Moving Average (ARMA) model**, 
- **Autoregressive Integrated Moving Average (ARIMA) model**, 
- **Seasonal Autoregressive Integrated Moving Average (SARIMA) model**, 
- **Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) Model**, 
- **Simple Exponential Smoothing (SES) model**, 
- **Holt-Winters (HW) model**, and
- **Prophet Univariate Time Series Modeling**.

Then, three vector models were created and trained in Python 3 and its libraries Statsmodels and Darts to predict the values of the selected macroeconomic indicators as a multivariate time series:
- **Vector Autoregressive (VAR) model**, 
- **Vector Autoregressive Moving Average (VARMA) model**, and 
- **Vector Autoregressive Integrated Moving Average (VARIMA) model**  

After that, two regression models were built using **Random Forests** and **Support Vector Machines** in Python 3 and its library Scikit-learn to predict WALMEX total sales based on the predictions for the best performing multivariate time series model. 

On the other hand, the models were assessed using **Root Mean Square Error (RMSE)**, **Mean Absolute Error (MAE**), and the **Coefficient of Determination ($r^{2}$)**. 

For reporting purposes, the model assessment plots were shown in the original scale of the WALMEX net sales data.

Finally, for sake of clarity, each model was described separately in the present section.

<br>

#### **6.5.1 Moving Average (MA) Model** <a class="anchor" id="ma_model"></a>

A **Moving Average (MA) model** was built as a baseline model. As suggested by the name of this technique, the MA model calculates the moving average changes over time for a certain variable using a specified window lenght [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni). This technique has the advantage of removing random fluctuations and smoothen the data. The MA model is denoted as **MA($q$)**, where $q$ is the number of past error terms that affects the present value of the time series [(Peixeiro, 2022)](#peixeiro).

The model was built using the library **pandas** in Python. 

It was assumed that the current values are linearly dependent on the mean of the series, the current and past error terms; the errors are normally distributed [(Peixeiro, 2022)](#peixeiro). On the other hand, from the ACF plot in the [EDA section](#acf), it was assumed that a window of 5 was enough to capture the overall trend of the data, as the ACF plot showed that lag 5 was the last significant coefficient.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively. 

Then, the model was built as follows:

```python
# Window length
ma_window = 5

# Calculating the moving average
y_pred_ma_model = y.rolling(ma_window).mean()

# Narrowing down the moving average calculations to the y_test time period
y_pred_ma_model = y_pred_ma_model[len(y_train) - 1:]
y_pred_ma_model
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

The predictions were plot against the historical net sales data to visually assess the performance of the MA model.

<p align="center">
	<img src="Images/fig_predictions_from_ma_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the MA model was able to capture the trend of the time series but not their stationality.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
net_sales
RMSE: 15216.901
MAE: 9651.900
Coefficient of Determination: 0.465
```

<br>

#### **6.5.2 Autoregressive (AR) Model** <a class="anchor" id="ar_model"></a>

As suggested by the ACF plot in the EDA, a **Autoregressive (AR) model** was built to forecast the net sales of WALMEX. 

An autoregressive model implies a regression of a variable against itself, which means that the present value of a given point in a time series is linearly dependent on its past values [(Peixeiro, 2022)](#peixeiro). 
An AR model is denoted as **AR($p$)**, where the order $p$ determines the number of past values that affect the present value  [(Peixeiro, 2022)](#peixeiro).

The model was built using the library **statmodels** in Python.

It was assumed that the current values are linearly dependent on their past values [(Peixeiro, 2022)](#peixeiro). Moreover, it was assumed that the time series was stationary, it was not a random walk, and seasonality effects were not relevant for modeling.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively. It is important to note that the second-order differenced net sales time series was used for modeling, as AR models require that the time series be stationary. 

Then, the order for each AR Model was identified by using the **partial autocorrelation function (PACF) plot** to assess the effect of past data (the so-called lags) on future data [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

<p align="center">
	<img src="Images/fig_partial_autocorrelation_for_second-order_differenced_net_sales.png?raw=true" width=70% height=60%>
</p>

So, the order of the AR model for the trend component was defined as 3.

Then, the model was built as follows:

```python
ar_model = AutoReg(y_diff2, lags=3).fit()
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

The predictions were plot against the historical net sales data to visually assess the performance of the AR model.

<p align="center">
	<img src="Images/fig_predictions_from_ar_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the AR model was able to capture both the trend and the stationality of the time series.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
net_sales
RMSE: 7687.806
MAE: 6558.916
Coefficient of Determination: 0.863
```

<br>

#### **6.5.3 Autoregressive (AR) Models with Additive Decomposition** <a class="anchor" id="ar_model_add_decomp"></a>

A set of **Autoregressive (AR) Models** based on the **Additive Decomposition** of the WALMEX net sales time series were built. The **Additive Decomposition** was deemed as an appropiate decomposition technique as the EDA suggested that the WALMEX net sales behave in a linear fashion, with constant changes over time, and with a regular seasonality with equal frequency and amplitude [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni). Then, a set of **Autoregressive Models** were built for each component of the time series obtained via the decomposition: trend, seasonality, and remainder. The net sales were forecasted by adding up the predictions from each AR model [(Jamieson, 2019)](#jamieson).

Likewise, the library **statsmodels** in Python was used to perform the **Additive Decomposition** and build the **AR Models**.

It is assumed that the components in the net sales time series can be added together as their changes over time are regular, the time trend is a straight line, and the seasonality has the same frequency and amplitude [(Brownlee, 2020)](#brownlee).

Moreover, it is assumed that each component of the time series can be predicted by means of a linear combination of their historical or lag values [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

As showed in the EDA, WALMEX net sales adjusted very well to an additive decomposition. So the training set was decomposed using the **Additive model** with the class *seasonal_decompose* from the **statsmodels** library. However, the AR model require the time series to be stationary, so the trend was differenced several times until the ADF test indicated that no unit root was found in the data.

Then, the model was built as follows:

```python
trend_ar_model = AutoReg(trend_diff3, lags=1).fit()
seasonality_ar_model = AutoReg(seasonality, lags=3).fit()
remainder_ar_model = AutoReg(remainder, lags=1).fit()
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

Likewise, the predictions were plot against the historical net sales data to visually assess the performance of the AR models with additive decomposition.

<p align="center">
	<img src="Images/fig_predictions_from_ar_models_with_additive_decomposition_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the additive decomposition and the AR models were able to provide closer predictions for the WALMEX net sales.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:


```bash
RMSE: 11272.204
MAE: 8970.403
Coefficient of Determination: 0.706
```

<br>

#### **6.5.4 Autoregressive Moving Average (ARMA) Model** <a class="anchor" id="arma_model"></a>

An **Autoregressive Moving Average (ARMA) model** based on the WALMEX net sales time series was built.

The ARMA model is a combination of the autoregressive process and the moving average process, and it is denoted as **ARMA($p$,$q$)**, where $p$ is the order of the autoregressive process, and $q$ is the order of the moving average process. The order of $p$ determines the number of past values that affect the present value; whereas the order of $q$ determines the number of past error terms that affect the present value [(Peixeiro, 2022)](#peixeiro). Indeed, the ACF and PACF plots shown earlier suggested that the time series was neither a pure moving average or a pure autorregressive process as sinusoidal patterns were found and no clear cut-off values between significant and no significant lags were identified. So, the ARMA model could yield better prediction results [(Peixeiro, 2022)](#peixeiro).

Several orders $p$ and $q$, for the autoregressive and moving average portions of the model, respectively, were tested and the best performing model was selected according to the Akaike Information Criterion (AIC) and a residual analysis [(Peixeiro, 2022)](#peixeiro).

Likewise, the function **ARIMA** from the library **statsmodels** in Python was used to build the **ARMA model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

It was assumed that the present values in the time series are linearly dependend on both its past values and on the mean, the current error and the past errors terms [(Peixeiro, 2022)](#peixeiro). Another key assumption is that the dataset was stationary when performing a second-order differencing. Moreover, it was assumed that the residuals from the model were normally distributed, independent, and uncorrelated; approximanting to white noise [(Peixeiro, 2022)](#peixeiro). 

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Then, the model was built as follows:

```python
def ARMA_model(p, q, time_series):
        """
        Fits and optimize an autoregressive moving average (ARMA) model given a set of p and q values, minimizing 
        the Akaike Information Criterion (AIC).

        Parameters:
        p (range): Range for order p in the autoregressive portion of the ARMA model.
        q (range): Range for order q in the moving average portion of the ARMA model.
        time_series (pandas.series): Time series data for fitting the ARMA model.

        Returns:
        ARMA_model (statsmodels.arima): An ARMA model object fitted according to the combination of p and q that minimizes 
        the Akaike Information Criterion (AIC).
        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                ARMA_model = ARIMA(time_series, order = (order[0], 0, order[1])).fit()
                order_results.append(order)
                aic_results.append(ARMA_model.aic)
        
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
        ARMA_model = ARIMA(time_series, order = (best_model[0], 0, best_model[1])).fit()

        return ARMA_model

arma_model = ARMA_model(p=range(1,6), q=range(1,7), time_series=y_train)
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After modeling, a residual analysis was carried out to assess whether the difference between the actual and predicted values of the model is due to randomness or, in other words, that the residuals are normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

To do so, the residual analysis was performed in two steps [(Peixeiro, 2022)](#peixeiro): 
* **Quantile-quantile plot (Q-Q plot)**: To qualitatively assess whether the residuals from the model are normally distributed.
* **Ljung-Box test**: To test whether the residuals from the model are uncorrelated.

<p align="center">
	<img src="Images/fig_diagnostic_plots_for_standardized_residuals_from_arma_model.png?raw=true" width=70% height=60%>
</p>

From the diagnostics plots, the residuals are not completely normally distributed. However, it was considered that the results were good enough for the purposes of the present study, as other more sophisticated models were fit below.

Then, the residuals were assessed by testing for uncorrelation with the **Ljung-Box test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: No\text{-}autocorrelation$$

$$H_{1}: Autocorrelation$$

$$\alpha = 0.95$$

| lag | lb_stat  | lb_pvalue |
| --- | -------- | --------- |
| 1   | 3.271269 | 0.070503  |
| 2   | 3.40143  | 0.182553  |
| 3   | 3.420409 | 0.331233  |
| 4   | 3.519843 | 0.474868  |
| 5   | 3.689612 | 0.594911  |
| 6   | 3.763542 | 0.708639  |
| 7   | 4.246189 | 0.751025  |
| 8   | 7.457154 | 0.488205  |
| 9   | 7.726973 | 0.561878  |
| 10  | 8.526754 | 0.577525  |

For each of the lags from 1 to 10, the $p\text{-}values$ were above $0.05$. Thus, the null hypothesis cannot be rejected and no autocorrelation was found on the set of residuals from the ARMA model. This means that the residuals are independently distributed and the model can be used for forecasting.

Likewise, the predictions were plot against the historical net sales data to visually assess the performance of the AR models with additive decomposition.

<p align="center">
	<img src="Images/fig_predictions_from_arma_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the ARMA model was able to provide very close predictions for the WALMEX net sales.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:


```bash
RMSE: 5006.673
MAE: 4190.297
Coefficient of Determination: 0.942
```

<br>

#### **6.5.5 Autoregressive Integrated Moving Average (ARIMA) Model** <a class="anchor" id="arima_model"></a>

An **Autoregressive Integrated Moving Average (ARIMA) model** based on the WALMEX net sales time series was built.

The ARIMA is a combination of the autoregressive process, integration and the moving average process for forecasting of non-stationary time series, meaning that the time series has a trend, and/or its variance is not constant. [(Peixeiro, 2022)](#peixeiro). It is denoted as **ARIMA($p$,$d$, $q$)**; where $p$ is the order of the autoregressive process, $d$ is the integration order, and $q$ is the order of the moving average process. The order of $p$ determines the number of past values that affect the present value, the order of $q$ determines the number of past error terms that affect the present value, and the order of integration $d$ indicates the number of times a time series has been differenced to become stationary [(Peixeiro, 2022)](#peixeiro).

Similar to the building of the ARMA model, several orders $p$ and $q$, for the autoregressive and moving average portions of the model, respectively, were tested and the best performing model was selected according to the Akaike Information Criterion (AIC) and a residual analysis [(Peixeiro, 2022)](#peixeiro).

The function **ARIMA** from the library **statsmodels** in Python was used to build the **ARIMA model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

It was assumed that the present values in the time series are linearly dependend on both its past values and on the mean, the current error and the past errors terms [(Peixeiro, 2022)](#peixeiro). It was also assumed that the residuals from the model were normally distributed, independent, and uncorrelated; approximanting to white noise [(Peixeiro, 2022)](#peixeiro).  
On the other hand, from the EDA, it was determined that the order of integration for the WALMEX net sales is $d = 2$.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Then, the model was built as follows:

```python
def ARIMA_model(p, d, q, time_series):
        """
        Fits and optimize an autoregressive integrated moving average (ARIMA) model based on the Akaike 
        Information Criterion (AIC), given a set of p and q values; while keeping the d order constant. 

        Parameters:
        p (range): Range for order p in the autoregressive portion of the ARIMA model.
        d (int): Integration order.
        q (range): Range for order q in the moving average portion of the ARIMA model.
        time_series (pandas.series): Time series data for fitting the ARIMA model.

        Returns:
        ARIMA_model (statsmodels.arima): An ARIMA model object fitted according to the combination of p and q that minimizes 
        the Akaike Information Criterion (AIC).
        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                ARIMA_model = ARIMA(time_series, order = (order[0], d, order[1])).fit()
                order_results.append((order[0], d, order[1]))
                aic_results.append(ARIMA_model.aic)
        
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
        ARIMA_model = ARIMA(time_series, order = (best_model[0], best_model[1], best_model[2])).fit()

        return ARIMA_model

arima_model = ARIMA_model(p=range(1,6), d=2, q=range(1,7), time_series=y_train)
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After modeling, a residual analysis was carried out to assess whether the difference between the actual and predicted values of the model is due to randomness or, in other words, that the residuals are normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

To do so, the residual analysis was performed in two steps [(Peixeiro, 2022)](#peixeiro): 
* **Quantile-quantile plot (Q-Q plot)**: To qualitatively assess whether the residuals from the model are normally distributed.
* **Ljung-Box test**: To test whether the residuals from the model are uncorrelated.

<p align="center">
	<img src="Images/fig_diagnostic_plots_for_standardized_residuals_from_arima_model.png?raw=true" width=70% height=60%>
</p>

From the diagnostics plots, the residuals are not completely normally distributed. However, it was considered that the results were good enough for the purposes of the present study.

Then, the residuals were assessed by testing for uncorrelation with the **Ljung-Box test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: No\text{-}autocorrelation$$

$$H_{1}: Autocorrelation$$

$$\alpha = 0.95$$

| lag | lb_stat  | lb_pvalue |
| --- | -------- | --------- |
| 1   | 0.996298 | 0.318208  |
| 2   | 1.272624 | 0.529241  |
| 3   | 1.287834 | 0.732024  |
| 4   | 2.389698 | 0.66449   |
| 5   | 2.507422 | 0.775377  |
| 6   | 2.584535 | 0.858889  |
| 7   | 2.58462  | 0.920591  |
| 8   | 3.390625 | 0.907511  |
| 9   | 3.550224 | 0.938451  |
| 10  | 3.915916 | 0.95106   |

For each of the lags from 1 to 10, the $p\text{-}values$ were well above $0.05$. Thus, the null hypothesis cannot be rejected, meaning that no autocorrelation was found on the set of residuals from the ARIMA model. Thus, the residuals are independently distributed and the model can be used for forecasting.

Likewise, the predictions were plot against the historical net sales data to visually assess the performance of the ARIMA model.

<p align="center">
	<img src="Images/fig_predictions_from_arima_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, is seems that the ARIMA model was not able to capture the seasonality of the time series. Notably, the ARMA model was able to yield even better predictions.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:


```bash
RMSE: 27274.028
MAE: 24120.853
Coefficient of Determination: -0.720
```

<br>

#### **6.5.6 Seasonal Autoregressive Integrated Moving Average (SARIMA) Model** <a class="anchor" id="sarima_model"></a>

A **Seasonal Autoregressive Integrated Moving Average (SARIMA) model** based on the WALMEX net sales time series was built. The SARIMA model is a combination of the autoregressive process, integration and the moving average process for forecasting of non-stationary time series, but also accounting for seasonal patterns. [(Peixeiro, 2022)](#peixeiro). It is denoted as ${\text{SARIMA}(p,d,q)(P,D,Q)_{m}}$; where $p$ is the order of the autoregressive process, $d$ is the integration order, $q$ is the order of the moving average process, $m$ is the frequency, and $P$, $D$, and $Q$ are the orders for the seasonal parameters for the autoregressive, integration, and moving average processes, respectively [(Peixeiro, 2022)](#peixeiro).

Similar to the building of the ARMA and ARIMA models, several orders $p$, $q$, $P$, and $Q$ were tested and the best performing model was selected according to the Akaike Information Criterion (AIC) and a residual analysis [(Peixeiro, 2022)](#peixeiro).

The function **SARIMAX** from the library **statsmodels** in Python was used to build the **SARIMA model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

It is assumed that the present values in the time series are linearly dependend on both its past values and on the mean, the current error and the past errors terms [(Peixeiro, 2022)](#peixeiro). 

It was also assumed that the residuals from the model were normally distributed, independent, and uncorrelated; approximanting to white noise [(Peixeiro, 2022)](#peixeiro). 

Furthermore, as the time series is reported in a quaterly basis, the frequency or number of observations per seasonal cycle was defined as $m = 4$. This assumption was also supported by the additive decomposition of the time series carried out above.

Finally, from the EDA, it was found that the original time series was not stationary. In this sense, it was proposed to use the first-differenced time series as a basis for seasonal differencing. So, the order of integration for the WALMEX net sales was $d = 1$ for the SARIMA model.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Then, the model was built as follows:

```python
def SARIMA_model(p, d, q, P, D, Q, m, time_series):
        """
        Fits and optimize a seasonal autoregressive integrated moving average (SARIMA) model based on the Akaike 
        Information Criterion (AIC), given a set of p, q, P, and Q values; while keeping the d and D orders constant. 
        The frequency m is also kept constant.        

        Parameters:
        p (range): Range for order p in the autoregressive portion of the SARIMA model.
        d (int): Integration order.
        q (range): Range for order q in the moving average portion of the SARIMA model.
        P (range): Range for order P in the seasonal autoregressive portion of the SARIMA model.
        D (int): Seasonal integration order.
        Q (range): Range for order P in the seasonal moving average portion of the SARIMA model.
        m (int): Number of observations per seasonal cycle.
        time_series (pandas.series): Time series data for fitting the SARIMA model.

        Returns:
        SARIMA_model (statsmodels.sarimax): An SARIMAX model object fitted according to the combination of p, q, P, 
        and Q values that minimizes the Akaike Information Criterion (AIC).
        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q, P, Q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                SARIMA_model = SARIMAX(endog=time_series, 
                                       order = (order[0], d, order[1]),
                                       seasonal_order=(order[2], D, order[3], m),
                                       ).fit(disp=False)
                order_results.append((order[0], d, order[1], order[2], D, order[3], m))
                aic_results.append(SARIMA_model.aic)
        
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
        SARIMA_model = SARIMAX(endog=time_series, 
                                order = (best_model[0], d, best_model[2]),
                                seasonal_order=(best_model[3], D, best_model[5], m),
                                ).fit(disp=False)

        return SARIMA_model

sarima_model = SARIMA_model(p=range(1,4), 
                            d=1, 
                            q=range(1,4), 
                            P=range(1,5), 
                            D=1, 
                            Q=range(1,5), 
                            m=4, 
                            time_series=y_train)
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After modeling, a residual analysis was carried out to assess whether the difference between the actual and predicted values of the model is due to randomness or, in other words, that the residuals are normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

To do so, the residual analysis was performed in two steps [(Peixeiro, 2022)](#peixeiro): 
* **Quantile-quantile plot (Q-Q plot)**: To qualitatively assess whether the residuals from the model are normally distributed.
* **Ljung-Box test**: To test whether the residuals from the model are uncorrelated.

<p align="center">
	<img src="Images/fig_diagnostic_plots_for_standardized_residuals_from_sarima_model.png?raw=true" width=70% height=60%>
</p>

From the diagnostics plots, the residuals are not completely normally distributed. However, it was considered that the results were good enough for the purposes of the present study.

Then, the residuals were assessed by testing for uncorrelation with the **Ljung-Box test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: No\text{-}autocorrelation$$

$$H_{1}: Autocorrelation$$

$$\alpha = 0.95$$

|    | lb_stat  | lb_pvalue |
| -- | -------- | --------- |
| 1  | 0.296909 | 0.585827  |
| 2  | 0.297667 | 0.861713  |
| 3  | 0.51856  | 0.914795  |
| 4  | 8.331227 | 0.080171  |
| 5  | 8.331242 | 0.1389    |
| 6  | 8.345408 | 0.213874  |
| 7  | 8.346893 | 0.303001  |
| 8  | 8.347182 | 0.400312  |
| 9  | 8.355757 | 0.49873   |
| 10 | 8.571613 | 0.573183  |

For each of the lags from 1 to 10, the $p\text{-}values$ were above $0.05$. Thus, the null hypothesis cannot be rejected, meaning that no autocorrelation was found on the set of residuals from the SARIMA model. Thus, the residuals are independently distributed and the model can be used for forecasting.

Later, the predictions were plot against the historical net sales data to visually assess the performance of the SARIMA model.

<p align="center">
	<img src="Images/fig_predictions_from_sarima_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the SARIMA model was able to neatly capture the trend and seasonality of the time series. So far, the best performance obtained.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
RMSE: 2675.576
MAE: 2372.531
Coefficient of Determination: 0.983
```

<br>

#### **6.5.7 Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) Model** <a class="anchor" id="sarimax_model"></a>

A **Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) model** based on the WALMEX net sales, units,	GDP, stock value, SP&500, the IPC, exchange rates, and interest rates time series was built.

The SARIMAX model is a combination of the autoregressive process, integration, moving average process for forecasting of non-stationary time series with seasonal patterns, but also including the effects of external variables [(Peixeiro, 2022)](#peixeiro). It is denoted as $\bold{\text{SARIMA}(p,d,q)(P,D,Q)_{m} + \sum_{i=1} ^{n} \beta_i X_t^i}$; where $p$ is the order of the autoregressive process, $d$ is the integration order, $q$ is the order of the moving average process, $m$ is the frequency; $P$, $D$, and $Q$ are the orders for the seasonal parameters for the autoregressive, integration, and moving average processes, respectively; and $X_t$ are any number of exogenous variables with their corresponding coefficients $\beta$ [(Peixeiro, 2022)](#peixeiro).

Similar to the building of the ARMA, ARIMA, and SARIMA models, several orders $p$, $q$, $P$, and $Q$ were tested and the best performing model was selected according to the Akaike Information Criterion (AIC) and a residual analysis [(Peixeiro, 2022)](#peixeiro).

The function **SARIMAX** from the library **statsmodels** in Python was used to build the **SARIMAX model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

It is assumed that the present values in the time series are linearly dependend on both its past values and on the mean, the current error and the past errors terms [(Peixeiro, 2022)](#peixeiro). It was also assumed that the residuals from the model were normally distributed, independent, and uncorrelated; approximanting to white noise [(Peixeiro, 2022)](#peixeiro). Furthermore, from the EDA, it was found that the original time series was not stationary. In this sense, it was proposed to use the first-differenced time series as a basis for seasonal differencing. So, the order of integration for the WALMEX net sales was $d = 1$ for the SARIMAX model.

As the time series is reported in a quaterly basis, the frequency or number of observations per seasonal cycle was defined as $m = 4$. This assumption was also supported by the additive decomposition of the time series carried out above. 

Finally, from the seasonal differencing carried out above at the SARIMA model, the seasonal order of integration was defined as $D = 1$.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Then, the model was built as follows:

```python
def SARIMAX_model(p, d, q, P, D, Q, m, endog, exog):
        """
        Fits and optimize a seasonal autoregressive integrated moving average with exogeneous variables (SARIMAX) model 
        based on the  Akaike Information Criterion (AIC), given a set of p, q, P, and Q values; while keeping the 
        d and D orders constant. The frequency m is also kept constant.

        Parameters:
        p (range): Range for order p in the autoregressive portion of the SARIMA model.
        d (int): Integration order.
        q (range): Range for order q in the moving average portion of the SARIMA model.
        P (range): Range for order P in the seasonal autoregressive portion of the SARIMA model.
        D (int): Seasonal integration order.
        Q (range): Range for order P in the seasonal moving average portion of the SARIMA model.
        m (int): Number of observations per seasonal cycle.
        endog (pandas.series): Time series of the endogenous variable for fitting the SARIMAX model.
        exog (pandas.dataframe): Time series of the exogenous variables for fitting the SARIMAX model.

        Returns:
        SARIMAX_model (statsmodels.sarimax): An SARIMAX model object fitted according to the combination of p, q, P, 
        and Q values that minimizes the Akaike Information Criterion (AIC).
        

        """
        # Obtaining the combinations of p and q
        order_list = list(product(p, q, P, Q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                SARIMAX_model = SARIMAX(endog=endog, 
                                        exog=exog,
                                       order = (order[0], d, order[1]),
                                       seasonal_order=(order[2], D, order[3], m),
                                       ).fit(disp=False)
                order_results.append((order[0], d, order[1], order[2], D, order[3], m))
                aic_results.append(SARIMAX_model.aic)
        
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
        SARIMAX_model = SARIMAX(endog=endog, 
                                exog=exog, 
                                order = (best_model[0], d, best_model[2]),
                                seasonal_order=(best_model[3], D, best_model[5], m),
                                ).fit(disp=False)

        return SARIMAX_model

sarimax_model = SARIMAX_model(p=range(1,4), 
                            d=1, 
                            q=range(1,4), 
                            P=range(1,2), 
                            D=1, 
                            Q=range(1,2), 
                            m=4, 
                            endog=y_train, 
                            exog=exog_train)
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After modeling, a residual analysis was carried out to assess whether the difference between the actual and predicted values of the model is due to randomness or, in other words, that the residuals are normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

To do so, the residual analysis was performed in two steps [(Peixeiro, 2022)](#peixeiro): 
* **Quantile-quantile plot (Q-Q plot)**: To qualitatively assess whether the residuals from the model are normally distributed.
* **Ljung-Box test**: To test whether the residuals from the model are uncorrelated.

<p align="center">
	<img src="Images/fig_diagnostic_plots_for_standardized_residuals_from_sarimax_model.png?raw=true" width=70% height=60%>
</p>

From the diagnostics plots, the residuals are not completely normally distributed. However, it was considered that the results were good enough for the purposes of the present study.

Then, the residuals were assessed by testing for uncorrelation with the **Ljung-Box test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: No\text{-}autocorrelation$$

$$H_{1}: Autocorrelation$$

$$\alpha = 0.95$$

|    | lb_stat  | lb_pvalue |
| -- | -------- | --------- |
| 1  | 0.21327  | 0.644216  |
| 2  | 0.220932 | 0.895417  |
| 3  | 0.825297 | 0.843407  |
| 4  | 9.256681 | 0.054994  |
| 5  | 9.257353 | 0.099229  |
| 6  | 9.259061 | 0.159524  |
| 7  | 9.259126 | 0.234583  |
| 8  | 9.272412 | 0.319839  |
| 9  | 9.272497 | 0.41251   |
| 10 | 9.292785 | 0.504561  |

For each of the lags from 1 to 10, the $p\text{-}values$ were above $0.05$. Thus, the null hypothesis cannot be rejected, meaning that no autocorrelation was found on the set of residuals from the SARIMAX model. Thus, the residuals are independently distributed and the model can be used for forecasting.

Later, the predictions were plot against the historical net sales data to visually assess the performance of the SARIMA model.

<p align="center">
	<img src="Images/fig_predictions_from_sarimax_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the SARIMAX model was able to capture the trend and seasonality of the time series. However, its performance was inferior to that from the SARIMA model.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
RMSE: 21608.095
MAE: 20415.994
Coefficient of Determination: -0.080
```

<br>

#### **6.5.8 Simple Exponential Smoothing (SES) Model** <a class="anchor" id="ses_model"></a>

A **Simple Expotential Smoothing (SES) model** based on the WALMEX net sales was also built. The SES model is a smoothening technique that uses an exponential window function [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni). It is useful when a time series do not exhibit neither trend nor seasonality [(Atwan, 2022)](#atwan).

The function **SimpleExpSmoothing** from the library **statsmodels** in Python was used to build the **SES model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

As the SES model requires a time series process to be stationary. So, when using the second-order differenced series, it was assumed that the stationarity requirement was fulfilled. Furthermore, it was assumed that the seasonal patterns were neglectable for the purposes of this model.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Then, the model was built as follows:

```python
ses_model = SimpleExpSmoothing(y_train).fit()
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

Likewise, the predictions were plot against the historical net sales data to visually assess the performance of the SES model.

<p align="center">
	<img src="Images/fig_predictions_from_ses_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, as expectable, the SES model was not able to capture neither the trend nor the seasonality of the time.

Then, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:


```bash
RMSE: 180107.784
MAE: 166655.346
Coefficient of Determination: -74.013
```

<br>

#### **6.5.9 Holt-Winters (HW) Model** <a class="anchor" id="hw_model"></a>

A **Holt-Winters (HW) model** based on the WALMEX net sales was built. The HW model is a smoothening technique that uses an exponential weighted moving average process [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni). It is an enhancement over simple exponential smoothing, and can be used when a time series exhibit both a trend and seasonality [(Atwan, 2022)](#atwan).

The function **ExponentialSmoothing** from the library **statsmodels** in Python was used to build the **HW model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

It has been assumed that the futures values of the time series are dependent upon the past errors terms.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Then, the model was built as follows:

```python
hw_model = ExponentialSmoothing(y_train, trend='additive', seasonal='additive', seasonal_periods=4).fit()
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After that, the predictions were plot against the historical net sales data to visually assess the performance of the HW model.

<p align="center">
	<img src="Images/fig_predictions_from_hw_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the HW model performed notably better than the SES model, as it was able to capture both the trend and seasonality of the WALMEX net sales series.

Finally, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:


```bash
RMSE: 9508.312
MAE: 7645.737
Coefficient of Determination: 0.791
```

<br>

#### **6.5.10 Prophet Univariate Time Series Model** <a class="anchor" id="prophet_model"></a>

A **Univariate Time Series model using Prophet** based on the WALMEX net sales was built. *Prophet* is an open source package built and maintained by Meta for forecasting at scale. It was released in 2017 [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni), and it is able to fit nonlinear trends and multiple seasonalities [(Peixeiro, 2022)](#peixeiro). Prophet implements a general additive model where a time series $y(t)$ is modeled as the linear combination of a trend $g(t)$, a seasonal component $s(t)$, holiday effects $h(t)$, and an error term $ϵ_{t}$ [(Peixeiro, 2022)](#peixeiro):

$$y(t) = g(t) + s(t) + h(t) + ϵ_{t}$$

Even though the model does not take into account any autoregressive process, it has the advantage to be very flexible as "it can accommodate multiple seasonal periods and changing trends [and] it is robust to outliers and missing data" [(Peixeiro, 2022)](#peixeiro):

The function **Prophet** from the library **prophet** in Python was used to build the **model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

It has been assumed that the net sales series has a non-linear trend, at least one seasonal component, holiday effects, and that the error terms are normally distributed.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

The model built with Prophet was optimized by using cross-validation and hyperparameter tuning. Specifically, the hypeparameters `changepoint_prior_scale` and `seasonality_prior_scale` were tunned.

```python
def prophet_model(series, changepoint_prior_scale, seasonality_prior_scale, metric):
        """
        Fits and optimizr a univariate time series model with Prophet, given a set of changepoint_prior_scale and 
        seasonality_prior_scale values.

        Parameters:
        changepoint_prior_scale (list): Values for the changepoint_prior_scale hyperparameter in Prophet.
        seasonality_prior_scale (list): Values for the seasonality_prior_scale hyperparameter in Prophet.        
        series (pandas.dataframe): Time series data in two columns: ds for dates in a date datatype, and y for the series.
        metric (str): Selected performance metric for optimization: One of 'mse', 'rmse', 'mae', 'mdape', or 'coverage'.
        
        Returns:
        m (prophet.prophet): An Prophet model object optimized according to the combination of tested hyperparameters, by using the 
        indicated metric.
        
        """
        # Obtaining the combinations of hyperparameters
        params = list(product(changepoint_prior_scale, seasonality_prior_scale))

        # Creating emtpy lists to store performance results        
        metric_results = []

        # Defining cutoff dates
        start_cutoff_percentage = 0.5 # 50% of the data will be used for fitting the model
        start_cutoff_index = int(round(len(series) * start_cutoff_percentage, 0))
        start_cutoff = series.iloc[start_cutoff_index].values[0] 
        end_cutoff = series.iloc[-4].values[0] # The last fourth value is taken as the series is reported in a quarterly basis

        cutoffs = pd.date_range(start=start_cutoff, end=end_cutoff, freq='12M')

        # Fitting models
        for param in params:
                m = Prophet(changepoint_prior_scale=param[0], seasonality_prior_scale=param[1])
                m.add_country_holidays(country_name='MX')
                m.fit(series)
        
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
        m.fit(series);        

        return m

# Defining hyperparameters values
changepoint_prior_scale = [0.001, 0.01, 0.1, 0.5]
seasonality_prior_scale = [0.01, 0.1, 1.0, 10.0]

# Fitting model
prophet_model = prophet_model(y_train, changepoint_prior_scale, seasonality_prior_scale, 'rmse')
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

The WALMEX Net Sales Forecast Components from Prophet Model are shown below:

<p align="center">
	<img src="Images/fig_walmex_net_sales_forecast_components_from_prophet_model.png?raw=true" width=70% height=60%>
</p>

Thus, according to the chart above, the model has detected strong seasonalities in the WALMEX net sales at the beginning and at the end of the year, which may correspond to several important holidays in Mexico such as the Independence Day and Christmas.

After that, the predictions were plot against the historical net sales data to visually assess the performance of the Prophet model.

<p align="center">
	<img src="Images/fig_predictions_from_prophet_model_vs_walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, the Prophet model performed well as it was able to capture both the trend and seasonality of the WALMEX net sales series.

Finally, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:


```bash
RMSE: 12323.068
MAE: 10710.708
Coefficient of Determination: 0.649
```

<br>

#### **6.5.11 Vector Autoregressive (VAR) Model** <a class="anchor" id="var_model"></a>

As shown above, the original dataset comprised $40$ historical data points about $7$ features (economic indicators) and $1$ response variable (net sales of WALMEX in millions of MXN). However, based on the results from the Johanssen's cointegration test, only the series *units*, *SP&500*, and *WALMEX* shared enough the same stochastic trend to be used in a multivariate times series model.

A first model for **multivariate times series prediction** was built using a **Vector Autoregression (VAR) model** for forecasting the $3$ selected features. A VAR model was selected due to its ability to forecast multiple features in an easy manner by using the library **statsmodels** in Python [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni). The VAR($p$) model describes the relationship among two or more time series and the impact that their past values have on each other [(Peixeiro, 2022)](#peixeiro). In this sense, the VAR($p$) model is a generalization of the AR model for multiple time series, and the order $p$ determines how many lagged values impact the present value of the different time series [(Peixeiro, 2022)](#peixeiro).

Several orders of $p$ were tested and the best performing model was selected according to the Akaike Information Criterion (AIC). Then, the Granger causality test was applied, which determines whether past values of a time series are statistically significant in forecasting another time series. Then, a residual analysis was finally carried out to assess whether the residuals were close to white noise [(Peixeiro, 2022)](#peixeiro).

The function **VAR** from the library **statsmodels** in Python was used to build the **VAR model** [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni).

It is assumed that the features are cointegrated, i.e., the different features share an underlying stochastic trend, and that the linear combination of the different features is stationary [(Clower, 2020)](#clower). 
Moreover, it was assumed that the current values are linearly dependent on their past values [(Peixeiro, 2022)](#peixeiro); and that the different time series were stationary, they were not a random walk, and seasonality effects were not relevant for modeling.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Later, the VAR model was built using the class *VAR* from the **statsmodels** library.

```python
def VAR_model(p, series):
        """
        Fits and optimizar VAR models using the method VAR from statsmodels given a set of lags, and returns the one
        with the lowest Akaike Information Criterion (AIC).

        Parameters:
        p (list): Lag values.
        series (pandas.dataframe): Time series data.

        Returns:
        model (statsmodels.var): VAR model object optimized according to the AIC criterion.

        """

        # Creating empty lists to store results

        aic_results = []

        for lag in p:
                VAR_model = VAR(endog=series).fit(maxlags=lag)
                aic_results.append(VAR_model.aic)
        
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
        VAR_model = VAR(endog=series).fit(maxlags=best_model)

        return VAR_model

p = list(range(1,6))

var_model = VAR_model(p=p, series=X_train)

```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After that, the Granger causality test was applied to determine whether past values of a time series are statistically significant in forecasting another time series [(Peixeiro, 2022)](#peixeiro). This test is unidirectional, so it has to be applied twice for each series.

$$H_{0}: y_{2,t}\text{ does not Granger-cause }y_{1,t}$$

$$H_{1}: y_{2,t}\text{ Granger-cause }y_{1,t}$$

$$\alpha = 0.95$$

|   | y2     | y1     | Granger Causality | Test Interpretation                   | SSR F-test p-values | SSR Chi2-test p-values | LR-test p-values | Params F-test p-values |
| - | ------ | ------ | ----------------- | ------------------------------------- | ------------------- | ---------------------- | ---------------- | ---------------------- |
| 0 | sp500  | units  | False             | sp500 does not Granger-causes units.  | 0.5784              | 0.5589                 | 0.5598           | 0.5784                 |
| 1 | units  | sp500  | False             | units does not Granger-causes sp500.  | 0.4875              | 0.4647                 | 0.4662           | 0.4875                 |
| 2 | walmex | units  | False             | walmex does not Granger-causes units. | 0.2435              | 0.2165                 | 0.221            | 0.2435                 |
| 3 | units  | walmex | False             | units does not Granger-causes walmex. | 0.4001              | 0.3748                 | 0.3772           | 0.4001                 |
| 4 | walmex | sp500  | False             | walmex does not Granger-causes sp500. | 0.3213              | 0.2945                 | 0.298            | 0.3213                 |
| 5 | sp500  | walmex | False             | sp500 does not Granger-causes walmex. | 0.2099              | 0.1833                 | 0.1883           | 0.2099                 |

Even though, the results from the Granger-causality tests suggested that it is not possible to reject the null hypothesis that the series does not Granger-causes each other, thus rendering the VAR model invalid, it was decided to go further with the model.

Later, a residual analysis was carried out to assess whether the difference between the actual and predicted values of the model is due to randomness or, in other words, that the residuals are normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

To do so, the residual analysis was performed in two steps [(Peixeiro, 2022)](#peixeiro): 
* **Quantile-quantile plot (Q-Q plot)**: To qualitatively assess whether the residuals from the model are normally distributed.
* **Ljung-Box test**: To test whether the residuals from the model are uncorrelated.

<p align="center">
	<img src="Images/fig_q-q_plot_for_standardized_residuals_for_units_series_from_var_model.png?raw=true" width=70% height=60%>
</p>

<p align="center">
	<img src="Images/fig_q-q_plot_for_standardized_residuals_for_sp500_series_from_var_model.png?raw=true" width=70% height=60%>
</p>

<p align="center">
	<img src="Images/fig_q-q_plot_for_standardized_residuals_for_walmex_series_from_var_model.png?raw=true" width=70% height=60%>
</p>

From the q-q plots above, it is clear that the standardized residuals for the series *units* and *SP&500* did not follow a normal distribution, which means that the VAR model was not able to capture some information from the data. Despite this result, the analysis continued below.

Then, the residuals were assessed by testing for uncorrelation with the **Ljung-Box test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: No\text{-}autocorrelation$$

$$H_{1}: Autocorrelation$$

$$\alpha = 0.95$$

For each of the lags from 1 to 10, the $p\text{-}values$ were above $0.05$ (please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a>). Thus, the null hypothesis cannot be rejected, meaning that no autocorrelation was found on the set of residuals from the VAR model. Thus, the residuals are independently distributed and the model could be used for forecasting.

Later, the predictions were plot against the historical net sales data to visually assess the performance of the VAR model.

<p align="center">
	<img src="Images/fig_predictions_from_var_model_vs_historical_time_series.png?raw=true" width=70% height=60%>
</p>

In view of the plots above, it can be seen that the VAR model was not able to yield accurate predictions. The only economic indicator whose predictions were more or less accurately was *units*. However, it is important to bear in mind that the COVID pandemic disrupted the historical trends not only in Mexico, but in the entire world. So, fairly, it was very dificult to expect that the VAR model could provide accurate predictions for the years 2021-2023 using the data from 2014-2020 for training.

Finally, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
units
RMSE: 41.684
MAE: 30.270
Coefficient of Determination: 0.750

sp500
RMSE: 741.698
MAE: 699.704
Coefficient of Determination: -9.352

walmex
RMSE: 7.841
MAE: 6.134
Coefficient of Determination: -7.682
```

<br>


#### **6.5.12 Vector Autoregressive Moving Average (VARMA) Model** <a class="anchor" id="varma_model"></a>

As shown above, the original dataset comprised $40$ historical data points about $7$ features (economic indicators) and $1$ response variable (net sales of WALMEX in millions of MXN). However, based on the results from the Johanssen's cointegration test, only the series *units*, *SP&500*, and *WALMEX* shared enough the same underlying stochastic trend to be used in a multivariate times series model.

In this sense, a second model for **multivariate times series prediction** was built using a **Vector Autoregressive Moving Average (VARMA) model** for forecasting the $3$ selected features. A VARMA model was selected as a generalization of the VAR model to include a moving average process [(Peixeiro, 2022)](#peixeiro).

The VARMA($p$, $q$) model describes the relationship among two or more time series and the impact that their past values  and past error terms have on each other [(Peixeiro, 2022)](#peixeiro). Similar to the VAR model, the VARMA($p$, $q$) model is a generalization of the ARMA model for multiple time series, where the order $p$ determines how many lagged values impact the present value of the different time series, and the order $q$ determines the number of past error terms that affect their present values.

Several orders of $p$ and $q$ were tested and the best performing model was selected according to the Akaike Information Criterion (AIC). Then, the Granger causality test was applied to determine whether past values of a time series are statistically significant in forecasting another time series. Then, a residual analysis was finally carried out to assess whether the residuals were normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

The function **VARMAX** from the library **statsmodels** in Python was used to build the **VARMA model** [(Peixeiro, 2022)](#peixeiro).

It was assumed that the features were cointegrated, i.e., the different features share an underlying stochastic trend, and that the linear combination of the different features is stationary [(Clower, 2020)](#clower). Moreover, it was assumed that the current values are linearly dependent on their past values and past error terms [(Peixeiro, 2022)](#peixeiro); and that the different time series were stationary, they were not a random walk, and seasonality effects were not relevant for modeling.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Later, the VARMA model was built as follows:

```python
def VARMA_model(p, q, series):
        """
        Fits a series of VARMA models based on the combinations of p and q provided and 
        returns the best performing model according to the Akaike Information Criterion (AIC).

        Parameters:
        p (list): Orders for the autoregressive process of the model.
        q (list): Orders for the moving average process of the model.
        series (pandas.dataframe): Data for the time series for fitting the model.

        Returns:
        model (statsmodels.varmax): A VARMA model object fitted according to the combination of p and q that minimizes 
        the Akaike Information Criterion (AIC).

        """
        
        # Obtaining the combinations of p and q
        order_list = list(product(p, q))

        # Creating emtpy lists to store results
        order_results = []
        aic_results = []

        # Fitting models
        for order in order_list:

                model = VARMAX(endog=series, order=(order[0], order[1])).fit()
                order_results.append(order)
                aic_results.append(model.aic)
        
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
        model = VARMAX(endog=series, order=(best_model[0], best_model[1])).fit()

        return model

p_list = [1,2,3,4]
q_list = [1,2,3,4]

varma_model = VARMA_model(p=p_list, q=q_list, series=X_train)

```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After that, the Granger causality test was applied to determine whether past values of a time series are statistically significant in forecasting another time series [(Peixeiro, 2022)](#peixeiro). This test is unidirectional, so it has to be applied twice for each series.

$$H_{0}: y_{2,t}\text{ does not Granger-cause }y_{1,t}$$

$$H_{1}: y_{2,t}\text{ Granger-cause }y_{1,t}$$

$$\alpha = 0.95$$

|   | y2     | y1     | Granger Causality | Test Interpretation                   | SSR F-test p-values | SSR Chi2-test p-values | LR-test p-values | Params F-test p-values |
| - | ------ | ------ | ----------------- | ------------------------------------- | ------------------- | ---------------------- | ---------------- | ---------------------- |
| 0 | sp500  | units  | False             | sp500 does not Granger-causes units.  | 0.5784              | 0.5589                 | 0.5598           | 0.5784                 |
| 1 | units  | sp500  | False             | units does not Granger-causes sp500.  | 0.4875              | 0.4647                 | 0.4662           | 0.4875                 |
| 2 | walmex | units  | False             | walmex does not Granger-causes units. | 0.2435              | 0.2165                 | 0.221            | 0.2435                 |
| 3 | units  | walmex | False             | units does not Granger-causes walmex. | 0.4001              | 0.3748                 | 0.3772           | 0.4001                 |
| 4 | walmex | sp500  | False             | walmex does not Granger-causes sp500. | 0.3213              | 0.2945                 | 0.298            | 0.3213                 |
| 5 | sp500  | walmex | False             | sp500 does not Granger-causes walmex. | 0.2099              | 0.1833                 | 0.1883           | 0.2099                 |

Even though, the results from the Granger-causality tests suggested that it is not possible to reject the null hypothesis that the series does not Granger-causes each other, thus rendering the VARMA model invalid, it was decided to go further with the model.

Later, a residual analysis was carried out to assess whether the difference between the actual and predicted values of the model is due to randomness or, in other words, that the residuals are normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

To do so, the residual analysis was performed in two steps [(Peixeiro, 2022)](#peixeiro): 
* **Quantile-quantile plot (Q-Q plot)**: To qualitatively assess whether the residuals from the model are normally distributed.
* **Ljung-Box test**: To test whether the residuals from the model are uncorrelated.

<p align="center">
	<img src="Images/fig_q-q_plot_for_standardized_residuals_for_unit_series_from_varma_model.png?raw=true" width=70% height=60%>
</p>

<p align="center">
	<img src="Images/fig_q-q_plot_for_standardized_residuals_for_sp&500_series_from_varma_model.png?raw=true" width=70% height=60%>
</p>

<p align="center">
	<img src="Images/fig_q-q_plot_for_standardized_residuals_for_walmex_series_from_varma_model.png?raw=true" width=70% height=60%>
</p>

From the plots above, it seems that the standardized residuals for the series have no trend and constant variance. The histograms also fairly resemble a normal distribution. Moreover, the Q-Q plots show almost straight lines. Finally, the correlograms show no significant coefficients. Thus, it is possible to conclude that the residuals were close to white noise.

Then, the residuals were assessed by testing for uncorrelation with the **Ljung-Box test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: No\text{-}autocorrelation$$

$$H_{1}: Autocorrelation$$

$$\alpha = 0.95$$

For each of the lags from 1 to 10, the $p\text{-}values$ were above $0.05$ (please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a>). Thus, the null hypothesis cannot be rejected, meaning that no autocorrelation was found on the set of residuals from the VARMA model. Thus, the residuals are independently distributed and the model could be used for forecasting.

Later, the predictions were plot against the historical net sales data to visually assess the performance of the VAR model.

<p align="center">
	<img src="Images/fig_predictions_from_varma_model_vs_historical_time_series.png?raw=true" width=70% height=60%>
</p>

In view of the plot above, it can be seen that the VARMA model was not able to yield accurate predictions. The only economic indicator whose predictions were accurately was *units*. However, it is important to bear in mind that the COVID pandemic disrupted the historical trends not only in Mexico, but in the entire world. So, fairly, it was very dificult to expect that the VARMA model could provide accurate predictions for the years 2021-2023 using the data from 2014-2020 for training. Overall, it seems that the performance of the VARMA model was better than that of the VAR model.

Finally, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
units
RMSE: 29.895
MAE: 26.242
Coefficient of Determination: 0.871

sp500
RMSE: 806.357
MAE: 763.090
Coefficient of Determination: -11.235

walmex
RMSE: 8.352
MAE: 7.041
Coefficient of Determination: -8.852
```

<br>

#### **6.5.13 Vector Autoregressive Integrated Moving Average (VARIMA) Model** <a class="anchor" id="varima_model"></a>

As shown above, the original dataset comprised $40$ historical data points about $7$ features (economic indicators) and $1$ response variable (net sales of WALMEX in millions of MXN). However, based on the results from the Johanssen's cointegration test, only the series *units*, *SP&500*, and *WALMEX* shared enough the same underlying stochastic trend to be used in a multivariate times series model.

In this sense, a third model for **multivariate times series prediction** was built using a **Vector Autoregressive Integrated Moving Average (VARIMA) model** for forecasting the $3$ selected features. A VARIMA model was selected as a generalization of the VARMA model to include an integration process [(Peixeiro, 2022)](#peixeiro) for forecasting non-stationary time series, meaning that the time series have a trend, and/or their variance is not constant.

The VARIMA($p$, $d$, $q$) model describes the relationship among two or more non-stationary time series and the impact that their past values  and past error terms have on each other [(Peixeiro, 2022)](#peixeiro). Similar to the VARMA model, the VARIMA($p$, $d$, $q$) model is a generalization of the ARMA model for multiple non-stationary time series, where the order $p$ determines how many lagged values impact the present value of the different time series, the order $d$ indicates the number of times the time series have been differenced to become stationary, and the order $q$ determines the number of past error terms that affect their present values.

The function **VARIMA** from the library **Darts** in Python was used to build the **VARIMA model** [(Herzen et al., 2022)](#herzen).

As the implementation of the VARIMA model in Darts does not supports the prediction of likelihood parameters, the Akaike Information Criterion (AIC) could not be computed for a different set of $p$ and $q$ values. So, the same orders of $p=1$ and $q=1$ obtained during the optimization of the VARMA model were used for VARIMA($1$,$1$).

Then, the Granger causality test was applied to determine whether past values of a time series are statistically significant in forecasting another time series. Then, a residual analysis was finally carried out to assess whether the residuals were normally and independently distributed [(Peixeiro, 2022)](#peixeiro).

It was assumed that the features are cointegrated, i.e., the different features share an underlying stochastic trend, and that the linear combination of the different features is stationary [(Clower, 2020)](#clower). 
Moreover, it was assumed that the current values are linearly dependent on their past values and past error terms [(Peixeiro, 2022)](#peixeiro), the time series are not a random walk, and seasonality effects were not relevant for modeling.

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Later, the VARIMA model was built as follows:

```python
def VARIMA_model(p, d, q, series):
        """
        Fits and optimize a VARIMA model based on the Akaike Information Criterion (AIC), given a set of p, d, and q values.

        Parameters:
        p (int): Order of the autoregressive process in the model.
        d (int): Integration order in the model.
        p (int): Order of the moving average process in the model.
        series (pandas.dataframe): Data for the time series for fitting the model.

        Returns:
        model (darts.varima): A VARIMA object fitted according to the combination of p, d and q that minimizes 
        the Akaike Information Criterion (AIC).


        """
        # Converting pandas.dataframe to Darts.TimeSeries
        series = TimeSeries.from_dataframe(series)


        model = VARIMA(p=p, d=d, q=q, trend="n") # No trend for models with integration
        model.fit(series)

        return model

varima_model = VARIMA_model(p=1, d=1, q=1, series=X_train)
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After that, the Granger causality test was applied to determine whether past values of a time series are statistically significant in forecasting another time series [(Peixeiro, 2022)](#peixeiro). This test is unidirectional, so it has to be applied twice for each series.

$$H_{0}: y_{2,t}\text{ does not Granger-cause }y_{1,t}$$

$$H_{1}: y_{2,t}\text{ Granger-cause }y_{1,t}$$

$$\alpha = 0.95$$
|   | y2     | y1     | Granger Causality | Test Interpretation                   | SSR F-test p-values | SSR Chi2-test p-values | LR-test p-values | Params F-test p-values |
| - | ------ | ------ | ----------------- | ------------------------------------- | ------------------- | ---------------------- | ---------------- | ---------------------- |
| 0 | sp500  | units  | False             | sp500 does not Granger-causes units.  | 0.1045              | 0.083                  | 0.0889           | 0.1045                 |
| 1 | units  | sp500  | False             | units does not Granger-causes sp500.  | 0.1382              | 0.1145                 | 0.1202           | 0.1382                 |
| 2 | walmex | units  | False             | walmex does not Granger-causes units. | 0.2046              | 0.1787                 | 0.1837           | 0.2046                 |
| 3 | units  | walmex | False             | units does not Granger-causes walmex. | 0.2054              | 0.1795                 | 0.1844           | 0.2054                 |
| 4 | walmex | sp500  | False             | walmex does not Granger-causes sp500. | 0.3576              | 0.332                  | 0.3349           | 0.3576                 |
| 5 | sp500  | walmex | True              | sp500 Granger-causes walmex.          | 0.0021              | 0.0006                 | 0.0013           | 0.0021                 |

Even though, the results from the Granger-causality tests suggested that it is not possible to reject the null hypothesis that the series does not Granger-causes each other, thus rendering the VARIMA model invalid, it was decided to go further with the model.

However, as the implementation of the VARIMA model in Darts only yielded the residuals for the last two observations, the residual analysis couldn't be performed properly.

Thus, the residuals were only assessed by testing for uncorrelation with the **Ljung-Box test** [(Peixeiro, 2022)](#peixeiro), using a 95% confidence level: 

$$H_{0}: No\text{-}autocorrelation$$

$$H_{1}: Autocorrelation$$

$$\alpha = 0.95$$

For the lag 1, the $p\text{-}values$ were above $0.05$ (please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a>). Thus, the null hypothesis cannot be rejected, meaning that no autocorrelation was found on the set of residuals from the VARMA model. Thus, in theory, the residuals are independently distributed and the model could be used for forecasting.

Of course, strictly speaking, as the residual analysis could not be properly performed, it is not possible to know whether the model is valid or not. However, for the purposes of the present study, the model was still used for generating predictions.

<p align="center">
	<img src="Images/fig_predictions_from_varima_model_vs_historical_time_series.png?raw=true" width=70% height=60%>
</p>

In view of the plots above, it can be seen that the VARIMA model was not able to yield accurate predictions. The only economic indicator whose predictions were more less accurate was *units*. However, it is important to bear in mind that the COVID pandemic disrupted the historical trends not only in Mexico, but in the entire world. So, fairly, it was very dificult to expect that the VARMA model could provide accurate predictions for the years 2021-2023 using the data from 2014-2020 for training. Notwithstanding with the above, it seems that the performance of the VARIMA model was slightly worse than that of the VARMA model.

Finally, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
units
RMSE: 57.604
MAE: 42.577
Coefficient of Determination: 0.523

sp500
RMSE: 939.278
MAE: 898.338
Coefficient of Determination: -15.601

walmex
RMSE: 14.040
MAE: 12.263
Coefficient of Determination: -26.838
```

<br>

#### **6.5.14 Random Forest (RF) Regression Model** <a class="anchor" id="regression_model_rf"></a>

As it was desired to predict a target numeric value, a **regression model** was built for predicting the net sales of WALMEX based on the forecast for the selected $3$ features from the multivariate time series models: *units*, *S&P500*, and *WALMEX* stock value.

It was decided to use a **Random Forest (RF)** approach, as this algorithm provides better models than decision trees and allows to easily identify the most important features in a model [(Géron, 2019)](#geron). Moreover, it is not necessary to neither comply with the assumptions of the linear regression nor to perform feature scaling.

The main assumption is that the net sales of WALMEX can be predicted as a function of the economic indicators of Mexico and USA. Furthermore, it was assumed that the problem was not linearly separable, thus, a linear regression approach was discarded. 

The dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Later, the RF model was built as follows:

```python
rf_model = RandomForestRegressor(n_estimators = 500, random_state = 0)
rf_model.fit(X_train, y_train)
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

After modeling, it was observed that the feature **Units** was the most important in the model:

<p align="center">
	<img src="Images/fig_feature_importances_in_random_forest_regression_model.png?raw=true" width=70% height=60%>
</p>

Later, the predictions were plot against the historical net sales data to visually assess the performance of the RF model.

<p align="center">
	<img src="Images/fig_predictions_from_rf_model_vs._walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

From the plot above, it can be seen that the predictions are very close of the historical time series. Thus, the Random Forest regression model was able to capture the most important features for predicting the net sales of WALMEX. 

Finally, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
net_sales
RMSE: 16256.433
MAE: 12380.448
Coefficient of Determination: 0.516
```

Several number of trees were tested, and the following results were obtained:

Trees | RMSE | MAE | $r^{2}$
:---:| :---:| :---: | :---:
50 | 16465.794 | 13240.795 | 0.504
100 | 16464.343 | 12533.214 | 0.513
500 | 16256.433 | 12380.448 | 0.516

Thus, based on their performance, the model selected was the RF model with 500 trees.

<br>

#### **6.5.15 Support Vector Regression (SVR) Model** <a class="anchor" id="regression_model_svm"></a>

Likewise, another **regression model** was built for predicting the net sales of WALMEX based on the forecast for the selected $3$ features from the multivariate time series models: *units*, *S&P500*, and *WALMEX stock value*.

A **Support Vector Regression (SVR)** approach was selected, as this algorithm is less restrictive in their assumptions in comparison to a linear regression, it's more robust to outliers, and does not strictly assume a linear relationship between the independent and dependent variables even if using a linear kernel  [(Géron, 2019)](#geron).

The main assumption is that the net sales of WALMEX can be predicted as a function of the economic indicators of Mexico and USA. Also, a strict linear relationship between the variables was not assummed, neither the requirements of normally distribution of the residuals, no multicollinearity, and homoscedasticity.

Firstly, according to the cointegration test, the data was sliced to select only the features *Units*, *SP&500*, and *WALMEX* stock value. Then, the dataset was split into a training and a testing sets, allocating 80% and 20% of the data, respectively.

Then, data was scaled using the function *StandardScaler*. And the SVR model was built using the *SVR* class from the **scikit-learn** library:

Later, the RF model was built as follows:

```python
svr_model = SVR(kernel='linear', C=1000.0, epsilon=0.1).fit(X_train_sc, y_train)
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/SalesForecasting/blob/35a592125ea91b0df1a0b61feb57d199478443e5/SalesForecasting.ipynb">notebook</a> for the full details.

Later, the predictions were plot against the historical net sales data to visually assess the performance of the SVR model.

<p align="center">
	<img src="Images/fig_predictions_from_svr_model_vs._walmex_historical_net_sales.png?raw=true" width=70% height=60%>
</p>

From the plot above, it can be seen that the predictions are close of the historical time series. Thus, the SVR model was able to capture the most important features for predicting the net sales of WALMEX. 

Finally, the **RMSE**, **MAE**, and $\bf{r^{2}}$ score were calculated as follows:

```bash
net_sales
RMSE: 14535.582
MAE: 10676.184
Coefficient of Determination: 0.402
```

Several values for the parameters `C` were tested, and the following results were obtained:

C | Epsilon | RMSE | MAE | $r^{2}$
:---:|:---:| :---:| :---: | :---:
1.0 | 0.1 | 21540.030 | 15545.984 | -176851.575
100.0 | 0.1 | 17257.012 | 13353.768 | -10.351
1000.0 | 0.1 | 14535.582 | 10676.184 | 0.402
5000.0 | 0.1 | 16414.855 | 11640.130 | 0.470
10000.0 | 0.1 | 16223.993 | 11624.612 | 0.475
100000.0 | 0.1 | 18720.282 | 14280.358 | 0.465

Thus, based on their performance, the model selected was the SVR model with `C=1000.0`.

<br>

### **6.6 Evaluation** <a class="anchor" id="evaluation"></a>

Pending...
