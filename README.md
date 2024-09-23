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

### **6.4 Modeling** <a class="anchor" id="modeling"></a>

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

#### **6.4.1 Moving Average (MA) Model** <a class="anchor" id="ma_model"></a>

A **Moving Average (MA) model** was built as a baseline model. As suggested by the name of this technique, the MA model calculates the moving average changes over time for a certain variable using a specified window lenght [(Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023)](#kulkarni). This technique has the advantage of removing random fluctuations and smoothen the data. The MA model is denoted as **MA($q$)**, where $q$ is the number of past error terms that affects the present value of the time series [(Peixeiro, 2022)](#peixeiro).

The model was built using the library **pandas** in Python. 

It was assumed that the current values are linearly dependent on the mean of the series, the current and past error terms; the errors are normally distributed [(Peixeiro, 2022)](#peixeiro). On the other hand, from the ACF plot in the [EDA section](#acf), it was assumed that a window of 5 was enough to capture the overall trend of the data, as the ACF plot showed that lag 5 was the last significant coefficient.

The dataset was split into a training and a testing sets, allocating $80\%$ and $20\%$ of the data, respectively. 

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
