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

Pending...

