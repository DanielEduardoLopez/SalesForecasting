# Sales Prediction for Walmart in Mexico
# Author: Daniel Eduardo López
# Github: https://github.com/DanielEduardoLopez
# LinkedIn: https://www.linkedin.com/in/daniel-eduardo-lopez
# Date: 2024/08/22

"""
Project's Brief Description:
Time Series Analyses for forecasting Walmart net sales over the next 10 years in Mexico.
"""

# Libraries importation
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🇲🇽",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/daniel-eduardo-lopez',
        'Report a bug': "https://www.linkedin.com/in/daniel-eduardo-lopez",
        'About': "Time Series Analysis for forecasting Walmart net sales over the next 10 years in Mexico."
    }
)

# Variables
forecast_str = 'Forecast'
time_series_str = 'Historical'

# Functions
def get_hist_data():
    link = 'https://raw.githubusercontent.com/DanielEduardoLopez/SalesForecasting/main/dataset_processed.csv'
    df = pd.read_csv(link)\
        .drop(columns=['units','gdp','walmex','sp500','ipc','exchange_rates','interest_rates']).set_index('date')
    return df

def get_model(time_series):
    model = SARIMAX(endog=time_series, 
                    order = (1, 1, 1),
                    seasonal_order=(1, 1, 1, 4),
                    ).fit(disp=False)
    return model

def get_forecast(model, periods):
    preds = model.forecast(steps=periods).rename(forecast_str) 
    preds = pd.DataFrame(preds)
    return preds

def extend_model(model, new_observations, historical):
    last_date = str(historical.index[-1])
    periods = len(new_observations)
    new_index = pd.period_range(start=last_date, periods=periods, freq='Q')
    new_observations = pd.Series(new_observations, index=new_index, name=forecast_str)
    model = model.extend(endog=new_observations)
    return model


def plot_chart(historical, forecasts):
    data = pd.concat([historical, forecasts], axis=1)
    data = data.rename(columns={'net_sales': time_series_str})
    fig = px.line(data, 
                  x=data.index, 
                  y=[time_series_str, forecast_str], 
                  title='WALMEX Net Sales Forecast',
                  color_discrete_map={
                 time_series_str: "lightblue",
                 forecast_str: "red"},
                  )    
    #fig.add_scatter(forecasts, x=forecasts.index, y=forecasts.net_sales, mode='lines', line_color='red')    
    return fig



# Disabling fullscreen view for images in app
hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''
st.markdown(hide_img_fs, unsafe_allow_html=True)

# Disabling displayModeBar in Plotly Charts
config = {'displayModeBar': False}

# App

st.title("Sales Forecasting for Walmart in Mexico")

# Defining page to display
if "app_page" not in st.session_state:
    page = "Homepage"
else:
    page = st.session_state["app_page"]
    

# Side bar
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("**About me and this project:**")
col1, col2 = st.sidebar.columns([0.3, 0.7], gap="small")
with col1:
    st.markdown("")
    st.image("Picture.jpg")

with col2:
    st.markdown("Hi! I'm Eduardo, senior data analyst at AstraZeneca. I love learning, and this is a personal project to play with time series analysis. :computer:")

st.sidebar.markdown("")

# Homepage
if page == "Homepage":

    # Header information
    col1, col2 = st.columns([0.1, 0.9], gap="small")

    with col1:
        st.image("Picture.jpg")

    with col2:
        st.markdown('##### :blue[Daniel Eduardo López]')
        html_contact = '<a href="https://www.linkedin.com/in/daniel-eduardo-lopez">LinkedIn</a> | <a href="https://github.com/DanielEduardoLopez">GitHub</a>'
        st.caption(html_contact, unsafe_allow_html=True)

    st.markdown("August 26, 2024")
    st.caption("5 min read")
    st.image("sales-figures-1473495.jpg")
    html_picture = '<p style="font-size: 12px" align="center">Image Credit: <a href="https://www.freeimages.com/es/photo/sales-figures-1473495/">wagg66</a> from <a href="https://www.freeimages.com/">FreeImages</a>.</p>'
    st.caption(html_picture, unsafe_allow_html=True)

    # Introduction    
    st.header(":blue[Welcome!]")
    st.markdown("Walmart of Mexico (or WALMEX) is one of the most important retail companies within the region, with 3,903 stores in Mexico and Central America, an equity of 199,086,037 MXN, and a yearly revenue of 880,121,761 MXN, according to the figures from December 2023. According to WALMEX last financial report, its goal is to double its sales in a period of 10 years (Wal-Mart de México S.A.B. de C.V., 2024).")
    st.markdown('Time series are "a set of data points ordered in time" (Pexeiro, 2022), which can be analyzed to calculate forecasts and get valuable insights (Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023).')
    st.markdown("Univariate time series is the most used approach when analyzing time series (Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023), by means of models such as Moving Average (MA), Autoregressive Moving Average (ARMA), Autoregressive Integrated Moving Average (ARIMA), or Simple Exponential Smoothing; which solely depend on the time and the variable under study.")
    st.markdown("On the other hand, it is also possible to forecast time series using regression-based modeling, in which other variables or features are used to predict the response variable (Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023). This approach could have the advantage of quantifying the impact of the external economic indicators in the performance of an organization.")
    st.markdown("In the case of Mexico, it is possible to collect public data from different government offices such as INEGI or BANXICO, or from international sources such as the S&P500, and to assess how they correlate to revenue.")
    st.markdown("In this context, it is desirable to explore both approaches to predict WALMEX net sales over the next years. Thus, the purpose of the present project was to forecast WALMEX net sales and, then, use that information to predict whether WALMEX will be able to achieve its long-term goal of doubling its sales within the next ten years.")
    st.markdown("To do so, several univariate, multivariate time series and regression models were built using Python 3 and its libraries Statsmodels, Prophet, Darts, and Scikit-learn:")
    st.markdown("- Moving Average (MA) model")
    st.markdown("- Autoregressive (AR) model")
    st.markdown("- A series of Autoregressive (AR) models with Additive Decomposition")
    st.markdown("- Autoregressive Moving Average (ARMA) model")
    st.markdown("- Autoregressive Integrated Moving Average (ARIMA) model")
    st.markdown("- Seasonal Autoregressive Integrated Moving Average (SARIMA) model")
    st.markdown("- Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) model")
    st.markdown("- Simple Exponential Smoothing (SES) model")
    st.markdown("- Holt-Winters (HW) model")
    st.markdown("- Prophet Univariate Time Series Modeling")
    st.markdown("- Vector Autoregressive (VAR) model")
    st.markdown("- Vector Autoregressive Moving Average (VARMA) model")
    st.markdown("- Vector Autoregressive Integrated Moving Average (VARIMA) model")
    st.markdown("- Random Forests (RF) model")
    st.markdown("- Support Vector Regression (SVR) model")
    st.markdown("All the models were fit using a training set with 80% of the data, and assessed using a testing set with the remaining 20% of the data. The scores **Root Mean Squared Error (RMSE)**, the **Mean Absolute Error (MAE)**, and **Coefficient of Determination** $(r^{2})$ were used for model assessment.")
    url_repository = "https://github.com/DanielEduardoLopez/SalesForecasting"
    st.write("All the technical details can be found at [GitHub](%s)." % url_repository)
    st.markdown("")

    # Model brief description
    st.subheader(":blue[Best Model]")
    st.markdown("Based on all the models fitted, the :blue[**$\t{SARIMA}(1,1,1)(1,1,1)_{4}$ model**] exhibited the best performance, achieving about **2675.576** of **RMSE**, about **2372.531** of **MAE**, and a $r^{2}$ of about **0.983**.")
    st.markdown("Thus, the resulting model had a good performance overall, outmatching the multivariate/regression approaches.")
    st.markdown("According to the results from this study, **Walmart of Mexico (WALMEX) will meet its goal of doubling its sales from 211,436 mdp to 424,050 mdp in the third quarter of 2033**.")
    st.markdown('Please go the :orange[**_Forecast_**] page to play with the model. :blush:')

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    with bcol2:
        if st.button('Go to Forecast Page'):
            st.session_state["app_page"] = "Forecast"            

    st.markdown("")

    # References
    st.subheader(":blue[References]")
    st.markdown("* **Kulkarni, A. R., Shivananda, A., Kulkarni, A., & Krishnan, V. A. (2023)**. *Time Series Algorithms Recipes: Implement Machine Learning and Deep Learning Techniques with Python*. Apress Media, LLC. https://doi.org/10.1007/978-1-4842-8978-5")
    st.markdown("* **Peixeiro, M. (2022)**. *Time Series Forecasting in Python*. Manning Publications Co.")
    st.markdown("* **Wal-Mart de México S.A.B. de C.V. (2024)**. *Información Financiera Trimestral 4T*. https://files.walmex.mx/upload/files/2023/ES/Trimestral/4T23/WALMEX_4T23_BMV.pdf")

# Predict Page
elif page == "Forecast":

    # Brief description of the app
    url_repository = "https://github.com/DanielEduardoLopez/SalesForecasting"
    st.write('Uses a $\t{SARIMA}(1,1,1)(1,1,1)_{4}$ model trained on the historical net sales data of WALMEX (Wal-Mart de México S.A.B. de C.V., 2024) from 2014Q1 to 2023Q4 to forecast net sales over the next 10 years. Check out the code [here](%s) and more details at the <h style="color:orange;"><i><b>Homepage</b></i></h>. Please do not take the predictions from this app so seriously. 😉' % url_repository, unsafe_allow_html=True)

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    with bcol2:
        if st.button('Go to Homepage'):
            st.session_state["app_page"] = "Homepage"
    
    # Initial Forecast
    historical = get_hist_data()
    model = get_model(historical)
    predictions = get_forecast(model, 10*4)
    line_chart = plot_chart(historical, predictions)

    st.markdown("")
    st.subheader(":blue[Forecast]")
    st.plotly_chart(line_chart, config=config, use_container_width=True)

    # Input data section
    st.markdown("")
    st.subheader(":blue[New Forecast]")
    st.markdown("The model has been trained with data from 2014Q1 to 2023Q4, please input the net sales values for the next periods:")
    

    st.markdown("")
    st.markdown("")

    # Results section

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    st.session_state["flag_charts"] = 1

    with bcol2:
        if st.button('Predict Probability :nerd_face:'):
            # Get input array from user's input
            
            # Model
            #model = get_model()

            # Prediction
            #preds = get_forecast(model, periods)
            st.success("Success! Please scroll down...")
            st.session_state["flag_charts"] = 2


    if st.session_state["flag_charts"] == 1:
        pass

    elif st.session_state["flag_charts"] == 2:

        # Charts sections
        st.subheader(":blue[Prediction Results]")
        st.markdown("According to the provided socioeconomic and demographic data, the probability of suffering different crimes in Mexico is as follows: :bar_chart:")
        st.markdown("")
        st.markdown("")

        
        
        

        # Pie chart
        bcol1, bcol2, bcol3 = st.columns([0.1, 0.8, 0.1])
        with bcol2:
            st.markdown('<p style="font-size: 22px" align="center"><b>Overall Probability of Suffering Any Crime in Mexico</b></p>', unsafe_allow_html=True)
            st.plotly_chart(pie_chart, config=config, use_container_width=True)
        st.markdown("Don't freak out if you get 100% or so. Everyone is exposed to suffer a crime in Mexico in their lifetime. Petty crimes most likely.")
        st.markdown("")
        st.markdown("")

        # Bar chart
        bcol1, bcol2, bcol3 = st.columns([0.1, 0.8, 0.1])
        with bcol2:
            st.markdown(
                '<p style="font-size: 22px" align="center"><b>Probability of Suffering Different Crimes in Mexico</b></p>',
                unsafe_allow_html=True)
            st.plotly_chart(bar_chart, config=config, use_container_width=True)


