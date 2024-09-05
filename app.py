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
new_observations_str = 'New Observations'
historical_color = 'blue'
pred_color = 'red' #'#C70039'
new_observations_color = 'green'

# Functions
def get_hist_data():
    link = 'https://raw.githubusercontent.com/DanielEduardoLopez/SalesForecasting/main/dataset_processed.csv'
    df = pd.read_csv(link)\
        .drop(columns=['units','gdp','walmex','sp500','ipc','exchange_rates','interest_rates']).set_index('date').sort_index(ascending=True)
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

def process_df(df):
    df['date'] = pd.to_datetime(df.Date)
    df = df.set_index('date').drop(columns=['Date', 'Quarter']).rename(columns={'Net_Sales':'net_sales'})
    return df 

def extend_model(model, new_observations):        
    model = model.extend(endog=new_observations.values)
    return model


def plot_chart(historical, forecasts):

    # Adjusting plots continuity    
    last_obs = historical.tail(1).rename(columns={'net_sales':forecast_str})
    forecasts = pd.concat([forecasts, last_obs], axis=0)
    forecasts.index = pd.to_datetime(forecasts.index)
    forecasts = forecasts.sort_index()
    
    # Concatenating historical and forecasts into a single dataframe
    data = pd.concat([historical, forecasts], axis=1)
    data = data.rename(columns={'net_sales': time_series_str})

    # Plot
    fig = px.line(data, 
                  x=data.index, 
                  y=[time_series_str, forecast_str], 
                  title=None,
                  template="plotly",
                  color_discrete_map={
                 time_series_str: historical_color,
                 forecast_str: pred_color},
                  )    
    fig.update_layout(
                        xaxis_title="Date", 
                        yaxis_title="Net Sales (Millions of MXN)",                                                
                        plot_bgcolor='rgba(137,196,244,0.15)',
                        legend_title_text='',
                        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_xaxes(title_font=dict(size=17, color='black'),
                     showgrid=True, gridwidth=1, gridcolor='white')
    fig.update_yaxes(title_font=dict(size=17, color='black'),
                     showgrid=True, gridwidth=1, gridcolor='white')
    
    return fig

def plot_chart_retrained_model(historical, forecasts, new_observations):

    historical = historical.rename(columns={'net_sales': time_series_str})
    new_observations = new_observations.rename(columns={'net_sales': new_observations_str})
    forecasts = forecasts.rename(columns={'net_sales': forecast_str})

    # Adjusting plots continuity    
    last_obs = historical.tail(1).rename(columns={time_series_str: new_observations_str})
    new_observations = pd.concat([new_observations, last_obs], axis=0)
    new_observations.index = pd.to_datetime(new_observations.index)
    new_observations = new_observations.sort_index()

    last_obs = new_observations.tail(1).rename(columns={new_observations_str: forecast_str})
    forecasts = pd.concat([forecasts, last_obs], axis=0)
    forecasts.index = pd.to_datetime(forecasts.index)
    forecasts = forecasts.sort_index()
    
    # Concatenating historical and forecasts into a single dataframe
    data = pd.concat([historical, new_observations, forecasts], axis=1)    

    # Plot
    fig = px.line(data, 
                  x=data.index, 
                  y=[time_series_str, new_observations_str, forecast_str], 
                  title=None,
                  template="plotly",
                  color_discrete_map={
                 time_series_str: historical_color,
                 new_observations_str: new_observations_color,
                 forecast_str: pred_color},
                  )    
    fig.update_layout(
                        xaxis_title="Date", 
                        yaxis_title="Net Sales (Millions of MXN)",                                                
                        plot_bgcolor='rgba(137,196,244,0.15)',
                        legend_title_text='',
                        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_xaxes(title_font=dict(size=17, color='black'),
                     showgrid=True, gridwidth=1, gridcolor='white')
    fig.update_yaxes(title_font=dict(size=17, color='black'),
                     showgrid=True, gridwidth=1, gridcolor='white')
    
    return fig

def get_conclusion(time_series, predictions, last_date='2023-07-01'):

    predictions = predictions.rename(columns={forecast_str:'net_sales'})
    time_series = time_series.rename(columns={time_series_str:'net_sales'})
    
    last_sales = time_series.loc[last_date].values
    sales_doubled = predictions.loc[predictions.iloc[:,0].values >= 2*last_sales,:]
    
    try:
            # Converting to datetime
            date = pd.to_datetime(str(sales_doubled.index[0]))

            # Converting to year-quarter format
            year_quarter = date.to_period('Q')

            # Printing result
            return f'WALMEX will double its net sales from 2023Q3 in **{str(year_quarter)}**, with a forecasted figure of about **${float(sales_doubled.values[0]):,.2f} (mdp)**.'
        
    except:
            return 'It seems that WALMEX will not be able to double its net sales within the selected time period.'


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
    st.markdown("Hi! I'm Eduardo, senior data analyst at AstraZeneca. I love learning, and this is a personal project to play with time series analysis. :chart_with_upwards_trend: :computer:")

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
    st.markdown("**Walmart of Mexico (or WALMEX)** is one of the most important retail companies within the region, with **3,903 stores** in Mexico and Central America, an equity of **199,086,037 MXN**, and a yearly revenue of **880,121,761 MXN**, according to the figures from December 2023. According to WALMEX last financial report, its goal is **to double its sales in a period of 10 years** (Wal-Mart de México S.A.B. de C.V., 2024).")
    st.markdown('**Time series** are "a set of data points ordered in time" (Pexeiro, 2022), which can be analyzed to calculate forecasts and get valuable insights (Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023).')
    st.markdown("**Univariate time series** is the most used approach when analyzing time series (Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023), by means of models such as Moving Average (MA), Autoregressive Moving Average (ARMA), Autoregressive Integrated Moving Average (ARIMA), or Simple Exponential Smoothing; which solely depend on the time and the variable under study.")
    st.markdown("On the other hand, it is also possible to forecast time series using **regression-based modeling**, in which other variables or features are used to predict the response variable (Kulkarni, Shivananda, Kulkarni, & Krishnan, 2023). This approach could have the advantage of quantifying the impact of the external economic indicators in the performance of an organization.")
    st.markdown("In the case of Mexico, it is possible to collect public data from different government offices such as INEGI or BANXICO, or from international sources such as the S&P500, and to assess **how they correlate to revenue.**")
    st.markdown("In this context, it is desirable to explore both approaches **to predict WALMEX net sales over the next years**. Thus, the purpose of the present project was to forecast WALMEX net sales and, then, use that information to predict whether WALMEX will be able to achieve its long-term goal of **doubling its sales within the next ten years.**")
    st.markdown("To do so, several **univariate, multivariate time series and regression models** were built using Python 3 and its libraries Statsmodels, Prophet, Darts, and Scikit-learn:")
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
    st.markdown('Please go the :blue[**_Forecast_**] page to play with the model. :blush:')

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    with bcol2:
        if st.button('Go to Forecast Page'):
            st.session_state["app_page"] = "Forecast"
            st.experimental_rerun        

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
    st.write('Uses a $\t{SARIMA}(1,1,1)(1,1,1)_{4}$ model trained on the historical net sales data of WALMEX (Wal-Mart de México S.A.B. de C.V., 2024) from **2014Q1** to **2023Q4** to forecast net sales over **the next 10 years**. Check out the code [here](%s) and more details at the :blue[**_Homepage_**].' % url_repository, unsafe_allow_html=True)

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    with bcol2:
        if st.button('Go to Homepage'):
            st.session_state["app_page"] = "Homepage"
            st.experimental_rerun
    
    # Initial Forecast
    historical = get_hist_data()
    model = get_model(historical)
    predictions = get_forecast(model, 10*4)
    line_chart = plot_chart(historical, predictions)
    
    st.markdown("")
    st.subheader(":blue[Forecast]")
    st.markdown("The net sales forecast for Walmart in Mexico over the next 10 years is as follows:")
    st.markdown('<p style="font-size: 18px" align="center"><b>Net Sales Forecast for Walmart in Mexico</b></p>', unsafe_allow_html=True)
    st.plotly_chart(line_chart, config=config, use_container_width=True)
    st.markdown("According to plot above, Walmart of Mexico (WALMEX) will meet its goal of **doubling its sales from 211,436 mdp in 2023Q3 to 424,050 mdp in 2033Q3**.")

    # Input data section
    st.markdown("")
    st.subheader(":blue[New Forecast]")
    st.markdown("The model has been trained with data from 2014Q1 to 2023Q4. Please select the number of net sales values you would like to input to retrain the model.")
    
    

    data_points = st.slider("Select number of observations to add to model:", min_value=1, max_value=20, value=4)

    st.markdown('Now, edit the net sales values at the **"Net_Sales"** column on the following table:')
    
    df = (predictions.reset_index()
          .rename(columns={'index':'Date'})          
          .assign(Quarter= lambda df: df.Date.dt.to_period('Q').astype(str))
          .assign(Date= lambda df: df.Date.dt.date)
          .assign(Net_Sales = lambda df: round(df.Forecast,0))
          .drop(columns=['Forecast'])
          .iloc[:data_points]   
          )   

    edited_df = st.experimental_data_editor(df, disabled=False)
    new_observations = process_df(edited_df) 

    st.markdown('Finally, select the number of periods (quarters) you would like to forecast:')

    periods = st.slider("Select number of periods to forecast:", min_value=4, max_value=40, value=20)  
    
    # Results section

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    st.session_state["flag_charts"] = 1

    js = '''
        <script>
            var body = window.parent.document.querySelector(".main");
            console.log(body);
            body.scrollTop = 0;
        </script>
        '''

    st.components.v1.html(js)

    with bcol2:
        if st.button('Forecast :nerd_face:'):
                        
            # Model
            new_time_series = pd.concat([historical,new_observations], axis=0)
            new_model = get_model(new_time_series)
            #new_model = extend_model(model, new_observations)

            # Prediction
            new_preds = get_forecast(new_model, periods)
            st.success("Success! Please scroll down...")
            st.session_state["flag_charts"] = 2


    if st.session_state["flag_charts"] == 1:
        
        pass

    elif st.session_state["flag_charts"] == 2:

        # Charts sections
        st.markdown("According to the provided input, the net sales forecast is as follows: :chart_with_upwards_trend:")
        st.markdown("")
        st.markdown('<p style="font-size: 18px" align="center"><b>New Net Sales Forecast for Walmart in Mexico</b></p>', unsafe_allow_html=True)
        line_chart_retrained_model = plot_chart_retrained_model(historical, new_preds, new_observations)
        st.plotly_chart(line_chart_retrained_model, config=config, use_container_width=True)
        st.markdown("")    
        conclusion = get_conclusion(historical, new_preds)
        st.markdown(conclusion) 

        st.markdown('Thanks for giving a try!')

    

