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
import pickle
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



# Functions




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
    page = "Forecast"
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
    st.markdown("Hi! I'm Eduardo, engineer specialized in data science. The thing I enjoy the most is working with data and people.  ")

st.sidebar.markdown("I love learning, and this is a personal project to play time series analysis. :computer:")
st.sidebar.markdown("Please don't take the predictions from this app so seriously.")

# Homepage
if page == "Homepage":

    # Header information
    col1, col2 = st.columns([0.1, 0.9], gap="small")

    with col1:
        st.image("Picture.jpg")

    with col2:
        st.markdown('##### :blue[Daniel Eduardo López]')
        html_contact = '<a href="https://github.com/DanielEduardoLopez">GitHub</a> | <a href="https://www.linkedin.com/in/daniel-eduardo-lopez">LinkedIn</a>'
        st.caption(html_contact, unsafe_allow_html=True)

    st.markdown("August 22, 2024")
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
    st.markdown("In this context, it is desirable to explore both approaches to predict WALMEX net sales over the next years. Thus, the purpose of the present project is to forecast WALMEX net sales and, then, use that information to predict whether WALMEX will be able to achieve its long-term goal of doubling its sales within the next ten years.")
    st.markdown("")

    # Model brief description
    st.subheader(":blue[Model]")
    st.markdown("Based on all the models fitted, :blue[**a $\text{SARIMA}(1,1,1)(1,1,1)_{4}$ model**] was built and trained using Python and statsmodels, achieving about **2675.576** of **RMSE**, about *2372.531** of **MAE**, and a **R^2** of about **0.983**.")
    url_repository = "https://github.com/DanielEduardoLopez/SalesForecasting"
    st.write("All the technical details can be found at [GitHub](%s)." % url_repository)
    st.markdown("Thus, the resulting model had a good performance. Please don't take its predictions so seriously :wink:")
    st.markdown("According to the developed model, **Walmart of Mexico (WALMEX) will meet its goal of doubling its sales from 211,436 mdp to 424,050 mdp in the third quarter of 2033**.")
    st.markdown('Please go the :orange[**_Forecast_**] page to play with the model. :blush:')

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    with bcol2:
        if st.button('Go to Forecast Page'):
            st.session_state["app_page"] = "Forecast"
            st.experimental_rerun()

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
    st.write('Uses a $\text{SARIMA}(1,1,1)(1,1,1)_{4}$ trained on the historical net sales data of WALMEX (Wal-Mart de México S.A.B. de C.V., 2024) from 2014Q1 to 2023Q4 to forecast net sales over the next 10 years. Check out the code [here](%s) and more details at the <h style="color:orange;"><i><b>Homepage</b></i></h>. Please do not take the predictions from this app so seriously. 😉' % url_repository, unsafe_allow_html=True)

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    with bcol2:
        if st.button('Go to Homepage'):
            st.session_state["app_page"] = "Homepage"
            st.experimental_rerun()

    # Input data section
    st.markdown("")
    st.subheader(":blue[Next Data Points]")
    st.markdown("The model has been trained with data from 2014Q1 to 2023Q4, please input the net sales values for the next periods:")


    st.markdown("")
    st.markdown("")

    # Results section

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])

    st.session_state["flag_charts"] = 1

    with bcol2:
        if st.button('Predict Probability :nerd_face:'):
            # Get input array from user's input
            input_array = get_input_array()
            # Model
            model = get_model()

            # Prediction
            Y = model.predict(input_array)
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

        df = get_df(Y)
        pie_chart = plot_pie_chart(df)
        bar_chart = plot_bar_chart(df)

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

