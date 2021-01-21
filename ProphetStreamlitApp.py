#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pmdarima as pm
import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

st.title('📈 Automated Time Series Forecasting')

## Step 1: Import Data

deaths_df = st.file_uploader('https://raw.githubusercontent.com/CSSEGISandData/COVID19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

if deaths_df is not None:
    deaths_df = pd.read_csv(deaths_df)
    d = deaths_df.loc[:, '1/22/20':]
    d = d.transpose()
    d = d.sum(axis=1)
    d = d.to_list()
    dataset = pd.DataFrame(columns=['ds', 'y'])
    dates = list(deaths_df.columns[4:])
    dates = list(pd.to_datetime(dates))
    dataset['ds'] = dates
    dataset['y'] = d
    dataset = dataset.set_index('ds')
    dataset.head()
    dataset = dataset.diff()
    dataset = dataset.loc['2020-01-23':]
    dataset = dataset.diff()
    dataset = dataset.loc['2020-01-24':]
    
    st.write(dataset)
    
    max_date = dataset['ds'].max()
    
    st.write(max_date)

## Step 2: Select Forecast Horizon

periods_input = st.number_input('Cuántos períodos querés predecir en el futuro?', min_value = 1, max_value = 365)

if dataset is not None:
    m = Prophet()
    m.fit(dataset)

## Step 3: Visualize Forecast Data

if dataset is not None:
    future = m.make_future_dataframe(periods = periods_input)
    
    forecast = m.predict(pm.diff_inv(future))
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
"""
The next visual shows the actual (black dots) and predicted (blue line) values over time.
"""
fig1 = m.plot(forecast)
st.write(fig1)

"""
The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
"""
fig2 = m.plot_components(forecast)
st.write(fig2)

## Step 4: Download the Forecast Data

"""
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)