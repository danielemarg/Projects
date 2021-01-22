import streamlit as st

import pandas as pd

import numpy as np

from fbprophet import Prophet

from fbprophet.diagnostics import performance_metrics

from fbprophet.diagnostics import cross_validation

from fbprophet.plot import plot_cross_validation_metric

import base64



st.title('COVID-19 Time Series Forecasting')



"""

### 1: Importar dataset

"""

df = st.file_uploader('Subir el archivo csv', type='csv')



if df is not None:

    data = pd.read_csv(df)

    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 

    

    st.write(data)

    

    max_date = data['ds'].max()

    #st.write(max_date)



"""

### Step 2: Seleccionar cantidad de días para realizar el pronóstico



"""



periods_input = st.number_input('¿Cuántos periodos(días) deseas pronosticar?',

min_value = 1, max_value = 365)



if df is not None:

    m = Prophet()

    m.fit(data)



"""

### Step 3: Visualización del pronóstico



La información presentada muestra valores predichos. "yhat" es el valor predicho junto con su limite inferior e superior de 80% de intervalo de confianza.

"""

if df is not None:

    future = m.make_future_dataframe(periods=periods_input)

    

    forecast = m.predict(future)

    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]



    fcst_filtered =  fcst[fcst['ds'] > max_date]    

    st.write(fcst_filtered)

    

    """

    El siguiente gráfico nos muestra los valores actuales (puntos negros) y los predichos (linea azul) en el tiempo.

    """

    fig1 = m.plot(forecast)

    st.write(fig1)



    """

    Los siguientes gráficos muestran tendencia de los valores predichos, dia de semana de la tendencia. El área azul representa los intervalos de confianza superiores e inferiores.

    """

    fig2 = m.plot_components(forecast)

    st.write(fig2)





"""

### Step 4: Descargar los valores pronosticados



El siguiente link permite descargar el pronóstico realizado.

"""

if df is not None:

    csv_exp = fcst_filtered.to_csv(index=False)

    # When no file name is given, pandas returns the CSV as a string, nice.

    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here

    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'

    st.markdown(href, unsafe_allow_html=True)

