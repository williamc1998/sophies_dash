# Data processing
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import json
import ssl,os,urllib
import altair as alt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Create synthetic time-series data
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Causal impact
from causalimpact import CausalImpact

# Set up a seed for reproducibility
np.random.seed(42)

# Autoregressive coefficients
arparams = np.array([.95, .05])

# Moving average coefficients
maparams = np.array([.6, .3])

# Create a ARMA process
arma_process = ArmaProcess.from_coeffs(arparams, maparams)
# model=sm.tsa.arima.ARIMA(data,order=(10, 1, 10))

# Create the control time-series
X = 10 + arma_process.generate_sample(nsample=500)

# Create the response time-series
y = 2 * X + np.random.normal(size=500)

# Add the true causal impact
y[300:] += 10

# Create dates
dates = pd.date_range('2021-01-01', freq='D', periods=500)

# Create dataframe
df = pd.DataFrame({'dates': dates, 'y': y, 'X': X}, columns=['dates', 'y', 'X'])

# Set dates as index
df.set_index('dates', inplace=True)

# Set pre-period
pre_period = [str(df.index.min())[:10], str(df.index[299])[:10]]

# Set post-period
post_period = [str(df.index[300])[:10], str(df.index.max())[:10]]

# Calculate the pre-daily average
pre_daily_avg = df['y'][:300].mean()

# Calculate the post-daily average
post_daily_avg = df['y'][300:].mean()

# Causal impact model
# impact = CausalImpact(data=df, pre_period=pre_period, post_period=post_period)

# Visualization
# impact_plot = impact.plot()
# impact_summary = impact.summary()


def plot_timeseries_data():
    fig = sns.set(rc={'figure.figsize':(12,8)})
    sns.lineplot(x=df.index, y=df['X'])
    sns.lineplot(x=df.index, y=df['y'])
    plt.axvline(x= df.index[300], color='red')
    plt.legend(labels = ['X', 'y'])
    return fig

ts_params = {
    "start_date": '2021-01-01',
    "ndays": 500, 
    "impact_ndays": 300,
    "base_value": 50,
    "arparams": np.array([.95, .05]),
    "maparams": np.array([.6, .3]), 
}
def generate_time_series(impact, start_date, ndays, impact_ndays, base_value, arparams, maparams):
    # Set up a seed for reproducibility
    np.random.seed(42)
    # Create a ARMA process
    arma_process = ArmaProcess.from_coeffs(arparams, maparams)
    # Create the control time-series
    control_ts = base_value + arma_process.generate_sample(nsample=ndays)
    # Create the response time-series
    response_ts = 2 * control_ts + np.random.normal(size=ndays)
    # Add the true causal impact
    response_ts[impact_ndays:] += impact
    # Create dates
    dates = pd.date_range(start_date, freq='D', periods=ndays)
    # Create dataframe
    df = pd.DataFrame({'dates': dates, 'y': response_ts, 'X': control_ts}, columns=['dates', 'y', 'X'])
    # Set dates as index
    df.set_index('dates', inplace=True)
    # Set pre-period
    pre_period = [str(df.index.min())[:10], str(df.index[impact_ndays - 1])[:10]]
    # Set post-period
    post_period = [str(df.index[impact_ndays])[:10], str(df.index.max())[:10]]
    return df, pre_period, post_period

def causal_impact_widget(impact):
    # Generate Teams A (control) and B (response) Time Series 
    df, pre_period, post_period = generate_time_series(impact=impact, **ts_params)
    # Causal Impact Model
    impact = CausalImpact(data=df, pre_period=pre_period, post_period=post_period)
    # Visualization
    return impact.plot()


st.set_option('deprecation.showPyplotGlobalUse', False)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_coding = load_lottieurl('https://assets5.lottiefiles.com/private_files/lf30_jyndijva.json')

with st.container():
    image = Image.open('/Users/william/Downloads/Agilisys-Logo-Black-RGB.png')
    st.image(image)
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Inference demonstration')
        st.title('Azure dummy data')
        st.write('Bayesian network based model which allows us to understand the impact of intervention after a certain point in time')
    with right_column:
        st_lottie(lottie_coding, height=300, key='coding')
    with st.container():
            option = st.selectbox('Select view',('Interaction event','Inference output'))
            if option == 'Interaction event':
                st.pyplot(plot_timeseries_data())
            else:
                values = st.slider(
    'Select a range of values',
    0, 20,0)
                impact = values
                
                st.pyplot(causal_impact_widget(impact))
                st.write(impact.summary())
                
                
                
    
                
