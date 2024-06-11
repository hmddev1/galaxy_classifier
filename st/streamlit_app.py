#Import packages
import streamlit as st
import numpy as np
import pandas as pd
import csv

import warnings
warnings.filterwarnings('ignore')

from models import classifiers

#Header
st.set_page_config(page_title='iFiMAS',  layout='wide', page_icon=':bar_chart:')
# col1, col2 = st.columns((0.09,1)) 
st.image('logo.png', width = 120)
st.title("""
                    iFiMAS 
                    The best Financial Market Analysis System
         
        """)

_, exp_col, _ = st.columns([1,3,1])
with exp_col:
    with st.expander("Read me first!"):
        st.info("""
                    However you like! But we believe you are facing the best financial market assistant. 🤷🏻
                    
                    iFiMAS (Ichimoku Financial Market Analysis System) comprises over 8,900 cryptocurrencies' historical 
                    OHLCV data sourced from the [Coin Market Cap](https://coinmarketcap.com//) pro API.
                    """)
        st.info("""
                    Version 0.1.12 updates:
                    - Auto Fetching system restarted!  
                """)
        st.info(""" 
                
                    Contact us:
                    - ifimas@gmail.com
                
                """)

