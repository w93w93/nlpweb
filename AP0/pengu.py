import streamlit as st
import pandas as pd

st.title('Penguins')

# import data
pengu_df = pd.read_csv('penguins.csv')
st.write(pengu_df.head(10))