"""
Applied Data Science Assignment 2 by K.Manivannan
The data set from WorldBank data. Look in CO2 emission by
countries
"""
# Import the python package
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from datetime import datetime 
import os

# Read csv file and set row 4 as header
df = pd.read_csv("CO2.csv", header=4)
df.describe()

df_Country = df.reset_index(drop=True)
df_Country = df_Country.set_index(df.columns[0])
df_Year = df_Country.transpose()
df_Year
df_Year.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=0, inplace = True) #axis = 0 drop the rows
df_Year
df_Year.info()


