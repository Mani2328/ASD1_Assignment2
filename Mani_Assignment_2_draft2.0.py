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
import seaborn as sns
import stats

# File name in the Path 
path = "API_EN.ATM.CO2E.GF.ZS_DS2_en_csv_v2_5995555.xlsx"

def Year_Country(path):
    """ This function take the file name from the path
    product 2 data frame, one with years as columns and 
    one with countries as columns""" 
    df = pd.read_excel(path, header=4)
    df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1, inplace = True)
    df_country = df
    df_year = df.transpose()
    df_year.reset_index(drop=False, inplace=True)
    new_header = df_year.iloc[0]
    df_year.drop([0], axis = 0, inplace=True)
    df_year.columns = new_header
    df_year = df_year.rename(columns = {'Country Name':'Year'})
    return df_year, df_country
    
df_year, df_country = Year_Country(path)
display(df_year, df_country)

df_year.describe()
df_year.info()

# Create list to filter only country of interest
list = ["Year", "Australia", "China", "India", "United States", "Russian Federation", "Japan", "Iran, Islamic Rep.", "Korea, Rep.", "Saudi Arabia", "Indonesia", "Singapore"] 
df_year = df_year[list]
df_year

# Check the column year to datetime format year
df_year['Year'] = pd.to_datetime(df_year['Year'],format='%Y')

""" Check for missing data """
    
percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df_year)

df_year = df_year[0:57]

percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df_year)


corr = df_year.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')