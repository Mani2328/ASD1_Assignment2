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

os.getcwd()
# File name in the Path 
path = "API_EN.ATM.CO2E.GF.ZS_DS2_en_csv_v2_5995555.xlsx"
path1 = "API_EN.ATM.CO2E.PC_DS2_en_excel_v2_5995021.xlsx"
path2 = "API_SP.POP.TOTL_DS2_en_csv_v2_6011311.xlsx"
path3 = "World_Energy_Consumption.xlsx"
path4 = "city_temperature.csv"

def Year_Country(path1):
    """ This function take the file name from the path
    and produce 2 data frame, one with years as columns and 
    one with countries as columns"""
    df = pd.read_excel(path1, header=3)
    df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1, inplace = True)
    df_country = df
    df_year = df.transpose()
    df_year.reset_index(drop=False, inplace=True)
    new_header = df_year.iloc[0]
    df_year.drop([0], axis = 0, inplace=True)
    df_year.columns = new_header
    df_year = df_year.rename(columns = {'Country Name':'Year'})
    return df_year, df_country
    
df_year, df_country = Year_Country(path1)
display(df_year, df_country)

# Create list to filter only country of interest
list = ["Year", "Australia", "China", "India", "United States", "Russian Federation", "Japan", "Iran, Islamic Rep.", "Korea, Rep.", "Saudi Arabia", "Indonesia", "Singapore"] 
df_year = df_year[list].astype(float)
df_year

#Check the column year to datetime format
df_year['Year'] = pd.to_datetime(df_year['Year'],format='%Y')
print(df_year.dtypes)


# Check for missing data
percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df_year)

#Select data from 1990 to 2016, due to entries missing on others years
df_year = df_year[30:57]
df_year.reset_index(drop=True, inplace = True)

# Check for missing data again filtering
percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df_year)

# Check the statistic of the selected countries
df_year.describe()

# Plot the time series plot for CO2 emission for selected countries
fig, ax = plt.subplots()
ax.plot(df_year["Year"], df_year["China"], label = "China")
ax.plot(df_year["Year"], df_year["Singapore"], label = "Singapore")
ax.plot(df_year["Year"], df_year["India"], label = "India")
ax.plot(df_year["Year"], df_year["United States"], label = "United States")
ax.plot(df_year["Year"], df_year["Japan"], label = "Japan")
ax.plot(df_year["Year"], df_year["Australia"], label = "Australia")
ax.plot(df_year["Year"], df_year["Indonesia"], label = "Indonesia")
ax.plot(df_year["Year"], df_year["Russian Federation"], label = "Russia")
ax.plot(df_year["Year"], df_year["Korea, Rep."], label = "Korea")
ax.set_xlabel("Year")
ax.set_ylabel("CO2 emissions (metric tons per capita)", fontsize=8)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("CO2 Emission from 1990 to 2016")
plt.show()

corr = df_year.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')























def Year_Country(path):
    """ This function take the file name from the path
    and produce 2 data frame, one with years as columns and 
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

# Check the statics of numerical data such as mean, min, max
df_year.describe()
# 
df_year.info()

# Create list to filter only country of interest
list = ["Year", "Australia", "China", "India", "United States", "Russian Federation", "Japan", "Iran, Islamic Rep.", "Korea, Rep.", "Saudi Arabia", "Indonesia", "Singapore"] 
df_year = df_year[list]
df_year

# Change the column year to datetime format year
df_year['Year'] = pd.to_datetime(df_year['Year'],format='%Y')
print(df_year.dtypes)

""" Check for missing data """
percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df_year)


#Select data from 1990 to 2016, due missing data in other years
df_year = df_year[30:57]
df_year.reset_index(drop=True, inplace = True)


# Check for missing data again
# Base on the percenage of missing Germany is 58.7% of missing 
# Base on the information we will drop Germany 
percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df_year)

# Checj the statistics based on the selected countries
df_year.describe()

#Check for correlation between the data



corr = df_year.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')