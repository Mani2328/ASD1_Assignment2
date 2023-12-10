"""
Applied Data Science Assignment 2 by K.Manivannan
The data set from data.worldBank org. Look in CO2 emission by
countries
"""
# Import the python package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import stats

# File name in the Path 
#path1 file is CO2 emissions (metric tons per) from worldbabk
path1 = "API_EN.ATM.CO2E.PC_DS2_en_excel_v2_5995021.xlsx" 

#path2 file is CO2 emissions from gaseous fuel consumption (% of total)
path2 = "API_EN.ATM.CO2E.GF.ZS_DS2_en_csv_v2_5995555.xlsx" 

#path3 is world population and forest area extract from worldbank data
path3 = "population_forest.xlsx"

#path4 Global Time Series - National Centers for Environmental Information
path4 = "Europe_Temperature_data.xlsx"

#path5 data extract from worldbank for forest area, CO2, population GHG
path5 = "World_Data.xlsx"

# Data set 1 - CO2 emission per capital emission

#Create function to split the dataframe
def Year_Country(path1, h=3):
    """
    This function take the file name as argument 
    and produce 2 data frame, one with years as columns and 
    transpose the first dataframe and produce another dataframe
    with countries as columns
    
    Parameters
    arg_1 : dataframe #from world bank format
    arg_2 : int
    
    Returns:
    Dataframe1 : with year as column
    Dataframe2 : with country as column
    """
    df = pd.read_excel(path1, header = h)
    df.drop(["Country Code", "Indicator Name", "Indicator Code"], 
            axis=1, inplace = True) #Drop 3 columns above
    df_country = df
    df_year = df.transpose()  # Transpose dataframe
    df_year.reset_index(drop = False, inplace = True)
    new_header = df_year.iloc[0] #Select the header row
    df_year.drop([0], axis = 0, inplace = True)
    df_year.columns = new_header  # Set header for columns
    df_year = df_year.rename(columns = {'Country Name':'Year'})
    return df_year, df_country  #return 2 data frame
    
df_year, df_country = Year_Country(path1, h = 3) # run Year_Country function
print(df_year, df_country)

# Create list to filter only countries of interest
list = ["Year", "Australia", "China", "India", "United States",
        "Russian Federation", "Japan", "Iran, Islamic Rep.", "Korea, Rep.",
        "Saudi Arabia", "Indonesia", "Singapore"] 
df_year = df_year[list].astype(float)
df_year

#Change the column year to datetime format
df_year['Year'] = pd.to_datetime(df_year['Year'], format ='%Y')
print(df_year.dtypes)


# Check for missing data
percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                      '%_missing_data': percent_missing})
print(missing_value_df_year)

#Select data from 1990 to 2016, due to entries missing on others years
df_year = df_year[30: 57]
df_year.reset_index(drop = True, inplace = True)

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
ax.set_ylabel("CO2 emissions (metric tons per capita)", fontsize = 8)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("CO2 Emission from 1990 to 2016")
plt.show()

# Change for any correlation amount the data
corr = df_year.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot = True, ax = ax, cmap = 'coolwarm')


def print_stats(data):
    """the print_stats function make use of the stats modules
    and calculate the skewness and kurtosis. Produce average 
    and standard deviation for the data set
    
    Parameters
    arg_1 : dataframe
        
    Returns: 
    average: float
    standard deviation: float
    skewness : float
    kurtosis : float   
    """
    print("average:", np.average(data))
    print("std_dev:", np.std(data))
    print("skewness:", stats.skew(data))
    print("kurtosis:", stats.kurtosis(data))
    return

# Use print_stats to produce statistical CO2 emisssion per capital data
print("China:")
print_stats(df_year["China"])
print("Singapore:")
print_stats(df_year["Singapore"])
print("India:")
print_stats(df_year["India"])
print("United States:")
print_stats(df_year["United States"])
print("Japan:")
print_stats(df_year["Japan"])
print("Australia:")
print_stats(df_year["Australia"])
print("Indonesia:")
print_stats(df_year["Indonesia"])
print("Russian Federation:")
print_stats(df_year["Russian Federation"])
print("Korea:")
print_stats(df_year["Korea, Rep."])


# Data Set 2 - CO2 Emission from gaseoous fule consumption
#Year_Country(path2)
  
df_year, df_country = Year_Country(path2, h = 4)
print(df_year, df_country)



# Create list to filter only country of interest
list = ["Year", "Australia", "China", "India", "United States",
        "Russian Federation", "Japan","Iran, Islamic Rep.", "Korea, Rep.",
        "Saudi Arabia", "Indonesia", "Singapore"] 
df_year = df_year[list].astype(float)
df_year

df_year['Year'] = pd.to_datetime(df_year['Year'], format = '%Y')
print(df_year.dtypes)

#Select data from 1990 to 2016, due missing data in other years
df_year = df_year[30: 57]
df_year.reset_index(drop = True, inplace = True)

# Check for missing data
percent_missing = df_year.isnull().sum() * 100 / len(df_year)
missing_value_df_year = pd.DataFrame({'column_name': df_year.columns,
                                      '%_missing_data': percent_missing})
print(missing_value_df_year)

# Time series pplot for CO2 emission(gaseous fuel)

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
ax.set_ylabel("CO2 emissions from gaseous fuel consumption (% of total)", 
              fontsize = 9)
ax.legend(loc ='center left', bbox_to_anchor=(1, 0.5))
plt.title("CO2 Emission from gaseous fuel consumption")
plt.show()

# Create list based on iloc for countries of interest
my_list = [13, 40, 55, 106, 109, 112, 119, 126, 202, 205, 208, 251]
df_country_interest = df_country.iloc[my_list] #Countries of intester
df_country_interest.reset_index(drop=True, inplace = True) #reset index number
df_country_interest = df_country_interest.set_index(df_country_interest.columns[0])
df_country_interest # show the df_countries_interest

# create 2 subplots
fig, ax = plt.subplots(nrows=1, ncols = 2, sharey = True) 
# Bar plot for 1995
ax[0].bar(df_country_interest.index, df_country_interest[1995], label="1995")
ax[0].set_ylabel("CO2 emissions from gaseous fuel consumption (% of total)",
                 fontsize = 9) 
ax[0].set_title('1995')
ax[0].set_xticklabels(df_country_interest.index, rotation = 90)

#Bar plot for 2015
ax[1].bar(df_country_interest.index, df_country_interest[2015], label="2015")
ax[1].set_xticklabels(df_country_interest.index, rotation = 90)
ax[1].set_title('2015')

fig.suptitle('CO2 Emission from Gaseous Fuel (% of Total)')
plt.show()

# Create CO2 Emission for year 1995 and 2015 for comparison
df_country_interest1995 = df_country_interest.loc[["Australia", "Germany",
                                                   "Japan", "Saudi Arabia",
                                                   "Singapore"],
                                                  [1995 , 2015]]
df_country_interest1995.plot.bar()


# Data Set 3 - World Population and Forest Area
df = pd.read_excel(path3)
# correlation between forest area and world population
corr = np.corrcoef(df["Forest area (% of land area)"], df["World Population"])
print(corr) # print corr values
fig, ax = plt.subplots(figsize=(4, 3)) # set figure size
ax.scatter(df["Forest area (% of land area)"], df["World Population"],
           s = 50, facecolor='C0', edgecolor = 'k')
ax.set_xlabel("Forest Area % of land area")
ax.set_ylabel("World Population")
ax.set_title("Pearson Correlation Coefficent: -0.99")

# Data 4 Adnormality in Temperature
Temperature = pd.read_excel(path4)
#rolling is done to smooth the times series plot
Temperature_rolling = Temperature.rolling(window = 10).mean()
fig, ax = plt.subplots(1, 1)
ax.plot(Temperature_rolling["Year"], 
        Temperature_rolling["Temperature Anomalies (°C)"])
plt.xlabel("Year")
plt.ylabel("Europe Temperature Anomalies(degree Celsius)", fontsize=8)
plt.title("Temperature Abnomalies in Europe due to CO2 Emission" )
plt.show()

#Data 5 Correlation Heatmap
df = pd.read_excel(path5)
corr = df.corr()
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(corr, annot=True , cmap = 'RdBu', annot_kws={"fontsize":20})
sns.set(font_scale=2)
plt.show()

