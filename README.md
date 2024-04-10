 # <span style = 'color : Green' >Bike Sharing Assignment by  Syed Sha Khalid 
## **Problem Statement**<br>
 
A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.


A US bike-sharing provider **BoomBikes** has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 


In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

* Which variables are significant in predicting the demand for shared bikes.
* How well those variables describe the bike demands <br>

Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 
 

**Business Goal**:<br>
You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 
# Importing required libraries
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
# <span style = 'color : Red' > Reading and Understanding the Data
## Read Bike Sharing Dataset file that is "day.csv" as bike
bike = pd.read_csv("day.csv")
## <span style = 'color : Red' > Exploratory Data Analysis
# Check the head of the dataset
bike.head()
# Check the descriptive information
bike.info()
bike.describe()
# Check the shape of data frame 

print(bike.shape)
<span style='background: lightGreen '> Finding</span> :

Dataset has 730 rows and 16 columns.

Except one column that is "dteday", all other are either float or integer type.

One column is date type.

Looking at the data, there seems to be some fields that are categorical in nature, but in integer/float type.

We will analyse and finalize whether to convert them to categorical or treat as integer.

# DATA QUALITY CHECK
<span style='background: lightGreen '> Check for NULL/MISSING values</span>
# percentage of missing values in each column
#bike.isnull().sum()*100/bike.shape[0]
#or
round(100*(bike.isnull().sum()/len(bike)), 2).sort_values(ascending=False)
# row-wise null count percentage
round(100*(bike.isnull().sum(axis=1)/len(bike)),2).sort_values(ascending=False)
### Conclusion: There are no missing / Null values either in columns or rows
<span style='background: lightGreen '> Duplicate Check</span>
bike_dupl = bike.copy()

# Checking for duplicates and dropping the entire duplicate row if any
bike_dupl.drop_duplicates(subset=None, inplace=True)
bike_dupl.shape
bike.shape
The shape after running the drop duplicate command is same as the original dataframe.
### Hence we can conclude that there were zero duplicate values in the dataset.
## Data Cleaning: 
    
 <span style='background: lightGreen '>Checking value_counts() for entire dataframe</span>.

This will help to identify any Unknow/Junk values present in the dataset
bike.value_counts(ascending=False).head(1)
There seems to be no Junk/Unknown values in the entire dataset.
 <span style='background: lightGreen '> Removing redundant & unwanted columns</span>

Based on the high level look at the data and the data dictionary, the following variables can be removed from further analysis:

1) instant : Its only an index value

2) dteday : This has the date, Since we already have seperate columns for 'year' & 'month',hence, we could live without this column.

3) casual & registered : Both these columns contains the count of bike booked by different categories of customers. Since our objective is to find the total count of bikes and not by specific category, we will ignore these two columns. More over, we have created a new variable to have the ratio of these customer types.

4) We will save the new dataframe as bike_new, so that the original dataset is preserved for any future analysis/validation
bike.columns
New daraframe without 'instant', 'dteday', 'casual' & 'registered' columns as bike_new
bike_new=bike[['season', 'yr', 'mnth', 'holiday', 'weekday','workingday', 'weathersit', 'temp',
               'atemp', 'hum', 'windspeed','cnt']]
bike_new.shape
<span style='background: lightGreen '> Conver **int64** to **Catagorical** Variables </span>
# Check the datatypes before convertion
bike_new.info()
* season','mnth','weekday','weathersit' are catagorical variables not int64 hence convreting them to catagorical 
# Converting 'Season' to a categorical variable
bike_new['season'].replace([1, 2, 3, 4], ['Spring', 'Summer', 'Fall', 'Winter'], inplace = True)
bike_new['season'].value_counts()
# Converting 'mnth' to categorical variable 

import calendar

bike_new['mnth'] = bike_new['mnth'].apply(lambda x: calendar.month_abbr[x])
bike_new['mnth'].unique()
# Converting 'weekday' to objectin preparation for making dummy variable

bike_new['weekday'] = bike_new['weekday'].astype('object')
# Converting 'weathersit' to a categorical variable

bike_new['weathersit']=bike_new['weathersit'].replace([1, 2, 3], ['Clear', 'Mist_Cloudy', 'Light_Snow_Rain'])
bike_new['weathersit'].value_counts()
bike_new.info()
##  <span style = 'color : Green' > Univariate Analysis
<span style='background: lightGreen '>Visualizing Binary Columns (Numerical Variables)</span>  
plt.figure(figsize = [16,15])
plt.subplot(131)
bike_new['yr'].value_counts(normalize = True).plot.pie(explode=(0.05, 0), autopct = "%1.0f%%", startangle=10)
plt.subplot(132) 
bike_new['holiday'].value_counts(normalize = True).plot.pie(explode=(0.4, 0), autopct = "%1.0f%%", startangle=110)
plt.subplot(133)
bike_new['workingday'].value_counts(normalize = True).plot.pie(explode=(0.05, 0),autopct = "%1.0f%%", startangle=10)
plt.title('yr, holiday and Workingday',fontsize=30)
plt.show()
* By observing the 3 pi charts we can came to conclusion that 
    - 'Yr' is expected to be 50%-50% daily records of bike usage. 
    - Significantly less number of holidays(1) as compared to non-holidays(0) hence bike usage is more in 0. 
    - The same case applies to 'workingday' due to higher number of days vs non-working days.
<span style='background: lightGreen '>Visualizing Binary Columns (Categorical Variables)  </span>  
plt.figure(figsize = [20,4])
plt.subplot(141)
sns.countplot(data = bike_new, x = 'weathersit')
plt.subplot(142)
sns.countplot(data = bike_new, x = 'season')
plt.subplot(143)
sns.countplot(data = bike_new, x = 'weekday')
plt.subplot(144)
plt.xticks(rotation = 45)
sns.countplot(data = bike_new, x = 'mnth')
plt.show()
* By observing the plots we can came to conclusion that
    - When 'weathersit' is Clear, Few clouds, Partly cloudy, Partly cloudy the bikes are usage are more. 
    - The rest of the variables are shows very close values.
##  <span style = 'color : Green' > Bivariate Analysis
<span style='background: lightGreen '>Visualizing Numerical Variables vs 'cnt' </span>  
plt.figure(figsize = [16,4])
plt.subplot(131)
sns.barplot('yr', 'cnt', data = bike_new )
plt.subplot(132)
sns.barplot('holiday', 'cnt', data = bike_new)
plt.subplot(133)
sns.barplot('workingday', 'cnt', data = bike_new)
plt.show()
* By observing the plots we can came to conclusion that
    - There is a increase in number of bike users from year 2018(0) to year 2019(1).
    - There are more users during holidays(0) as compared to Non holidays(1).
    - There is a very little discrepancy between users of BoomBike on a working day(1) and non-working day(0).
<span style='background: lightGreen '>Visualizing Catagorical Variables Variables vs 'cnt' </span> <br>
* Build boxplot of all categorical variables (before creating dummies) againt the target variable 'cnt' 
* To see how each of the predictor variable stackup against the target variable.
plt.figure(figsize=(20, 8))
plt.subplot(2,3,1)
sns.boxplot(x = 'season', y = 'cnt', data =bike_new)
plt.subplot(2,3,2)
sns.boxplot(x = 'mnth', y = 'cnt', data =bike_new)
plt.subplot(2,3,3)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike_new)
plt.subplot(2,3,4)
sns.boxplot(x = 'holiday', y = 'cnt', data =bike_new)
plt.subplot(2,3,5)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike_new)
plt.subplot(2,3,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike_new)
plt.show()
Insights <br>
There were 6 categorical variables in the dataset.<br>

We used Box plot (refer the fig above) to study their effect on the dependent variable (‘cnt’) .<br>

The inference that We could derive were:<br>

* **season**: Almost 32% of the bike booking were happening in season3(fall) with a median of over 5000 booking (for the period of 2 years). This was followed by season2(summer) & season4(winter) with 27% & 25% of total booking. This indicates, season can be a good predictor for the dependent variable.<br>

* **mnth**: Almost 10% of the bike booking were happening in the months 5,6,7,8 & 9 with a median of over 4000 booking per month. This indicates, mnth has some trend for bookings and can be a good predictor for the dependent variable.<br>

* **weathersit**: Almost 67% of the bike booking were happening during ‘weathersit1 with a median of close to 5000 booking (for the period of 2 years). This was followed by weathersit2 with 30% of total booking. This indicates, weathersit does show some trend towards the bike bookings can be a good predictor for the dependent variable.<br>

* **holiday**: Almost 97.6% of the bike booking were happening when it is not a holiday which means this data is clearly biased. This indicates, holiday CANNOT be a good predictor for the dependent variable.<br>

* **weekday**: weekday variable shows very close trend (between 13.5%-14.8% of total booking on all days of the week) having their independent medians between 4000 to 5000 bookings. This variable can have some or no influence towards the predictor. I will let the model decide if this needs to be added or not.<br>

* **workingday**: Almost 69% of the bike booking were happening in ‘workingday’ with a median of close to 5000 booking (for the period of 2 years). This indicates, workingday can be a good predictor for the dependent variable<br>


##  <span style = 'color : Green' >  Correlation Matrix
plt.figure(figsize = (25,20))
sns.heatmap(bike.corr(), annot = True, cmap="RdBu")
plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated. Note:
# here we are considering only those variables (dataframe: bike_new) that were chosen for analysis

plt.figure(figsize = (25,20))
sns.heatmap(bike_new.corr(), annot = True, cmap="RdBu")
plt.show()  
Insights:<br>

The heatmap clearly shows which all variable are multicollinear in nature, and which variable have high collinearity with the target variable.<br>

We will refer this map back-and-forth while building the linear model so as to validate different correlated values along with VIF & p-value, for identifying the correct variable to select/eliminate from the model.<br>
# <span style = 'color : Red' > Creating Dummy Variables
We will create DUMMY variables for 4 categorical variables 'mnth', 'weekday', 'season' & 'weathersit'.
This below code does 3 things:<br>
        1) Create Dummy variable. <br>
        2) Drop original variable for which the dummy was created. <br>
        3) Drop first dummy variable for each set of dummies created.
bike_new = pd.get_dummies(bike_new, drop_first=True)
bike_new.info()
bike_new.shape
bike_new.columns
# <span style = 'color : Red' >  SPLITTING THE DATA
Splitting the data to <span style='background: lightGreen '> Train and Test</span>  : - We will now split the data into TRAIN and TEST (70:30 ratio)
We will use train_test_split method from sklearn package for this
* Check the shape & info  before spliting
bike_new.shape
bike_new.info()
import sklearn
from sklearn.model_selection import train_test_split
* We should specify 'random_state' so that the train and test data set always have the same rows, respectively
df_train, df_test = train_test_split(bike_new, train_size = 0.70, test_size = 0.30, random_state = 333)
* Verify the info and shape of the dataframes after split:
df_train.info()
df_train.shape
df_test.info()
df_test.shape
df_train.info()
df_train.columns
* By Observing tha data we can say that 'temp', 'atemp', 'hum', 'windspeed','cnt' are the numerical variables 
<span style='background: lightGreen '> Let's make a pairplot of all the numeric variables.</span> <br>
# Create a new dataframe of only numeric variables:

bike_num=df_train[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]

sns.pairplot(bike_num, diag_kind='kde')
plt.show()
The above Pair-Plot  tells us that there is a LINEAR RELATION between 'temp','atemp' and 'cnt'
plt.figure(figsize = (12,7))
sns.heatmap(bike_num.corr(), annot = True, cmap="RdBu")
#plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.show()
The above correlation plot tells us that there is a high correlation between 'temp','atemp' vs 'cnt'
# <span style = 'color : Red' > RESCALING THE FEATURES
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Checking the values before scaling
df_train.head()
df_train.columns
# Apply scaler() to all the numeric variables

num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
# Checking values after scaling
df_train.head()
df_train.describe()
# <span style = 'color : Red' > BUILDING A LINEAR MODEL

Dividing into X and Y sets for the model building

y_train = df_train.pop('cnt')
X_train = df_train
y_train
**RFE**(
Recursive feature elimination): We will be using the LinearRegression function from SciKit Learn for its compatibility with RFE (which is a utility from sklearn)
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)             # running RFE
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
# <span style = 'color : Red' >Building Linear Model using STATS MODEL
## <span style='background: lightGreen '> ***Model - 1***  </span>  <br>
VIF Check

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
    # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
    vif = pd.DataFrame()
    vif['Features'] = X_train_rfe.columns
    vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    vif
import statsmodels.api as sm
# Add a constant
X_train_lm1 = sm.add_constant(X_train_rfe)

# Create a first fitted model
lr1 = sm.OLS(y_train, X_train_lm1).fit()
# Check the parameters obtained

lr1.params
# Print a summary of the linear regression model obtained
print(lr1.summary())
## <span style='background: lightGreen '> ***Model - 2***  </span>  <br>
* Removing the variable 'atemp' based on its Very High 'VIF' value.<br>
* Even though the VIF of atemp is second highest, we decided to drop 'atemp' and not 'temp' based on general knowledge that temperature can be an important factor for a business like bike rentals, and wanted to retain 'temp'.<br> 
X_train_new = X_train_rfe.drop(["atemp"], axis = 1)
VIF Check
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm2 = sm.add_constant(X_train_new)

# Create a first fitted model
lr2 = sm.OLS(y_train, X_train_lm2).fit()
lr2.params
# Print a summary of the linear regression model obtained
print(lr2.summary())
## <span style='background: lightGreen '> ***Model - 3***  </span>  <br>
* Removing the variable 'hum' based on its Very High 'VIF' value.
X_train_new = X_train_new.drop(["hum"], axis = 1)
VIF Check
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm3 = sm.add_constant(X_train_new)

# Create a first fitted model
lr3 = sm.OLS(y_train, X_train_lm3).fit()
# Check the parameters obtained

lr3.params
# Print a summary of the linear regression model obtained
print(lr3.summary())
## <span style='background: lightGreen '> ***Model - 4***  </span>  <br>
* Removing the variable windspeed based on its Very High 'VIF' value.
* Even though the VIF of windspeed is second highest, we decided to drop 'windspeed' and not 'temp' based on general knowledge that temperature can be an important factor for a business like bike rentals, and wanted to retain 'temp'.<br> 
X_train_new = X_train_new.drop(['windspeed'], axis = 1)
VIF Check
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm4 = sm.add_constant(X_train_new)

# Create a first fitted model
lr4 = sm.OLS(y_train, X_train_lm4).fit()
# Check the parameters obtained

lr4.params
# Print a summary of the linear regression model obtained
print(lr4.summary())
## <span style='background: lightGreen '> ***Model - 5***  </span>  <br>
* All Variables VIF values are well below 5. The 'mnth_Jul' variable having its High P-value 0.048 which is close to 0.05 hence for safety purpose I drop this variable.
X_train_new = X_train_new.drop(["mnth_Jul"], axis = 1)
VIF Check
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm5 = sm.add_constant(X_train_new)

# Create a first fitted model
lr5 = sm.OLS(y_train, X_train_lm5).fit()
# Check the parameters obtained

lr5.params
# Print a summary of the linear regression model obtained
print(lr5.summary())

* This model looks good, as there seems to be VERY LOW Multicollinearity between the predictors and the p-values for all the predictors seems to be significant. For now, we will consider this as our final model (unless the Test data metrics are not significantly close to this number).
# <span style = 'color : Red' > Final Model Interpretation <br>
Hypothesis Testing:<br>
Hypothesis testing states that:<br>

H0:B1=B2=...=Bn=0<br>
H1: at least one Bi!=0
lr6 model coefficient values
const                `=`          0.248891<br>
yr                       `=`      0.232965<br>
temp                         `=`  0.375922<br>
season_Spring      `=`           -0.087867<br>
season_Winter          `=`        0.084976<br>
mnth_Dec                   `=`   -0.080011<br>
mnth_Feb              `=`        -0.057742<br>
mnth_Jan                  `=`    -0.080914<br>
mnth_Nov                     `=` -0.082132<br>
mnth_Sep              `=`         0.065011<br>
weathersit_Light_Snow_Rain `=`   -0.333164<br>
weathersit_Mist_Cloudy        `=`-0.072447<br>

* From the lr5 model summary, it is evident that all our coefficients are not equal to zerowhich means We REJECT the NULL HYPOTHESIS
F Statistics
F-Statistics is used for testing the overall significance of the Model: Higher the F-Statistics, more significant the Model is.

*  F-statistic:                     202.8
* Prob (F-statistic):          5.79e-176<br>
The F-Statistics value are 202.8 (which is greater than 1) and the p-value of all the variables are '0.000' Except 'mnth_Feb' = 0.011 which is also well below 0.05 it states that the overall model is significant. 
## The equation of best fitted surface based on model lr6:
**cnt = 0.248891 + (yr × 0.232965) + (temp × 0.375922) - (season_Spring × 0.087867) + (season_Winter × 0.084976) - (mnth_Dec × 0.080011) - (mnth_Feb × 0.057742) - (mnth_Jan × 0.080914) - (mnth_Nov × 0.082132) + (mnth_Sep × 0.065011) − (weathersit_Light_Snow_Rain × 0.333164) − (weathersit_Mist_Cloudy × 0.072447)**
## <span style='background: lightGreen '> Interpretation of Coefficients:  </span>  <br>

This is similar to equation: Y = B0 + B1*x1 + B2*X2 ...Bn*Xn <br>
where:<br>
* If Positive sign: A coefficients value of (B1,B2,B3...Bn) indicated that a unit increase in Independent variable(X1,X2,X3...Xn), increases the bike hire numbers by (B1,B2,B3...Bn) units.


* If Negative sign: A coefficients value of (B1,B2,B3...Bn) indicated that a unit increase in Independent variable(X1,X2,X3...Xn), decreases the bike hire numbers by (B1,B2,B3...Bn) units.


* const: The Constant value of ‘0.248891’ indicated that, in the absence of all other predictor variables <br> (i.e. when x1,x2...xn =0), The bike rental can still increase by 0.248891 units.

## <span style='background: lightGreen '> ASSUMPTIONS  </span>  <br> 
Error terms are normally distributed with mean zero (not X, Y) <br>
* Residual Analysis Of Training Data
y_train_pred = lr5.predict(X_train_lm5)
res = y_train-y_train_pred
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((res), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label

* From the above histogram, we could see that the Residuals are normally distributed. Hence our assumption for Linear Regression is valid.
### There is a linear relationship between X and Y
bike_new=bike_new[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]

sns.pairplot(bike_num, diag_kind='kde')
plt.show()

* Using the pair plot, we could see there is a linear relation between temp and atemp variable with the predictor ‘cnt’.
### There is No Multicollinearity between the predictor variables
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

* From the VIF calculation we could find that there is no multicollinearity existing between the predictor variables, as all the values are within permissible range of below 5
# <span style = 'color : Red' > MAKING PREDICTION USING FINAL MODEL
Now that we have fitted the model and checked the assumptions, it's time to go ahead and make predictions using the final model (lr5)

<span style='background: lightGreen '> Applying the scaling on the test sets  </span>  <br> 
# Apply scaler() to all numeric variables in test dataset. Note: we will only use scaler.transform, 
# as we want to use the metrics that the model learned from the training data to be applied on the test data. 
# In other words, we want to prevent the information leak from train to test dataset.

num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()
df_test.describe()
<span style='background: lightGreen '>  Dividing into X_test and y_test </span>  <br>
y_test = df_test.pop('cnt')
X_test = df_test
X_test.info()
#Selecting the variables that were part of final model.
col1=X_train_new.columns
X_test=X_test[col1]
# Adding constant variable to test dataframe
X_test_lm5 = sm.add_constant(X_test)
X_test_lm5.info()
# Making predictions using the final model (lr6)

y_pred = lr5.predict(X_test_lm5)
# <span style = 'color : Red' > MODEL EVALUATION
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred, alpha=.5)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16) 
plt.show()
## R^2 Value for TEST
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
r2_score(y_test, y_pred)
Adjusted R^2 Value for TEST
# We already have the value of R^2 (calculated in above step)

r2=0.8224454904426144
# Get the shape of X_test
X_test.shape
# n is number of rows in X

n = X_test.shape[0]


# Number of features (predictors, p) is the shape along axis 1
p = X_test.shape[1]

# We find the Adjusted R-squared using the formula

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2
# <span style = 'color : Red' > Final Result Comparison
* Train R^2 :0.817
* Train Adjusted R^2 :0.813
* Test R^2 :0.822
* Test Adjusted R^2 :0.813
* This seems to be a really good model that can very well 'Generalize' various datasets.
r2_train=0.817
r2_test=0.822
# Checking the difference between the test-train r2 score 
print('Difference in r2 Score(%)',(-r2_train + r2_test)*100)
Train_Adjusted_R2 = 0.813
Test_Adjusted_R2 = 0.813
# Checking the difference between the test-train Adjusted_R2 score 
print('Difference in Adjusted_R2 Score(%)',(Train_Adjusted_R2-Test_Adjusted_R2)*100)
# <span style = 'color : Green' > **FINAL REPORT**

As per our final Model, the top 3 predictor variables that influences the bike booking are:

* Temperature (temp) - A coefficient value of ‘0.375922’ indicated that a unit increase in temp variable increases the bike hire numbers by 0.375922 units. <br>

* Weather Situation 3 (weathersit_3)(Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered) - A coefficient value of ‘-0.333164’ indicated that, w.r.t Weathersit_3, a unit increase in Weathersit_3 variable decreases the bike hire numbers by 0.333164 units.<br>

* Year (yr) - A coefficient value of ‘0.232965’ indicated that a unit increase in yr variable increases the bike hire numbers by 0.232965 units.<br>



So, it's suggested to consider these variables utmost importance while planning, to achieve maximum Booking<br>
