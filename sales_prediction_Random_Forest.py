import   pandas   as pd
from    sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestRegressor
from   sklearn.metrics import  mean_absolute_percentage_error
import    matplotlib.pyplot as plt
import numpy as np

#load datas
data_retail = pd.read_csv('D:\online_retail_II.csv')

#view first rows of data
data_retail.head()
data_retail.info()
print(data_retail)

#check null values
print(data_retail.isnull().sum())

# Data Cleaning 
#data_retail_cleaned = data_retail.dropna(subset=['Customer ID'])
#data_retail_cleaned = data_retail['Customer ID'].fillna(-1, inplace=True)
data_retail_cleaned = data_retail[~data_retail['Invoice'].str.startswith('C')]
data_retail_cleaned['Description'].fillna('NoDescription', inplace=True)
data_retail_cleaned['Customer ID'].fillna(-1, inplace=True)

# Remove rows where Quantity's value or Price's value is negative
data_retail_cleaned = data_retail_cleaned[(data_retail_cleaned['Quantity'] > 0) & (data_retail_cleaned['Price'] > 0)] 

# Extract features from 'InvoiceDate'
# Convert to datetime format
data_retail_cleaned['InvoiceDate'] = pd.to_datetime(data_retail_cleaned['InvoiceDate'])
data_retail_cleaned['Year'] = data_retail_cleaned['InvoiceDate'].dt.year
data_retail_cleaned['Month'] = data_retail_cleaned['InvoiceDate'].dt.month
data_retail_cleaned['Day'] = data_retail_cleaned['InvoiceDate'].dt.day
data_retail_cleaned['DayOfWeek'] = data_retail_cleaned['InvoiceDate'].dt.dayofweek 
data_retail_cleaned['WeekOfYear'] = data_retail_cleaned['InvoiceDate'].dt.isocalendar().week

#create new field 'Amn_sales' that present total sale value
data_retail_cleaned['Amn_sales'] = data_retail_cleaned['Quantity'] * data_retail_cleaned['Price']

# Aggregate sales by these features
sales_per_day = data_retail_cleaned.groupby(['Year','Month','Day','DayOfWeek', 'WeekOfYear']).agg({'Amn_sales':'sum'}).reset_index()

#Define features and target
X= sales_per_day[['Year','Month','Day', 'DayOfWeek','WeekOfYear']] 
y=sales_per_day['Amn_sales']  #Target

#set training and testing sets 
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

#ini random forest model
randomforest_model =RandomForestRegressor(n_estimators=100, random_state=42)

#Train model
randomforest_model.fit(X_train, y_train)

#Predict
y_pred = randomforest_model.predict(X_test)
################## 
#Evaluate Model (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape * 100:.2f}%")

#Plot Actual vs Predicted Sales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual sales')
plt.ylabel('predicted Sales')
plt.title('actual vs predicted sales (random forest)')
plt.show()

#create future dates for prediction (next 90 days)
future_dates = pd.DataFrame({
    'Year': [2024]* 90,
    'Month': [i // 30 +1 for i in range(90)] ,  
    'Day': [(i % 30) +1 for i in range(90)] ,
    'DayOfWeek': [(i %7) for i in range(90)] ,
    'WeekOfYear': [(i //7 + 1) for i in range(90)]  
})

#Predict future sales using the trained model
future_sales_predictions = randomforest_model.predict(future_dates)

#Plot futur sales predictions
plt.plot(future_sales_predictions)
plt.xlabel('days')
plt.ylabel('Predicted Sales')
plt.title('future Sales predict for Next 90 Days (random forest)')
plt.show()

data_retail_cleaned.to_csv('D:\cleaned_retail_data.csv', index=False)
print(len(future_sales_predictions)) 

future_sales_df = pd.DataFrame({
    'Day': np.arange(1, 91),
    'Random_forest': future_sales_predictions  
})

future_sales_df.to_csv('D:\\future_sales_predictions.csv', index=False)

