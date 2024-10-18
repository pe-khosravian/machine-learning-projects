import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

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

#limits for outliers 
quantil_lim = data_retail_cleaned[['Quantity', 'Price']].quantile([0.01, 0.99])

# Filter outliers rows
data_retail_cleaned = data_retail_cleaned[
    (data_retail_cleaned['Quantity'] >= quantil_lim.loc[0.01, 'Quantity']) & 
    (data_retail_cleaned['Quantity'] <= quantil_lim.loc[0.99, 'Quantity']) & 
    (data_retail_cleaned['Price'] >= quantil_lim.loc[0.01, 'Price']) & 
    (data_retail_cleaned['Price'] <= quantil_lim.loc[0.99, 'Price'])
]
#create new field 'Amn_sales' that present total sale value
data_retail_cleaned['Amn_sales'] = data_retail_cleaned['Quantity'] * data_retail_cleaned['Price']

# Convert to datetime format
data_retail_cleaned['InvoiceDate'] = pd.to_datetime(data_retail_cleaned['InvoiceDate'])

# Extract features from the date (year, month, day)
data_retail_cleaned['Year'] = data_retail_cleaned['InvoiceDate'].dt.year
data_retail_cleaned['Month'] = data_retail_cleaned['InvoiceDate'].dt.month
data_retail_cleaned['Day'] = data_retail_cleaned['InvoiceDate'].dt.day

# Aggregate sales by day
sales_per_day = data_retail_cleaned.groupby(['Year', 'Month', 'Day']).agg({'Amn_sales': 'sum'}).reset_index()

# Display prepared data
print(sales_per_day.head())

# features and target 
X = sales_per_day[['Year', 'Month', 'Day']]  # Features
y = sales_per_day['Amn_sales']  # Target 

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression model
lineReg_model = LinearRegression()

# Train model
lineReg_model.fit(X_train, y_train)
# Predict on test set
y_pred = lineReg_model.predict(X_test)

# evaluate model
mape = mean_absolute_percentage_error(y_test, y_pred)
print(mape)

# Plot actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# create future dates for prediction (next 90 days)
future_dates = pd.DataFrame({
    'Year': [2024] * 90,
    'Month': [i // 30 + 1 for i in range(90)],  
    'Day': [(i % 30) + 1 for i in range(90)]   # Simple day approximation
})

# Predict  sales using the train model
future_sales_predictions = lineReg_model.predict(future_dates)

# Plot future sales predictions
plt.plot(future_sales_predictions)
plt.xlabel('Days')
plt.ylabel('Predicted Sales')
plt.title('Future Sales Predictions for Next 90 Days')
plt.show()





