# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('d:\marketing_campaign.csv', sep='\t')

df_info = df.info()
df_head = df.head()
df_info, df_head

print(df)
df.dropna(inplace=True)

numerical_columns = ['Income', 'Kidhome', 'Teenhome', 'Recency', 
                     'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                     'NumStorePurchases', 'NumWebVisitsMonth']

# افزودن ستون جدید برای مجموع هزینه‌ها
df['Total_spend'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                      'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

numerical_columns.append('Total_spend')

# حذف ردیف‌های دارای داده‌های گمشده در ستون‌های انتخاب شده
df_cleaned = df[numerical_columns].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned)
kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_data)
# اضافه کردن برچسب خوشه‌ها به دیتاست اصلی
df_cleaned['Cluster'] = kmeans.labels_
print(df_cleaned.head())
plt.scatter(scaled_data[:, numerical_columns.index('Income')], scaled_data[:, numerical_columns.index('Total_spend')], 
            c=kmeans.labels_, cmap='viridis')
plt.title("K-Means Clustering (Income vs Total_spend)")
plt.xlabel('Income')
plt.ylabel('Total_spend')
plt.show()

print(df_cleaned.head())
# Export Data to CSV
df_cleaned.to_csv('clustering_file.csv', index=False)




