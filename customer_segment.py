
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_1 = pd.read_csv('d:\marketing_campaign.csv', sep='\t')
df_info = data_1.info()
df_head = data_1.head()
df_info, df_head

print(data_1)
print(data_1.isnull().sum())

#Missing Values (Income)
#data_1['Income'].fillna(data_1['Income'].median(), inplace=True)
data_1.dropna(inplace=True)

#Create new feature for total spending
data_1['TotalSpending'] = data_1[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

features = data_1[['Income', 'TotalSpending']]

#Normalizing data
scaler = StandardScaler()
normalized_data_1 = scaler.fit_transform(features)
normalized_df = pd.DataFrame(normalized_data_1, columns=features.columns)
#normalized_df = data_1

#K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_df[['Income', 'TotalSpending']])

#Add clusters to  original dataset
data_1['Cluster_TotalSpending'] = clusters

#showing clusters based on Income and Total Spending
plt.figure(figsize=(8,6))
plt.scatter(normalized_df['Income'], normalized_df['TotalSpending'], c=clusters, cmap='plasma', marker='o', edgecolor='k', s=50)
plt.title('Customer Clusters based on Income and Total Spending')
plt.xlabel('Normalized Income')
plt.ylabel('Normalized Total Spending')
plt.grid(True)
plt.show()

#Export the final dataset with clusters for using in Tableau 
export_path = 'D:\python_prj\customer_clusters_with_python.csv'
data_1.to_csv(export_path, index=False)

