import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path=f"D:\\New folder (2)\\Myskills\\Python\\fraud.csv"
dataset=pd.read_csv(path)
# print(" Head :"'\n',dataset.head())
# print("Info :",'\n',dataset.info())
# print("Columns:",'\n',dataset.columns)
# print("statistical calculation:",'\n',dataset.describe())
null_values=dataset.isnull().sum()
# print("Check Null values :",'\n',null_values)
ds=dataset.copy()


cat_values=ds[['merchant','category','first', 'last', 'gender', 'street', 'city', 'state','job','trans_num']]
num_values=ds[['Column1','cc_num','amt','zip','lat','long','city_pop','unix_time','merch_lat','merch_long','is_fraud']]
date_values=ds[['trans_date_trans_time','dob']]

for col in cat_values:
    ds[col].fillna(ds[col].mode()[0],inplace=True)
for col in num_values:
    ds[col].fillna(ds[col].median(),inplace=True)
for col in date_values:
    ds[col].fillna(method="ffill",inplace=True)        
# print(ds.isnull().sum())

  ### Transaction Amount
plt.figure(figsize=(8,6))
sns.histplot(ds["amt"],bins=50,kde=True)
plt.title("Transaction Amount")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

   ### Top merchants by transactions count
merchants=ds["merchant"].value_counts().head(10)
plt.figure(figsize=(8,6))
sns.barplot(x=merchants.values,y=merchants.index,color="skyblue")
plt.title("Top merchants by trans_count" )
plt.xlabel("count")
plt.ylabel("Merchant")
plt.show()

  ### Transaction trends over time
ds["trans_date_trans_time"]=pd.to_datetime(ds["trans_date_trans_time"])
ds.set_index("trans_date_trans_time")["amt"].resample("D").sum().plot(figsize=(10,6),title="Daily transaction amount")
plt.xlabel("Date")
plt.ylabel("Total amount")
plt.show()

   ## Fraud count and rate
fraud_count=ds["is_fraud"].sum()
total_count=len(ds)
fraud_rate=(fraud_count/total_count)*100
print(f"Total fraudulent transaction:{fraud_count}")
print(f"Fraud rate:{fraud_rate:.2f}%")

## Amount analysis for fraudulent vs non-fraudulent
plt.figure(figsize=(8,6))
sns.boxplot(x="is_fraud",y="amt",data=ds)
plt.title("Transaction amount vs fraud")
plt.yscale("log")
plt.show()

## Fraud by category 
plt.figure(figsize=(10,6))
sns.countplot(data=ds,x="category",hue="is_fraud")
plt.xticks(rotation=45)
plt.title("Fraudulent transaction by category")
plt.legend(["Not Fraud","Fraud"])
plt.show()

