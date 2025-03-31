
# #  Credit card transaction Fraud Detection


# import pandas as pd
# import numpy as np

# #Uploading files

# from google.colab import files
# uploaded =files.upload()

# #Reading

# dataset=pd.read_csv("fraud.csv")
# #print(dataset.head())
# print(dataset.shape)
# print(dataset.columns)

# #  Copy of the selected column

# need_columns=['merchant','category','amt','state','lat','long','city_pop','is_fraud']
# ds=dataset[need_columns].copy()

# #Empty and Null values checking

# amt=ds["amt"].isnull().sum()
# category=(ds["category"]=='').sum()+ds["category"].isnull().sum()
# merchant=(ds["merchant"]=='').sum()+ds["merchant"].isnull().sum()
# state=(ds["state"]=='').sum()+ds["state"].isnull().sum()
# lat=ds["lat"].isnull().sum()
# long=ds["long"].isnull().sum()
# city_pop=ds["city_pop"].isnull().sum()
# is_fraud=ds["is_fraud"].isnull().sum()
# print(amt)
# print(category)
# print(merchant)
# print(state)
# print(lat)
# print(long)
# print(city_pop)
# print(is_fraud)

# #Missing Values to be fill

# ds["amt"]=ds["amt"].fillna(ds["amt"].median())
# ds["category"]=ds["category"].fillna(ds["category"].mode()[0])
# ds["merchant"]=ds["merchant"].fillna(ds["merchant"].mode()[0])
# ds["state"]=ds["state"].fillna(ds["state"].mode()[0])
# ds["lat"]=ds["lat"].fillna(ds["lat"].mean())
# ds["long"]=ds["long"].fillna(ds["long"].mean())
# ds["city_pop"]=ds["city_pop"].fillna(ds["city_pop"].median())

# # Features**(input)---- **Target**(output)-segregate

# x=ds[['amt','category','merchant','state','lat','long','city_pop']]
# #print(x)
# y=ds["is_fraud"]
# print(y)

# Splitting of Train and Test  datas

# x = x[y.notna()]
# y = y[y.notna()]
# print(x.shape)
# print(y.shape)

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# # LabelEncoder** assigning of **Category** values to **Numerical** values

# from sklearn.preprocessing import StandardScaler, LabelEncoder
# i = ['category', 'merchant', 'state']
# le = {}
# for j in i:
#     le[j] = LabelEncoder()
#     x_train[j] = le[j].fit_transform(x_train[j])
#     x_test[j] = le[j].transform(x_test[j])

# from sklearn.preprocessing import StandardScaler
# scale=StandardScaler()
# X_train=scale.fit_transform(x_train)
# X_test=scale.transform(x_test)
# print(X_train.shape)
# print(X_test.shape)

# # MODEL SELECTION AND ALGORITHMS

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# model1=LogisticRegression()
# model2=KNeighborsClassifier()
# model3=SVC()
# model4=GaussianNB()
# model5=DecisionTreeClassifier()
# model6=RandomForestClassifier()

## Trainning

# model1.fit(X_train,y_train)
# model2.fit(X_train,y_train)
# model3.fit(X_train,y_train)
# model4.fit(X_train,y_train)
# model5.fit(X_train,y_train)
# model6.fit(X_train,y_train)

# Models={'LR' :model1,
#         'KNN':model2,
#         'SVC':model3,
#        'GNB' :model4,
#         'DTC':model5,
#         'RFC':model6}
# import pickle
# with open("Trainedmodels.pkl","wb") as file:
#   pickle.dump(Models,file)

## Prediction

# #with open("Trainedmodels.pkl","rb") as file:
#  # loaded_models=pickle.load(file)
# #y_pred1=loaded_models['LR'].predict(x_test)

# y_pred1=model1.predict(X_test)
# y_pred2=model2.predict(X_test)
# y_pred3=model3.predict(X_test)
# y_pred4=model4.predict(X_test)
# y_pred5=model5.predict(X_test)
# y_pred6=model6.predict(X_test)

## Accuracy score

# from sklearn.metrics import accuracy_score,confusion_matrix
# print("1.Logistic Regression     : %f",accuracy_score(y_test,y_pred1)*100)
# print("2.KNearestNeighbours      : %f",accuracy_score(y_test,y_pred2)*100)
# print("3.SupportVectorclassifier : %f",accuracy_score(y_test,y_pred3)*100)
# print("4.GaussianNB              : %f",accuracy_score(y_test,y_pred4)*100)
# print("5.DecisionTreeclassifier  : %f",accuracy_score(y_test,y_pred5)*100)
# print("6.Randomforestclassifier  : %f",accuracy_score(y_test,y_pred6)*100)

# cm=confusion_matrix(y_test,y_pred6)
# print("Confusion Matrix:")
# print(cm)

## USER INPUT TO PREDICTION


New=[2.86,"personal_care","fraud_kirlin and Sons","351 Darlene Green",33.9659, -80.9355,333497]
New_df=pd.DataFrame([New],columns=["amt","category","merchant","state","lat","long","city_pop"])

for j in ["category","merchant","state"]:
  if New_df[j][0] in le[j].classes_:
    New_df[j]=le[j].transform(New_df[j])
  else:
    New_df[j] = -1

#with open("Trainedmodels.pkl","rb") as file:
 # loaded_models=pickle.load(file)
#y_pred1=loaded_models['LR'].predict(x_test)
result=model6.predict(scale.transform(New_df)) # Pass New_df instead of New
print(result)
if result == 0:
  print("NO fraud detect for the user's credit card transaction.")
else:
  print("Fraud detect in the user's credit card transaction.")