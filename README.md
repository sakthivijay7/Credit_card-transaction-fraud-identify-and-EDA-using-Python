# Transaction-Healthcare-management
#### Date:May16 -2025
Credit_card fraud transaction identification using Python 
- **Data collection :**
[Link] (https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTest.csv )
Dataset collected from kaggle.

- **Data handle :**
Pandas to read the dataset and check null values,categorical values fill with mode and numerical values fill with median.

- **Data split:**
Split the dataset to Features (input) and Target (output) for training and testing 

- **Encoding:**
The category, merchant, State to be label encoding ,it was convert into numerical labels.

- **Scalling:**
Support vector, Naive bayes needed scalling then features standard scaler.

- **Algorithms:**
Using multiple algorithms for comparison `Logistic Regreesion, Kneighbours, Support vector, Naive Bayes, Decision Tree classifier, Random Forest classifier`

- **Save model:**
Models are saved in the pickle file for prediction.

- **Techniques:**
Python
Jupyter Notebook

- **Data analysis :**
Performed EDA using Matplotlib and seaborn for visual insights.

- **Result:**
Random forest achived  `accuracy 95% `it perform well on fraud identification.



