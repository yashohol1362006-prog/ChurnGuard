import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df= pd.read_csv(r"C:\Users\Yash Ohol\OneDrive\ChurnGuard\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df)

# # Checking top 10 rows
print('\nFirst 10 Rows from Dataset are : ')
print(f'{df.head(10)}')


print(f'\nShape of the Dataset is : ')
print(f'{df.shape}')

# # Information of the dataset
print(f'\nThe information about dataset is :')
print(f'{df.info()}')

# # Statistics of dataset
print(f'\nThe statistical of Dataset is :')
print(df.describe())


# # Checking how many are No and Yes 
# print('\nChurn Counts : ')
# print(df['Churn'].value_counts())

# # Checking Percentage of yes and no 
# print('\nChurn Percentage : ')
# print(df['Churn'].value_counts(normalize=True)*100) 


# # checking Churn category on graph (how ,any people are leaving and staying on app )
# sns.countplot(x= 'Churn', data= df)
# plt.title('Customer Churn Distribution')
# plt.xlabel('Churn')
# plt.ylabel('Number of customer')

# plt.show()


# Checking if Dupliacte Values aare present or not 
print('\nDuplicate Rows :')
print(df.duplicated().sum())


# Chicking missing values
print('\nMissing values present in Dataset :')
print(df.isna().sum())

print('\nChurn by Contract :')
print(pd.crosstab(df['Contract'], df['Churn']))


# we are checking if "Is there a relationship between the type of contract 
# a customer has and whether they churn?"
# sns.countplot(x= 'Contract', hue= 'Churn', data= df)
# plt.title('Churn by Contract Type')
# plt.xlabel('Contract by Types')
# plt.ylabel('Number of Customers')

# plt.show()

# # Checking Relationship with Churn to tenure
print('\nTenure staticstics :')
print(df.groupby('Churn')['tenure'].mean())

# sns.boxplot(x= 'Churn', y= 'tenure', data= df)
# plt.title('Tenure vs Churn')
# plt.xlabel('Churn')
# plt.ylabel('Tenure (Months)')

# plt.show()

print('\nMonthly Charges by Churn :')
print(df.groupby('Churn')['MonthlyCharges'].describe())


# sns.boxplot(x= 'Churn', y = 'MonthlyCharges', data= df)
# plt.title('Monthly Charges VS Churn')
# plt.xlabel('Churn')
# plt.ylabel('Monthly Charges')

# plt.show()


print('\nThe Internet Charges are ')
print(pd.crosstab(df['InternetService'],
                   df['Churn'], 
                   normalize='index')*100)

# sns.countplot( x= 'InternetService',hue= 'Churn', data= df)
# plt.title(' Churn VS Internet Servies')
# plt.xlabel('Internet Service ')
# plt.ylabel('No. of Customers')

# plt.show()



print('\n Churn Rate by Tech Support : ')
print(pd.crosstab(df['TechSupport'],
                   df['Churn'],
                     normalize='index')*100)

# sns.countplot(x='TechSupport', hue= 'Churn', data= df )
# plt.title('Churn by Tech Support')
# plt.xlabel('Tech Support ')
# plt.ylabel('No. of Customers ')

# plt.show()



print('\n Churn Rate by Online Security :')
print(
    pd.crosstab(df['OnlineSecurity'],
                 df['Churn'], 
                 normalize='index')*100
)

# sns.countplot(x ='OnlineSecurity', 
#               hue= 'Churn', 
#               data= df)
# plt.title('Churn by Online Security ')
# plt.xlabel('Online Security')
# plt.ylabel('Number of Customers')

# plt.show()


print('\n Churn Rate by Payments method :')
print(pd.crosstab(df['PaymentMethod'],
                   df['Churn'],
                   normalize='index')*100
                   )

# sns.countplot(x= 'PaymentMethod', hue= 'Churn', data= df)
# plt.title('Churn by Payment Methods ')
# plt.xlabel('Payment Methods ')
# plt.xticks(rotation= 30)

# plt.show()

print('\nChurn Rate by Senior Citizen :')
print(
    pd.crosstab(df['SeniorCitizen'],
                 df['Churn'],
                 normalize='index'

    ) * 100
)

# sns.countplot(x= 'SeniorCitizen',
#               hue= 'Churn',
#               data= df)
# plt.title('Churn by Senior Citizen Statue ')
# plt.xlabel('Senior Citizen (0= no , 1 = yes)')
# plt.ylabel('Number of Customers :  ')

# plt.show()


# print(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr())

# Here we Faced Error cuz one out of three feature was containing
# string value as '  '  which was giving error while we were making correlation matrix 

print(df[["tenure", "MonthlyCharges", "TotalCharges"]].dtypes)

# So we checked it by the help of dtypes fun and we got to know that tenure and 
# MonthlyCharges were fine but only TotalChargrs was containing the string value
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')     #error='coerce' means if pandas cant converrt ' ' into number then convert it into a null value
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].dtypes)

print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr())
# sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(),
#              annot= True,
#              cmap='coolwarm')
# plt.title('Correlation Betn Numerical Features ')
# plt.show()


# Data Cleaning 
print(df.isna().sum()) 
# Here we got 11 Missing Values in TotalCharges

# so we are filling it by meadian 
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
print(df.isna().sum())

# Checking Duplicate Values 
print(f'Duplicate Rows : {df.duplicated().sum()}')


# Chacing all Colunms
print(df.columns)

# Here after doing all the analysis we got to know that customer Di 
# is the only feature which isnt making any sence for now

X = df.drop(['customerID', 'Churn'], axis= 1)
y= df['Churn']

# Converting every category column into new feature/column then into nunmber 
X = pd.get_dummies(X, columns=[
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
], drop_first=True)
print(X.columns)


# Doing Binary Mapping
y = y.map({'No': 0, 'Yes':1})
print(y.value_counts())


# Converting into 0 and 1 values from True and False
X = X.astype(int)
print(X)

# Doing Train Test Split at 80% Training and 20% Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler


# Scaling The Dayaset
scaler = StandardScaler()

feature_names = X_train.columns

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)

#importing Logistic Regression to make model 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting by model
y_pred = model.predict(X_test)
print(y_pred)


# Model Evaluation
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'\nThe Accuracy of Model is : {accuracy}')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(f'\nThe Confusion Metrics of : {cm}')

from sklearn.metrics import precision_score
ps = precision_score(y_test, y_pred)
print(f'\nThe Precesion Score of Test is : {ps}')

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print(f'\nThe Recall Score of Model is {recall}')

from sklearn.metrics import f1_score
f1_s = f1_score(y_test, y_pred)
print(f'\n The  F1 Score of Mofel is {f1_s}')

scores = {
    'Accuracy': accuracy,
    'Precision': ps,
    'Recall':recall,
    'F1 Score':f1_s
}

sns.barplot(
    x=list(scores.keys()),
    y=[score * 100 for score in scores.values()]
)

plt.ylabel('Score (%)')
plt.xlabel('Metrics')
plt.title('Logistic Regression Model Evaluation')
plt.ylim(0, 100)

plt.show()


# Heatmap for Confusion Metrix
sns.heatmap(cm, annot=True)
plt.show()

x_praba = model.predict_proba(X_test) 
print(f'\nProobality of X test is : {x_praba}')

from sklearn.metrics import roc_curve, roc_auc_score

#Getting probablity of Churn 
y_prob = model.predict_proba(X_test)[:, 1]

#Calculating ROC Values
# We are setting  
fpr, tpr, threshold = roc_curve(y_test, y_prob)

# Calculating AUC 
aus =  roc_auc_score(y_test, y_prob)
print(f'\nAUC : {aus}')

from sklearn.metrics import precision_score, recall_score, f1_score

threshold = 0.4

# Convert probabilities into 0 or 1
y_pred_04 = (y_prob >= threshold).astype(int)

# Calculate metrics
precision_04 = precision_score(y_test, y_pred_04)
recall_04 = recall_score(y_test, y_pred_04)
f1_04 = f1_score(y_test, y_pred_04)

print("Threshold:", threshold)
print("Precision:", precision_04)
print("Recall:", recall_04)
print("F1 Score:", f1_04)

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

results = []

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    results.append([threshold, precision, recall, f1])

print('Results',results)

coefficients = model.coef_[0]
print(coefficients)


feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
feature_importance = feature_importance.sort_values(
    by='Coefficient',
    ascending=False
)

print(feature_importance)


plt.figure(figsize=(10,8))
sns.barplot(
    data=feature_importance,
    x='Coefficient',
    y='Feature'
)

plt.title('Feature Influence on Churn')
plt.xlabel('Coefficient')
plt.ylabel('Feature')

plt.show()

import joblib

joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_names, "models/model_columns.pkl")
 















