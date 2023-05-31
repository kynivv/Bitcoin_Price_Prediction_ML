import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
 
import warnings
warnings.filterwarnings('ignore')


# Dataset
df = pd.read_csv('dataset.csv')

# EDA
plt.figure(figsize=(13,5))
plt.plot(df['Close'])
plt.title('Bitcoin_Close_Price', fontsize=12)
plt.ylabel('Price_In_USD', fontsize=12)
#plt.show()

df = df.drop(['Adj Close'], axis=1)

# Finding null values
#print(df.isnull().sum())


# Feature Engineering
features = ['Open', 'High', 'Low', 'Close']

splitted = df['Date'].str.split('-', expand=True)
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

df['is_quarter_end'] = np.where(df['month']%3 == 0,1,0)
df = df.drop(['Date'], axis=1)

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
#plt.show()

plt.figure(figsize=(10,10))

sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
#plt.show()

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.2, random_state=2022)
print(X_train.shape, X_valid.shape)



# Model Development and Evaluation
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
    models[i].fit(X_train, Y_train)
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))