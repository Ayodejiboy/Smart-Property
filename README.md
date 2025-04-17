Introduction:

Property management in Nigeria is plagued by inefficiencies, from manual rent collection to delayed maintenance requests.

=> Inefficiencies and high operational costs
=> Poor communication and tenant dissatisfaction
=> Data management issues and lack of insights
=> Maintenance and repair delays
PROJECT AIM
This project aims to design a smart property management platform to improve the experience for property managers, landlords, and tenants.

In [74]:
# Importing Libery

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
In [2]:
# loading the dataset
df = pd.read_csv('Smart_Property_Management _Dataset.csv')
# Display basic information about the dataset
df_info = df.info()
df_head = df.head()

display(df_info, df_head)
# Descriptive Statistics
numerical_summary = df.describe()
categorical_summary = df.describe(include=['object'])

display(numerical_summary, categorical_summary)
# Identify and drop irrelevant features
irrelevant_features = ['User_ID', 'Property_ID']
df = df.drop(columns = irrelevant_features, errors = 'ignore')

df.head()# To check the change effect
# convertingvariable to appropriate datatype
df['Age'] = pd.to_numeric(df['Age'], errors='coerce', downcast='float')


df.head()
:
# Handle missing values
# df.select_dtypes(include = ['object']).columns: df[columns].fillna(df[columns]).mode()[0], inplace = True)
df['Sex'].fillna(df['Sex'].mode()[0], inplace=True)
df['Used_Property_Management_Software'].fillna(df['Used_Property_Management_Software'].mode()[0], inplace=True)
df['Devices_Used_Most'].fillna(df['Devices_Used_Most'].mode()[0], inplace=True)
df['Preferred_Contact_Method'].fillna(df['Preferred_Contact_Method'].mode()[0], inplace=True)
df['Family_status'].fillna(df['Family_status'].mode()[0], inplace=True)
df['Thoughts_On_Smart_Home_Integration'].fillna(df['Thoughts_On_Smart_Home_Integration'].mode()[0], inplace=True)
df['Property_Management_Service_Need_Improvement'].fillna(df['Property_Management_Service_Need_Improvement'].mode()[0], inplace=True)
df['Used_Property_Management_Software'].fillna(df['Used_Property_Management_Software'].mode()[0], inplace=True)
df['Preferred_Payment_Method'].fillna(df['Preferred_Payment_Method'].mode()[0], inplace=True)
df['State'].fillna(df['State'].mode()[0], inplace=True)
df['Remainder_Report_Duration'].fillna(df['Remainder_Report_Duration'].mode()[0], inplace=True)
df['Country'].fillna(df['Country'].mode()[0], inplace=True)
df['City'].fillna(df['City'].mode()[0], inplace=True)
df['Type'].fillna(df['Type'].mode()[0], inplace=True)
df['Property_size'].fillna(df['Property_size'].median(), inplace=True)
df['Years_Of_Experience'].fillna(df['Years_Of_Experience'].median(), inplace=True)

# Verify changes
df.info()
missing_values = df.isnull().sum()
print(missing_values)
df.head()
# Visualizations
# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

:
# Property Size Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Property_size'].dropna(), bins=20, kde=True)
plt.title('Property Size Distribution')
plt.xlabel('Property Size')
plt.ylabel('Frequency')
plt.show()
# Devices Used Most Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Devices_Used_Most', data=df)
plt.title('Devices Used Most Distribution')
plt.xlabel('Count')
plt.ylabel('Devices Used Most')
plt.show()
display(df)

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Used_Property_Management_Software', 'Devices_Used_Most', 'Sex', 'Preferred_Contact_Method', 
                                         'Family_status', 'Thoughts_On_Smart_Home_Integration',
                                        'Property_Management_Service_Need_Improvement', 'Preferred_Payment_Method', 'Remainder_Report_Duration',
                                        'Country', 'State', 'City', 'Type'], drop_first=True)

display(df_encoded)
# One-hot encoding of categorical variables
categorical_columns = df_encoded.select_dtypes(include=['object']).columns
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_categorical_data = one_hot_encoder.fit_transform(df[categorical_columns])

# Create a DataFrame for the encoded categorical variables
encoded_categorical_df_encoded = pd.DataFrame(
    encoded_categorical_data,
    columns=one_hot_encoder.get_feature_names_out(categorical_columns)
)


# Drop original categorical columns and concatenate encoded columns
dataset = df_encoded.drop(columns=categorical_columns)
dataset = pd.concat([df_encoded, encoded_categorical_df_encoded], axis=1)
