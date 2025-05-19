import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Load datasets
train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/test.csv")
sample_df = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/sampleSubmission.csv")

# Define required columns
drop_cols = ['Dates', 'Descript', 'Resolution', 'Address', 'Id']
target_col = 'Category'
category_names = sample_df.columns[1:]  # 39 categories from sampleSubmission.csv

# Clean train/test data
X = train.drop(columns=drop_cols + [target_col], errors='ignore')
y = train[target_col].astype('category')
test_ids = test['Id']
test = test.drop(columns=drop_cols, errors='ignore')

# Drop rare classes
class_counts = y.value_counts()
y = y[~y.isin(class_counts[class_counts < 2].index)]
X = X.loc[y.index]

# Encode categorical variables
low_card_cols = ['PdDistrict', 'DayOfWeek']
X_cat = pd.get_dummies(X[low_card_cols])
test_cat = pd.get_dummies(test[low_card_cols])

X_num = X[['X', 'Y']]
test_num = test[['X', 'Y']]

X_full = pd.concat([X_num, X_cat], axis=1)
test_full = pd.concat([test_num, test_cat], axis=1)

# Align columns
X_final, test_final = X_full.align(test_full, join='left', axis=1, fill_value=0)

# Encode target and scale features
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_final = X_final.sample(n=50000, random_state=42)
y_encoded = y_encoded[X_final.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)
test_scaled = scaler.transform(test_final)

# Train SVM
svm_model = LinearSVC(random_state=42, max_iter=5000)
svm_model.fit(X_scaled, y_encoded)

# Predict on test set
svm_preds = svm_model.predict(test_scaled)
svm_labels = le.inverse_transform(svm_preds)

# Create submission dataframe with all 39 category columns
svm_submission = pd.DataFrame(0, index=range(len(test_ids)), columns=category_names)
for i, label in enumerate(svm_labels):
    if label in category_names:
        svm_submission.loc[i, label] = 1.0

# Insert ID column and save submission
svm_submission.insert(0, 'Id', test_ids)
svm_submission.to_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/svm_submission.csv", index=False)
