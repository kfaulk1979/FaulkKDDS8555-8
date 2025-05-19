import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

# Load the dataset
train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/test.csv")

# Load sample submission to get full category names
sample_df = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/sampleSubmission.csv")
category_names = sample_df.columns[1:]  # exclude 'Id'

# Define target and columns to drop
drop_cols = ['Dates', 'Descript', 'Resolution', 'Address', 'Id']
target_col = 'Category'

# Keep ID for submission later
test_ids = test['Id']


# Prepare features and target
X = train.drop(columns=drop_cols + [target_col], errors='ignore')
y = train[target_col].astype('category')

# Save test IDs
test_ids = test['Id']
test = test.drop(columns=drop_cols, errors='ignore')

# Drop rare classes
class_counts = y.value_counts()
y = y[~y.isin(class_counts[class_counts < 2].index)]
X = X.loc[y.index]

# One-hot encode categorical features
cat_cols = ['PdDistrict', 'DayOfWeek']
X_cat = pd.get_dummies(X[cat_cols])
test_cat = pd.get_dummies(test[cat_cols])
X_num = X[['X', 'Y']]
test_num = test[['X', 'Y']]

X_full = pd.concat([X_num, X_cat], axis=1)
test_full = pd.concat([test_num, test_cat], axis=1)

# Align test and train
X_final, test_final = X_full.align(test_full, join='left', axis=1, fill_value=0)

# Encode and scale
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save category names only used in training
category_names = le.inverse_transform(np.unique(y_encoded))

X_final = X_final.sample(n=50000, random_state=42)
y_encoded = y_encoded[X_final.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)
test_scaled = scaler.transform(test_final)

# Split into training/validation
X_train, X_val, y_train_enc, y_val_enc = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

y_train_cat = le.inverse_transform(y_train_enc)
y_val_cat = le.inverse_transform(y_val_enc)

# Train models
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train_cat)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_cat)

svm_model = LinearSVC(random_state=42, max_iter=5000)
svm_model.fit(X_train, y_train_enc)

gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train_cat)

# Class distribution plot
plt.figure(figsize=(10, 8))
pd.Series(le.inverse_transform(y_encoded)).value_counts().plot(kind='barh')
plt.title("Crime Category Distribution (Filtered)")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/class_distribution.png")
plt.close()

# Feature importances
importances = rf_model.feature_importances_
features = X_final.columns
forest_importance = pd.Series(importances, index=features)

plt.figure(figsize=(10, 6))
forest_importance.sort_values(ascending=True).tail(15).plot(kind='barh')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/feature_importance.png")
plt.close()

# Confusion matrices
cm_dt = confusion_matrix(y_val_cat, dt_model.predict(X_val), labels=category_names)
plt.figure(figsize=(14, 12))
sns.heatmap(cm_dt, cmap='Blues', xticklabels=category_names, yticklabels=category_names)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/confusion_matrix_dt.png")
plt.close()

cm_rf = confusion_matrix(y_val_cat, rf_model.predict(X_val), labels=category_names)
plt.figure(figsize=(14, 12))
sns.heatmap(cm_rf, cmap='Greens', xticklabels=category_names, yticklabels=category_names)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/confusion_matrix_rf.png")
plt.close()

# Accuracy comparison
models = ['Decision Tree', 'Random Forest', 'Linear SVM', 'Gradient Boosting']
accuracies = [
    accuracy_score(y_val_cat, dt_model.predict(X_val)),
    accuracy_score(y_val_cat, rf_model.predict(X_val)),
    accuracy_score(y_val_enc, svm_model.predict(X_val)),
    accuracy_score(y_val_cat, gb_model.predict(X_val))
]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/model_accuracy_comparison.png")
plt.close()

# Save probabilistic predictions in sample submission format
def save_prob_predictions(model, model_name, X_test_scaled):
    if hasattr(model, "predict_proba"):
        prob_preds = model.predict_proba(X_test_scaled)
        model_classes = model.classes_
        
        # Create empty dataframe with all categories (columns) initialized to 0
        prob_df = pd.DataFrame(0, index=np.arange(len(X_test_scaled)), columns=category_names)

        # Fill in only the classes the model actually predicted
        prob_df.loc[:, model_classes] = prob_preds

        submission_df = pd.concat([test_ids.reset_index(drop=True), prob_df], axis=1)
        submission_df.to_csv(f"/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/{model_name}_submission.csv", index=False)
    else:
        print(f"{model_name} does not support probability predictions.")


# Save submissions
save_prob_predictions(dt_model, "decision_tree", test_scaled)
save_prob_predictions(rf_model, "random_forest", test_scaled)
save_prob_predictions(gb_model, "gradient_boosting", test_scaled)

# SVM does not support predict_proba, so we use class predictions
svm_preds = le.inverse_transform(svm_model.predict(test_scaled))
pd.DataFrame({
    'Id': test_ids,
    'Category': svm_preds
}).to_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Crime Classification/svm_submission.csv", index=False)

# Evaluation reports
print("Decision Tree Report:")
print(classification_report(y_val_cat, dt_model.predict(X_val)))

print("\nRandom Forest Report:")
print(classification_report(y_val_cat, rf_model.predict(X_val)))

print("\nLinear SVM Report:")
print(classification_report(y_val_enc, svm_model.predict(X_val)))

print("\nGradient Boosting Report:")
print(classification_report(y_val_cat, gb_model.predict(X_val)))
