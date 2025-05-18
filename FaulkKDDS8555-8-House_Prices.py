import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.compose import make_column_selector as selector

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Load the dataset
train=pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/house_prices/train.csv")
test=pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/house_prices/test.csv")

X=train.drop(columns=['Id','SalePrice'])
y=np.log1p(train['SalePrice'])  # Log transformation of SalePrice to reduce skewness and heteroscedasticity
X_test=test.drop(columns=['Id'])
test_id=test['Id']

# Check for missing values
print("\nMissing values in training data:")
print(train.isnull().sum())
# Check for missing values in the test data
print("\nMissing values in test data:")
print(test.isnull().sum())

# Column Selectors
num_selector = selector(dtype_include=np.number)
cat_selector = selector(dtype_exclude=np.number)
num_cols = num_selector(X)
cat_cols = cat_selector(X)

# Preprocessing pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))    
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

# Feature Matrix for VIF 
X_vif = preprocessor.fit_transform(X)
X_vif_df = pd.DataFrame(X_vif, columns=preprocessor.get_feature_names_out())

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif_df.values, i) for i in range(X_vif_df.shape[1])]
print("\nVIF Data:", vif_data.sort_values("VIF", ascending=False).head(10))

# Polynomial Model
poly_model= Pipeline([
    ('pre', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linreg', LinearRegression())
])

# Fit the model
poly_model.fit(X, y)

# Cross-validation
poly_rmse = -cross_val_score(poly_model, X, y, scoring='neg_root_mean_squared_error', cv=5).mean()

# Feature Importance
X_poly = poly_model.named_steps['poly'].fit_transform(preprocessor.transform(X))
feature_names = poly_model.named_steps['poly'].get_feature_names_out(preprocessor.get_feature_names_out())
coef= poly_model.named_steps['linreg'].coef_
important_features = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
important_features["abs_coef"]=np.abs(important_features["Coefficient"])
print(important_features.sort_values("abs_coef", ascending=False).head(10))

# Save the model
poly_preds= np.expm1(poly_model.predict(X_test))
pd.DataFrame({"Id": test_id, "SalePrice": poly_preds}).to_csv("poly_model_predictions.csv", index=False)

# PCA + Lasso Model
pca_model = Pipeline([
    ('pre', preprocessor),
    ('pca', PCA(n_components=0.95)),
    ('lasso', LassoCV(cv=5, random_state=42))
])

# Fit the model
pca_model.fit(X, y)

# RMSE
pca_rmse = -cross_val_score(pca_model, X, y, scoring='neg_root_mean_squared_error', cv=5).mean()

# Save prediction
pca_preds = np.expm1(pca_model.predict(X_test))
pd.DataFrame({"Id": test_id, "SalePrice": pca_preds}).to_csv("pca_model_predictions.csv", index=False)

# Residuals vs Fitted Values (PCA + Lasso Model)
plt.figure(figsize=(12, 6))
plt.scatter(pca_model.predict(X), pca_model.predict(X) - y, color='blue', s=10, label='Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals vs Fitted Values (PCA + Lasso Model)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.legend()
plt.show()



# Residuals and Q-Q plot 
residuals = y - poly_model.predict(X)
plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True)
plt.title("Histogram of Residuals (Polynomial Model)")
plt.show()

plt.figure(figsize=(12, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# RMSE Summary
print(f"Polynomial Model RMSE: {poly_rmse:.2f}")
print(f"PCA + Lasso Model RMSE: {pca_rmse:.2f}")

