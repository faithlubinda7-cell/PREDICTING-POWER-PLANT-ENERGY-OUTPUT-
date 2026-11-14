#!/usr/bin/env python
# coding: utf-8

# # Predicting Power Plant Energy Output
# **Combined Cycle Power Plant (CCPP) dataset** — Linear Regression vs Random Forest with 5-fold cross-validation.
# 
# - **Target:** `PE` – net hourly electrical energy output (MW)
# - **Features:** `AT` ambient temperature (°C), `V` exhaust vacuum (cm Hg), `AP` ambient pressure (mbar), `RH` relative humidity (%)
# 
# > Put `CCPP_data.csv` in the same folder as this notebook (or change `DATA_PATH` below).

# ## 1. Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

get_ipython().run_line_magic('matplotlib', 'inline')
RANDOM_STATE = 42


# ## 2. Load the data

# In[2]:


DATA_PATH = 'CCPP_data.csv'
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')   # utf-8-sig strips the hidden BOM before 'AT'
print('Shape:', df.shape)
df.head()


# In[3]:


df.info()
print('\nMissing values per column:')
print(df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())


# In[4]:


df.describe().round(2)


# ## 3. Feature correlation analysis

# In[5]:


corr = df.corr()
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Feature Correlation Matrix (with Target PE)')
plt.tight_layout()
plt.show()

print('Correlation with PE (strongest first):')
print(corr['PE'].drop('PE').sort_values(key=abs, ascending=False).round(3))


# ## 4. Train / test split (80 / 20)

# In[6]:


features = ['AT', 'V', 'AP', 'RH']
X = df[features]
y = df['PE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

print(f'Training samples: {len(X_train)}')
print(f'Test samples:     {len(X_test)}')


# ## 5. Define the models

# In[7]:


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
}


# ## 6. 5-fold cross-validation on the training set

# In[8]:


cv_rmse = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring='neg_root_mean_squared_error')
    cv_rmse[name] = -scores.mean()
    print(f'{name:<18} CV RMSE: {-scores.mean():.4f} MW  (± {scores.std():.4f})')


# ## 7. Train on the full training set and evaluate on the held-out test set

# In[9]:


results, predictions = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    results[name] = {
        'CV_RMSE':   cv_rmse[name],
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Test_R2':   r2_score(y_test, y_pred),
    }
    print(f"{name:<18} Test RMSE: {results[name]['Test_RMSE']:.4f} MW | R²: {results[name]['Test_R2']:.4f}")


# ## 8. Actual vs predicted

# In[10]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
lims = [y_test.min() - 2, y_test.max() + 2]
for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.5, s=12)
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_title(f"{name}\nRMSE: {results[name]['Test_RMSE']:.2f}")
    ax.set_xlabel('Actual PE (MW)')
    ax.set_ylabel('Predicted PE (MW)')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ## 9. Linear Regression coefficients vs Random Forest feature importance

# In[11]:


lr = models['Linear Regression']
rf = models['Random Forest']

coefs = pd.Series(lr.coef_, index=features).sort_values()
importances = pd.Series(rf.feature_importances_, index=features).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(coefs.index, coefs.values)
axes[0].axvline(0, color='black', linestyle='--')
axes[0].set_title('Linear Regression Coefficients')
axes[1].barh(importances.index, importances.values)
axes[1].set_title('Random Forest Importance')
plt.tight_layout()
plt.show()

print('Linear Regression intercept:', round(lr.intercept_, 4))
print(pd.DataFrame({'LR_coefficient': coefs.round(4), 'RF_importance': importances.round(4)}))


# ## 10. Summary and final model

# In[14]:


summary = pd.DataFrame(results).T.round(4)
print('SUMMARY:')
print(summary)

best = summary['Test_RMSE'].idxmin()
lr_rmse, rf_rmse = summary.loc['Linear Regression', 'Test_RMSE'], summary.loc['Random Forest', 'Test_RMSE']
print(f'\n FINAL MODEL: {best}')
print(f'Random Forest test RMSE is {(lr_rmse - rf_rmse) / lr_rmse * 100:.1f}% lower than Linear Regression')


# ## 11. Predict on new conditions (example)

# In[15]:


new_data = pd.DataFrame({'AT': [15.0, 30.0], 'V': [40.0, 70.0],
                         'AP': [1015.0, 1010.0], 'RH': [75.0, 50.0]})
new_data['Predicted_PE (MW)'] = models[best].predict(new_data[features]).round(2)
new_data


# In[ ]:





# In[ ]:





# In[ ]:




