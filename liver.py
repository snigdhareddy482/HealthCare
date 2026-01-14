import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Synthetic Data Generation
# Means and Stds from streamlit_app.py stats
# Features: Age, Gender(0/1), TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G Ratio
n_samples = 1000
means = [44.75, 0.5, 3.3, 1.49, 290.6, 80.7, 109.9, 6.48, 3.14, 0.95] # Gender mean approx 0.5
stds = [16.19, 0.5, 6.21, 2.81, 242.94, 182.62, 288.92, 1.09, 0.80, 0.32]

data_dict = {
    "Age": np.random.normal(means[0], stds[0], n_samples),
    "Gender": np.random.randint(0, 2, n_samples),
    "Total_Bilirubin": np.random.normal(means[2], stds[2], n_samples),
    "Direct_Bilirubin": np.random.normal(means[3], stds[3], n_samples),
    "Alkaline_Phosphotase": np.random.normal(means[4], stds[4], n_samples),
    "Alamine_Aminotransferase": np.random.normal(means[5], stds[5], n_samples),
    "Aspartate_Aminotransferase": np.random.normal(means[6], stds[6], n_samples),
    "Total_Protiens": np.random.normal(means[7], stds[7], n_samples),
    "Albumin": np.random.normal(means[8], stds[8], n_samples),
    "Albumin_and_Globulin_Ratio": np.random.normal(means[9], stds[9], n_samples),
    "Dataset": np.random.randint(1, 3, n_samples) # 1 or 2
}
data = pd.DataFrame(data_dict)
data["Dataset"]=data["Dataset"].map({1:0,2:1})
print("Synthetic data generated.")
print(data.shape[1])
print(data.columns)


target=data["Dataset"]
source=data.drop(["Dataset"],axis=1)
sm=SMOTE()
# sc=StandardScaler()
lr=LogisticRegression()
# source=sc.fit_transform(source) # Removed scaling to match app.py expectation (simple model)
X_train,X_test,y_train,y_test= train_test_split(source,target,test_size=0.01)
X_train, y_train=sm.fit_resample(X_train,y_train)
cv=cross_validate(lr,X_train,y_train,cv=10)
lr.fit(X_train,y_train)
print(cv)
joblib.dump(lr,"model4")



