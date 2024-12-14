# Classification-Breast-Cancer
After working with linear regression, I wanted to explore machine learning further and developed a classification model to predict whether a breast tumor was benign or malignant.

# Building the model 
Firstly, I imported the dataset and defined my data frame as usual. In this case, it was necessary to normalize the features in X (scaling) to ensure that all variables were on the same scale, thereby improving the performance and stability of the model.
'''pyhton
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
X_df = pd.DataFrame(X, columns=data.feature_names)
y_df = pd.DataFrame(y, columns=["Target"])
df = pd.concat([X_df, y_df], axis=1)
print(df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"X_scaled:\n{X_scaled}")
'''
