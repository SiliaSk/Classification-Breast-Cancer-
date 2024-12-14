# Classification-Breast-Cancer
After working with linear regression, I wanted to explore machine learning further and developed a classification model to predict whether a breast tumor was benign or malignant.

# Building the model 
Firstly, I imported the dataset and defined my data frame as usual. In this case, it was necessary to normalize the features in X (scaling) to ensure that all variables were on the same scale, thereby improving the performance and stability of the model.
```python
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
```

Next, I split the data into training and testing sets using an 80-20 ratio, with 80% for training and 20% for testing, in order to ensure proper evaluation of the model's performance. I trained the model using Logistic Regression and evaluated its performance on the testing set. Then, I evaluated the accuracy to assess the effectiveness of the model, which, fortunately, was sufficiently high.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(f"y_pred:\n{y_pred}")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

Finally, I displayed a confusion matrix with a heatmap to better understand the model's accuracy and performance, highlighting true positives, true negatives, false positives, and false negatives across classes.
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```
