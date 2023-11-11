import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
data = pd.read_excel('embeddingsdata.xlsx')

# 'embed_0' and 'embed_1' are the features and 'Label' is the target variable
features = data[['embed_0', 'embed_1']]
target = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train and test SVM with 'linear' kernel
clf_linear = SVC(kernel='linear')
clf_linear.fit(X_train, y_train)
accuracy_linear = clf_linear.score(X_test, y_test)
print(f'Accuracy with linear kernel: {accuracy_linear}')

# Train and test SVM with 'poly' kernel
clf_poly = SVC(kernel='poly')
clf_poly.fit(X_train, y_train)
accuracy_poly = clf_poly.score(X_test, y_test)
print(f'Accuracy with poly kernel: {accuracy_poly}')

# Train and test SVM with 'rbf' kernel
clf_rbf = SVC(kernel='rbf')
clf_rbf.fit(X_train, y_train)
accuracy_rbf = clf_rbf.score(X_test, y_test)
print(f'Accuracy with rbf kernel: {accuracy_rbf}')

# Train and test SVM with 'sigmoid' kernel
clf_sigmoid = SVC(kernel='sigmoid')
clf_sigmoid.fit(X_train, y_train)
accuracy_sigmoid = clf_sigmoid.score(X_test, y_test)
print(f'Accuracy with sigmoid kernel: {accuracy_sigmoid}')