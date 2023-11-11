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

# Initialize and train the Support Vector Machine (SVM) model
clf = SVC()
clf.fit(X_train, y_train)

# Get the support vectors
support_vectors = clf.support_vectors_

# Print the support vectors
print(f'Support Vectors ={support_vectors}')

# Testing the accuracy of the SVM on the test set
accuracy = clf.score(X_test[['embed_0', 'embed_1']], y_test)
print(f"Accuracy of the SVM on the test set: {accuracy}")
 
# Perform classification for the given test vector
test_vector = X_test[['embed_0', 'embed_1']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")
 
