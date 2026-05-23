import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


data = pd.read_csv("data/cleaned_data.csv")

data.columns = data.columns.str.strip()
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
print(data.columns)


# Convert diagnosis (target variable)
data['class_filtered'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
x = data.drop(['id', 'diagnosis', 'class_filtered'], axis=1)
y = data['class_filtered']


#reduced number of features for training
selected_features = ["radius_mean","texture_mean","perimeter_mean","area_mean"]

x = x[selected_features]

#Splitting 80% Training & 20 % testing
x_train,x_test,y_train,y_test = train_test_split( x , y, train_size= 0.8, random_state=42)

# Train model (simple and safe)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save model + features
joblib.dump(model, "model.pkl")
joblib.dump(list(selected_features), "features.pkl")

print("Model trained and saved!")
