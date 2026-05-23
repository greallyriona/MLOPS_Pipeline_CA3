import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
import joblib



data = pd.read_csv("data/cleaned_data.csv")

data.columns = data.columns.str.strip()

print(data.columns)


# Convert diagnosis (target variable)
data['class_filtered'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
x = data.drop(['id', 'diagnosis', 'class_filtered'], axis=1)
y = data['class_filtered']



from sklearn.model_selection import train_test_split


# Perform Chi-square correlation analysis
from scipy.stats import chi2_contingency

chi2_results = {}
for col in x.columns:
    contingency_table = pd.crosstab(x[col], y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_results[col] = p

# Sort and select features with p-value < 0.05
selected_features = [f for f, p in chi2_results.items() if p < 0.05]
x = x[selected_features]

#Splitting 80% Training & 20 % testing
x_train,x_test,y_train,y_test = train_test_split( x , y, train_size= 0.8, random_state=42)

# Train model (simple and safe)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save model + features
joblib.dump(model, "model.pkl")
joblib.dump(selected_features, "features.pkl")

print("Model trained and saved!")
