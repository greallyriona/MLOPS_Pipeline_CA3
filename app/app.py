from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model + features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

#home page
@app.route('/')
def home():
    return "Welcome to the Breast Cancer Prediction App"


#Predict rout taking user input to amke prediction
@app.route('/predict', methods=['GET', 'POST'])

def predict():

    #Home page
    if request.method == 'GET':
        return '''
        <h2>Breast Cancer Prediction</h2>

        <form method="post">
            <label>Radius Mean:</label><br>
            <input type="text" name="radius_mean"><br><br>

            <label>Texture Mean:</label><br>
            <input type="text" name="texture_mean"><br><br>

            <label>Perimeter Mean:</label><br>
            <input type="text" name="perimeter_mean"><br><br>

            <label>Area Mean:</label><br>
            <input type="text" name="area_mean"><br><br>

            <input type="submit" value="Predict">
        </form>
        '''

   #handling user input  to make prediciton based off the trained model
    if request.method == 'POST':

        try:
            data = {
                "radius_mean": float(request.form.get("radius_mean")),
                "texture_mean": float(request.form.get("texture_mean")),
                "perimeter_mean": float(request.form.get("perimeter_mean")),
                "area_mean": float(request.form.get("area_mean"))
            }
        except:
            return "<h3>Error: Please enter numerical values</h3>"

        # Convert to dataframe
        input_df = pd.DataFrame([data]) 

        # Ensure correct feature order
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]

        # Convert numeric output to label
        result = "Malignant" if prediction == 1 else "Benign"

        return f"<h3>Prediction: {result}</h3>"




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)