from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
gender_encoder = pickle.load(open("gender_encoder.pkl", "rb"))
education_encoder = pickle.load(open("education_encoder.pkl", "rb"))
job_encoder = pickle.load(open("job_encoder.pkl", "rb"))

# Load dataset to extract dropdown options
df = pd.read_csv("Salary Data.csv")

gender_options = sorted(df['Gender'].dropna().unique())
education_options = sorted(df['Education Level'].dropna().unique())
job_options = sorted(df['Job Title'].dropna().astype(str).unique())

@app.route('/')
def index():
    return render_template("index.html", 
        gender_options=gender_options, 
        education_options=education_options, 
        job_options=job_options
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        education = request.form['education']
        job = request.form['job']
        experience = int(request.form['experience'])

        if age < 18 or experience >= age:
            return "Invalid input: Age must be ≥ 18 and experience < age."

        # Encoding
        gender_encoded = gender_encoder.transform([gender])[0]
        education_encoded = education_encoder.transform([education])[0]
        job_encoded = job_encoder.transform([job])[0]

        input_data = np.array([[age, gender_encoded, education_encoded, job_encoded, experience]])
        prediction_scaled = model.predict(input_data)[0]
        prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

        return render_template("index.html",
            prediction=prediction,
            gender_options=gender_options,
            education_options=education_options,
            job_options=job_options
        )
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
