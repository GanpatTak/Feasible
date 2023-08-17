
import sys
import json
import numpy as np
import pickle as pkl
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model
dtr = pkl.load(open('dtr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input_data = np.array(list(data.values())).reshape(1, -1)
    output = dtr.predict(input_data)
    return jsonify(output.tolist())  # Convert numpy array to a Python list

@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in request.form.values()]

    # Extract the 'Country' value from the form data
    selected_country = request.form['country']

    # Create a dictionary to map each country column to its value
    country_columns = {
        "Country_CANADA": 1 if selected_country == "CANADA" else 0,
        "Country_FRANCE": 1 if selected_country == "FRANCE" else 0,
        "Country_GERMANY": 1 if selected_country == "GERMANY" else 0,
        "Country_INDIA": 1 if selected_country == "INDIA" else 0,
        "Country_ITALY": 1 if selected_country == "ITALY" else 0,
        "Country_SPAIN": 1 if selected_country == "SPAIN" else 0,
        "Country_UNITED KINGDOM": 1 if selected_country == "UNITED KINGDOM" else 0,
        "Country_USA": 1 if selected_country == "USA" else 0,

    }

    # Special handling for Australia (since it's not a feature)
    if selected_country == "AUSTRALIA":
        input_values = [0 if column.startswith("Country_") else value for column, value in zip(request.form.keys(), data)]
    else:
        input_values = [country_columns[column] if column in country_columns else value for column, value in zip(request.form.keys(), data)]

    final_input = np.array(input_values).reshape(1, -1)

    output = dtr.predict(final_input)[0]
    return render_template("home.html", prediction_text="The internal feasibility value is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)