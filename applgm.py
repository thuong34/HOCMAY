from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load XGBoost model và scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = {
    'Region': joblib.load('label_encoder_Region.pkl'),
    'Soil_Type': joblib.load('label_encoder_Soil_Type.pkl'),
    'Crop': joblib.load('label_encoder_Crop.pkl'),
    'Weather_Condition': joblib.load('label_encoder_Weather_Condition.pkl')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    required_fields = ['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
                       'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

    encoded = []
    for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
        le = label_encoders[col]
        encoded.append(le.transform([data[col]])[0])

    encoded.append(data['Fertilizer_Used'])
    encoded.append(data['Irrigation_Used'])
    encoded.append(data['Temperature_Celsius'])
    encoded.append(data['Rainfall_mm'])
    encoded.append(data['Days_to_Harvest'])

    input_array = np.array(encoded).reshape(1, -1)

    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)[0]

    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
