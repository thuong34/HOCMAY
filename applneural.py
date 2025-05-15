from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Neural Network và scaler
model = load_model('model/neural_network_model.keras')
scaler = joblib.load('model/scaler.pkl')
label_encoders = {
    'Region': joblib.load('model/label_encoder_Region.pkl'),
    'Soil_Type': joblib.load('model/label_encoder_Soil_Type.pkl'),
    'Crop': joblib.load('model/label_encoder_Crop.pkl'),
    'Weather_Condition': joblib.load('model/label_encoder_Weather_Condition.pkl')
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

    # Đúng thứ tự cột lúc train:
    encoded = []
    encoded.append(label_encoders['Region'].transform([data['Region']])[0])
    encoded.append(label_encoders['Soil_Type'].transform([data['Soil_Type']])[0])
    encoded.append(label_encoders['Crop'].transform([data['Crop']])[0])
    encoded.append(data['Rainfall_mm'])
    encoded.append(data['Temperature_Celsius'])
    encoded.append(int(data['Fertilizer_Used']))
    encoded.append(int(data['Irrigation_Used']))
    encoded.append(label_encoders['Weather_Condition'].transform([data['Weather_Condition']])[0])
    encoded.append(data['Days_to_Harvest'])

    input_array = np.array(encoded).reshape(1, -1)

    # Scale dữ liệu
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input).flatten()[0]

    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
