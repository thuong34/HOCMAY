from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load mô hình và encoder
model = joblib.load('random_forest_model.pkl')
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

    # Kiểm tra dữ liệu người dùng đã nhập
    required_fields = ['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius', 
                       'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

    # Encode lại dữ liệu
    encoded = []
    for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
        le = label_encoders[col]
        encoded.append(le.transform([data[col]])[0])

    # Các giá trị số
    encoded.append(data['Fertilizer_Used'])
    encoded.append(data['Irrigation_Used'])
    encoded.append(data['Temperature_Celsius'])
    encoded.append(data['Rainfall_mm'])

    # Thêm đặc trưng Days_to_Harvest vào
    encoded.append(data['Days_to_Harvest'])

    # Dự đoán
    input_array = np.array(encoded).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
