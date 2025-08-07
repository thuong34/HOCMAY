from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load mô hình và scaler
model = load_model('model/neural_network_model.keras')
scaler = joblib.load('model/scaler.pkl')

# Load các LabelEncoder
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

    # Cột đầu vào mới theo dữ liệu mới
    required_fields = [
        'Region', 'Soil_Type', 'Crop',
        'Rainfall_mm', 'Temperature_Celsius',
        'N', 'P', 'K',
        'Irrigation_Used', 'Weather_Condition',
        'Days_to_Harvest'
    ]

    # Kiểm tra thiếu thông tin
    missing_fields = [field for field in required_fields if field not in data or data[field] == '']
    if missing_fields:
        return jsonify({'error': f'Thiếu thông tin: {", ".join(missing_fields)}'}), 400

    try:
        # Ép kiểu và kiểm tra giới hạn
        rainfall = float(data['Rainfall_mm'])
        temperature = float(data['Temperature_Celsius'])
        n = float(data['N'])
        p = float(data['P'])
        k = float(data['K'])
        irrigation = int(data['Irrigation_Used'])
        days = int(data['Days_to_Harvest'])

        if not (0 <= rainfall <= 1000):
            return jsonify({'error': 'Lượng mưa không hợp lệ (0-1000 mm).'}), 400
        if not (0 <= temperature <= 60):
            return jsonify({'error': 'Nhiệt độ không hợp lệ (0-60°C).'}), 400
        if not (0 <= n <= 300 and 0 <= p <= 300 and 0 <= k <= 300):
            return jsonify({'error': 'Giá trị N/P/K phải nằm trong khoảng 0–300.'}), 400
        if not (irrigation in [0, 1]):
            return jsonify({'error': 'Giá trị Irrigation_Used chỉ được là 0 hoặc 1.'}), 400
        if not (60 <= days <= 300):
            return jsonify({'error': 'Số ngày thu hoạch không hợp lệ (60–300 ngày).'}), 400

        # Chuẩn hóa văn bản
        region = data['Region'].strip()
        soil = data['Soil_Type'].strip()
        crop = data['Crop'].strip()
        weather = data['Weather_Condition'].strip()

        # Kiểm tra nhãn có trong encoder
        for col_name, value in zip(['Region', 'Soil_Type', 'Crop', 'Weather_Condition'],
                                   [region, soil, crop, weather]):
            if value not in label_encoders[col_name].classes_:
                return jsonify({'error': f"Giá trị '{value}' trong trường '{col_name}' không hợp lệ."}), 400

        # Mã hóa & tạo mảng đầu vào theo đúng thứ tự mô hình huấn luyện
        encoded_input = [
            label_encoders['Region'].transform([region])[0],
            label_encoders['Soil_Type'].transform([soil])[0],
            label_encoders['Crop'].transform([crop])[0],
            rainfall,
            temperature,
            n, p, k,
            irrigation,
            label_encoders['Weather_Condition'].transform([weather])[0],
            days
        ]

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý dữ liệu: {str(e)}'}), 400

    # Biến đổi & chuẩn hóa
    input_array = np.array(encoded_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input).flatten()[0]

    return jsonify({'prediction': round(float(prediction), 3)})

if __name__ == '__main__':
    app.run(debug=True)
