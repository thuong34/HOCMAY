import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ========== BƯỚC 1: Đọc dữ liệu ==========
df = pd.read_csv('crop_yield_vi_with_fertilizer.csv')

# ========== BƯỚC 2: Tiền xử lý dữ liệu ==========
df = df.dropna()

# Encode các cột phân loại
label_cols = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # lưu để dùng cho API hoặc giải mã ngược

# Xử lý cột nhị phân Irrigation_Used

df['Irrigation_Used'] = df['Irrigation_Used'].str.strip().map({'Có': 1, 'Không': 0})

# ========== BƯỚC 3: Tách features và labels ==========

X = df.drop('Yield_tons_per_hectare', axis=1)
y = df['Yield_tons_per_hectare']

# ========== BƯỚC 4: Chia dữ liệu ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== BƯỚC 5: Chuẩn hóa (cho Neural Network) ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== BƯỚC 6: Huấn luyện mô hình ==========
# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_mse = mean_squared_error(y_test, lgb_pred)
lgb_r2 = r2_score(y_test, lgb_pred)

# Neural Network

nn_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

nn_model.fit(X_train_scaled, y_train, epochs=200, batch_size=64, 
             validation_data=(X_test_scaled, y_test), 
             callbacks=[early_stopping], verbose=0)
nn_pred = nn_model.predict(X_test_scaled).flatten()
nn_mse = mean_squared_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)

# ========== BƯỚC 7: In kết quả ==========
print(f"Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")
print(f"XGBoost       - MSE: {xgb_mse:.4f}, R²: {xgb_r2:.4f}")
print(f"LightGBM      - MSE: {lgb_mse:.4f}, R²: {lgb_r2:.4f}")
print(f"Neural Network- MSE: {nn_mse:.4f}, R²: {nn_r2:.4f}")

# ========== BƯỚC 8: Biểu đồ so sánh ==========
plt.figure(figsize=(16, 4))
models = {
    'Random Forest': rf_pred,
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
    'Neural Network': nn_pred
}
for i, (name, pred) in enumerate(models.items()):
    plt.subplot(1, 4, i+1)
    sns.scatterplot(x=y_test, y=pred, alpha=0.5)
    plt.xlabel('Thực tế')
    plt.ylabel('Dự đoán')
    plt.title(name)
    plt.grid(True)
plt.tight_layout()
plt.show()

# ========== BƯỚC 9: Lưu mô hình ==========
os.makedirs("model", exist_ok=True)

joblib.dump(rf, 'model/random_forest_model.pkl')
joblib.dump(xgb_model, 'model/xgboost_model.pkl')
joblib.dump(lgb_model, 'model/lightgbm_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

for col, le in label_encoders.items():
    joblib.dump(le, f'model/label_encoder_{col}.pkl')

nn_model.save('model/neural_network_model.keras')
