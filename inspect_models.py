import joblib
from tensorflow.keras.models import load_model
import numpy as np

# === 1. MỞ VÀ XEM MÔ HÌNH KERAS ===
print("===== MÔ HÌNH NEURAL NETWORK (.keras) =====")
try:
    keras_model = load_model('model/neural_network_model.keras')
    keras_model.summary()

    for i, layer in enumerate(keras_model.layers):
        weights, biases = layer.get_weights()
        print(f"\n🔹 Lớp {i} - {layer.name}")
        print(f"  Trọng số: {weights.shape}")
        print(f"  Bias: {biases.shape}")
except Exception as e:
    print("Lỗi khi mở file .keras:", e)
