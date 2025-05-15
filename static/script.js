async function predict() {
    const data = {
        Region: document.getElementById("Region").value,
        Soil_Type: document.getElementById("Soil_Type").value,
        Crop: document.getElementById("Crop").value,
        Weather_Condition: document.getElementById("Weather_Condition").value,
        Fertilizer_Used: parseInt(document.getElementById("Fertilizer_Used").value),
        Irrigation_Used: parseInt(document.getElementById("Irrigation_Used").value),
        Temperature_Celsius: parseFloat(document.getElementById("Temperature_Celsius").value),
        Rainfall_mm: parseFloat(document.getElementById("Rainfall_mm").value),
        Days_to_Harvest: parseInt(document.getElementById("Days_to_Harvest").value)
    };

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (result.prediction !== undefined) {
        document.getElementById("result").innerText = ` Năng suất dự đoán: ${result.prediction.toFixed(2)} tấn/ha`;
    } else {
        document.getElementById("result").innerText = ` Lỗi: ${result.error}`;
    }
}
