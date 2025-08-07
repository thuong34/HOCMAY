async function predict() {
    const data = {
        Region: document.getElementById("Region").value,
        Soil_Type: document.getElementById("Soil_Type").value,
        Crop: document.getElementById("Crop").value,
        Weather_Condition: document.getElementById("Weather_Condition").value,
        N: parseFloat(document.getElementById("N").value),
        P: parseFloat(document.getElementById("P").value),
        K: parseFloat(document.getElementById("K").value),
        Irrigation_Used: parseInt(document.getElementById("Irrigation_Used").value),
        Temperature_Celsius: parseFloat(document.getElementById("Temperature_Celsius").value),
        Rainfall_mm: parseFloat(document.getElementById("Rainfall_mm").value),
        Days_to_Harvest: parseInt(document.getElementById("Days_to_Harvest").value)
    };

    // Mapping tên tiếng Việt cho thông báo
    const fieldNames = {
        Region: "Khu vực",
        Soil_Type: "Loại đất",
        Crop: "Loại cây trồng",
        Weather_Condition: "Điều kiện thời tiết",
        N: "Hàm lượng N",
        P: "Hàm lượng P",
        K: "Hàm lượng K",
        Irrigation_Used: "Tưới tiêu",
        Temperature_Celsius: "Nhiệt độ",
        Rainfall_mm: "Lượng mưa",
        Days_to_Harvest: "Số ngày đến thu hoạch"
    };

    // Kiểm tra các trường rỗng hoặc không hợp lệ
    for (let field in data) {
        const value = data[field];

        if (typeof value === 'string' && value.trim() === "") {
            document.getElementById("result").innerText = `!!! Vui lòng nhập: ${fieldNames[field] || field}`;
            return;
        }

        if (value === null || value === undefined || value === "" || Number.isNaN(value)) {
            document.getElementById("result").innerText = `!!! Vui lòng chọn mục: ${fieldNames[field] || field}`;
            return;
        }
    }

    // Gửi yêu cầu đến API
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok && result.prediction !== undefined) {
            document.getElementById("result").innerText = ` Năng suất dự đoán: ${result.prediction.toFixed(2)} tấn/ha`;
        } else {
            document.getElementById("result").innerText = ` Lỗi: ${result.error || "Không rõ nguyên nhân"}`;
        }
    } catch (error) {
        document.getElementById("result").innerText = `Lỗi kết nối với server: ${error.message}`;
    }
}
