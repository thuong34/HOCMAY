import pandas as pd

# Đọc file CSV gốc (dữ liệu tiếng Anh)
df = pd.read_csv("crop_yield.csv")

# Từ điển dịch các giá trị sang tiếng Việt

region_map = {'North': 'Miền Bắc', 'South': 'Miền Nam', 'West': 'Tây Nguyên', 'East': 'Duyên hải miền Trung'}
soil_map = {'Sandy': 'Đất Cát', 'Clay': 'Đất Sét', 'Silt': 'Đất Bùn', 'Loam': 'Đất Mùn', 'Peaty': 'Đất Than bùn', 'Chalky': 'Đất Phấn'}
crop_map = {'Rice': 'Lúa', 'Wheat': 'Lúa mì', 'Cotton': 'Bông', 'Barley': 'Đại mạch', 'Soybean': 'Đậu nành', 'Maize': 'Ngô'}
weather_map = {'Sunny': 'Nắng', 'Rainy': 'Mưa', 'Cloudy': 'Âm u'}
bool_map = {True: 'Có', False: 'Không', 1: 'Có', 0: 'Không', 'True': 'Có', 'False': 'Không'}

# Dịch các cột tương ứng
df['Region'] = df['Region'].map(region_map).fillna(df['Region'])
df['Soil_Type'] = df['Soil_Type'].map(soil_map).fillna(df['Soil_Type'])
df['Crop'] = df['Crop'].map(crop_map).fillna(df['Crop'])
df['Weather_Condition'] = df['Weather_Condition'].map(weather_map).fillna(df['Weather_Condition'])
df['Fertilizer_Used'] = df['Fertilizer_Used'].map(bool_map).fillna(df['Fertilizer_Used'])
df['Irrigation_Used'] = df['Irrigation_Used'].map(bool_map).fillna(df['Irrigation_Used'])

# Đọc dữ liệu phân bón mẫu (file data.xlsx chứa N, P, K)
df_fert = pd.read_excel("data.xlsx")

# Lấy ngẫu nhiên dữ liệu N, P, K tương ứng số dòng
fertilizer_samples = df_fert[['N', 'P', 'K']].sample(
    n=len(df), replace=True, random_state=42
).reset_index(drop=True)

# Xác định vị trí cột Fertilizer_Used, xoá và thay bằng N, P, K
if 'Fertilizer_Used' in df.columns:
    insert_position = df.columns.get_loc('Fertilizer_Used')
    df = df.drop(columns=['Fertilizer_Used'])
else:
    insert_position = len(df.columns)

#  Chèn N, P, K vào đúng vị trí cũ
df_before = df.iloc[:, :insert_position]
df_after = df.iloc[:, insert_position:]
df_final = pd.concat([df_before, fertilizer_samples, df_after], axis=1)

# Lưu dữ liệu cuối cùng
df_final.to_csv("crop_yield_vi_with_fertilizer.csv", index=False, encoding='utf-8-sig')

print("Kết quả đã lưu: crop_yield_vi_with_fertilizer.csv")
