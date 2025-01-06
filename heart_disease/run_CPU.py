import onnx
import onnxruntime as ort
import numpy as np
import time

onnx_model_path = 'model.onnx'

cpu_options = ort.SessionOptions()

# Khởi tạo InferenceSession với các trình cung cấp
providers = ['CPUExecutionProvider']
ort_session = ort.InferenceSession(onnx_model_path, providers=providers, sess_options=cpu_options)

# Chuẩn bị dữ liệu đầu vào
input_name = ort_session.get_inputs()[0].name
input_data = np.array([[20, 1, 1, 100, 10, 0, 0, 250, 0, 3, 2, 1, 2]], dtype=np.float32)

start_time = time.time()
outputs = ort_session.run(None, {input_name: input_data})
end_time = time.time()

# Lấy dự đoán
prediction = outputs[0][0]
print(f'Kết quả: {prediction}')

print(f'Thời gian xử lý (CPU): {end_time - start_time:.4f} giây')
