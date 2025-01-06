import onnx
import onnxruntime as ort
import numpy as np
import time

onnx_model_path = 'heart_disease\model.onnx'
config_file_path = 'vaip_config.json'

# Tạo SessionOptions cho NPU (Neural Processing Unit)
npu_options = ort.SessionOptions()

# Khởi tạo InferenceSession với các trình cung cấp và tùy chọn cấu hình
npu_session = ort.InferenceSession(
    onnx_model_path,
    providers=['VitisAIExecutionProvider'],
    sess_options=npu_options,
    provider_options=[{'config_file': config_file_path}]
)

input_name = npu_session.get_inputs()[0].name
input_data = np.array([[20, 1, 1, 100, 10, 0, 0, 250, 0, 3, 2, 1, 2]], dtype=np.float32)

start_time = time.time()
outputs = npu_session.run(None, {input_name: input_data})
end_time = time.time()

prediction = outputs[0][0]
print(f'Kết quả: {prediction}')

print(f'Thời gian xử lý (VitisAI): {end_time - start_time:.4f} giây')
