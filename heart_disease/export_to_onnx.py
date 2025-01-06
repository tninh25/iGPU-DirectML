from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from heart_disease.model import *

#Định nghĩa kiểu dữ liệu đầu vào cho mô hình
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# Chuyển mô hình sạng Onnx
onnx_model = convert_sklearn(gs_log_reg.best_estimator_, initial_types=initial_type)

# Lưu mô hình
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
