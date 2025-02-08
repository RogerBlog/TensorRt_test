from ultralytics import YOLO

# 加载预训练的 YOLOv5 模型（YOLOv11x是YOLOv5的一种变体）
model = YOLO('models/yolo11x.pt')  # 直接从路径加载

# 模型转为推理模式
model.eval()

# 导出为ONNX格式
onnx_path = "models/yolov11x.onnx"
model.export(format="onnx")  # 自动导出ONNX文件

print(f"ONNX模型已保存为: {onnx_path}")
