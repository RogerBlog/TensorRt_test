from ultralytics import YOLO

# 加载官方预训练模型
model = YOLO("models/yolo11x.pt")  # 请确认文件名正确

# 导出ONNX（推荐使用官方导出方式）
success = model.export(
    format="onnx",
    imgsz=(640, 640),       # 输入尺寸
    dynamic=False,          # 是否使用动态维度
    simplify=True,          # 简化模型
    opset=12,               # ONNX版本
    device=0                # 使用GPU
)