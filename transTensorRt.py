import tensorrt as trt

# 初始化TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 创建TensorRT builder和network
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# 读取ONNX模型
with open('models/yolo11x.onnx', 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# 配置builder
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# 构建引擎
engine = builder.build_engine(network, config)

# 保存引擎文件
with open('yolo11x.engine', 'wb') as f:
    f.write(engine.serialize())