import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# 加载TensorRT引擎
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open('models/yolo11x.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 创建执行上下文
context = engine.create_execution_context()

# 分配输入输出内存
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
for i in range(engine.num_bindings):
    binding_name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(binding_name)
    dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
    mode = engine.get_tensor_mode(binding_name)

    # 分配内存
    size = trt.volume(shape)  # 计算内存大小
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))

    if mode == trt.TensorIOMode.INPUT:
        inputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name, 'shape': shape})
    else:
        outputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name, 'shape': shape})

# 加载图像并预处理
def preprocess_image(image_path, input_shape):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 调整图像大小并归一化
    resized_image = cv2.resize(image, (input_shape[3], input_shape[2]))  # 调整为模型输入尺寸
    normalized_image = resized_image / 255.0  # 归一化到 [0, 1]
    normalized_image = normalized_image.transpose((2, 0, 1))  # HWC -> CHW
    normalized_image = np.ascontiguousarray(normalized_image, dtype=np.float32)  # 确保内存连续
    return image, normalized_image

# 后处理函数（解析YOLO输出）
def postprocess_output(output, confidence_threshold=0.5, iou_threshold=0.5):
    # 这里需要根据YOLO的输出格式解析边界框、类别和置信度
    # 假设输出格式为 [batch, num_boxes, 5 + num_classes]
    # 其中 5 表示 [x, y, w, h, confidence]
    # 你需要根据你的模型输出格式调整解析逻辑
    boxes = []
    scores = []
    class_ids = []

    # 示例：假设 output 是 [1, 25200, 85] 的形状
    for detection in output[0]:
        scores = detection[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            # 解析边界框
            x, y, w, h = detection[0:4]
            boxes.append([x, y, w, h])
            class_ids.append(class_id)
            scores.append(float(confidence))

    # 非极大值抑制 (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
    return boxes, scores, class_ids, indices

# 绘制检测结果
def draw_detections(image, boxes, scores, class_ids, indices):
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f"Class {class_ids[i]} {scores[i]:.2f}"
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# 主函数
def infer_image(image_path):
    # 预处理图像
    input_shape = inputs[0]['shape']  # 获取输入形状 [batch, channel, height, width]
    original_image, processed_image = preprocess_image(image_path, input_shape)

    # 将输入数据复制到设备
    np.copyto(inputs[0]['host'], processed_image.ravel())

    # 执行推理
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # 获取输出
    output = outputs[0]['host'].reshape(outputs[0]['shape'])

    # 后处理输出
    boxes, scores, class_ids, indices = postprocess_output(output)

    # 绘制检测结果
    result_image = draw_detections(original_image, boxes, scores, class_ids, indices)

    # 显示结果
    cv2.imshow("Detection Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    # cv2.imwrite("detection_result.jpg", result_image)

# 运行推理
infer_image("images/1.jpg")  # 替换为你的图像路径