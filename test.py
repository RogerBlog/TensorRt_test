import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTYOLO:
    def __init__(self, engine_path):
        # 初始化TensorRT运行时
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        # 加载引擎文件
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 分配输入输出内存
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))  # 使用 get_tensor_shape
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))  # 使用 get_tensor_dtype

            # 分配内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:  # 使用 get_tensor_mode
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        # 预处理
        input_img = self.preprocess(img)

        # 拷贝输入数据到GPU
        np.copyto(self.inputs[0]['host'], input_img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        # 拷贝输出回CPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        # 后处理
        return self.postprocess(self.outputs[0]['host'])

    def preprocess(self, img):
        # 官方YOLO预处理流程
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)  # 添加batch维度

    def postprocess(self, output):
        # 输出格式为 1x84x8400
        output = output.reshape(84, 8400)  # 调整为 [84, 8400]

        # 解析输出
        boxes = output[:4, :]  # 前4行是框的坐标 (x, y, w, h)
        scores = output[4:5, :]  # 第5行是置信度
        classes = output[5:, :]  # 后80行是类别概率

        # 找到每个框的最大类别分数
        class_ids = np.argmax(classes, axis=0)
        max_scores = np.max(scores, axis=0)

        # 过滤低置信度的框
        keep = max_scores > 0.5  # 置信度阈值
        boxes = boxes[:, keep]
        class_ids = class_ids[keep]
        max_scores = max_scores[keep]

        # 将框的格式从 (x, y, w, h) 转换为 (x1, y1, x2, y2)
        boxes = self.xywh_to_xyxy(boxes)

        return boxes, class_ids, max_scores

    def xywh_to_xyxy(self, boxes):
        # 将 (x, y, w, h) 转换为 (x1, y1, x2, y2)
        x, y, w, h = boxes
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.stack([x1, y1, x2, y2], axis=0)


# 使用示例
if __name__ == "__main__":
    # 初始化引擎
    detector = TRTYOLO("models/yolo11x.engine")

    # 读取测试图像
    img = cv2.imread("images/2.jpg")

    # 执行推理
    boxes, class_ids, scores = detector.infer(img)

    # 可视化结果
    for box, cls_id, score in zip(boxes.T, class_ids, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, f"Class {cls_id} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()