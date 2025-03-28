#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>  // 新增：用于调试输出
#include <vector>
#include <numeric>

namespace fs = std::filesystem;

// 辅助函数声明
void processYOLOv8Output(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh);

void processYOLOv8FlatOutput(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh);

void processYOLOv11Output(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh);

// 新增：处理YOLOv11转置输出 [1, 5, num_predictions]
void processYOLOv11TransposedOutput(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh, int num_classes);

// 图像旋转校正函数
cv::Mat houguLinesDetectAndRotated(const cv::Mat& image) {
    int h = image.rows;
    int w = image.cols;
    cv::Point2f center(w / 2.0f, h / 2.0f);
    cv::Mat image_copy;
    cv::cvtColor(image, image_copy, cv::COLOR_GRAY2BGR);

    // 1. 阈值处理
    cv::Mat thresh;
    cv::threshold(image, thresh, 200, 255, cv::THRESH_BINARY_INV);

    // 2. Canny 边缘检测
    cv::Mat edges;
    cv::Canny(thresh, edges, 0, 50);

    // 3. Hough直线检测
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 100, 5);

    if (lines.empty()) {
        return image.clone();
    }

    // 4. 计算斜率角度
    std::vector<float> k_means;
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        float deltaY = y2 - y1;
        float deltaX = x2 - x1;
        float k = std::atan2(deltaY, deltaX) / CV_PI;
        if (std::abs(k) <= 0.05f) {
            k_means.push_back(k);
            cv::line(image_copy, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
        }
    }

    if (k_means.empty()) {
        return image.clone();
    }

    // 5. 计算平均角度
    float k_mean = std::accumulate(k_means.begin(), k_means.end(), 0.0f) / k_means.size();
    float angle = k_mean * 180.0f;

    // 6. 旋转图像
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(image, rotated, M, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    return rotated;
}

// 主检测函数
void detection(const std::string& file_path, int expected_screws) {
    // 配置参数 --------------------------------
    const std::string model_path = "E:/opencvProject/LYJ/ultralytics-main/weights/20250319best_MDL_det.onnx";
    const int input_size = 640;          // 模型输入尺寸
    const float conf_threshold = 0.5f;   // 置信度阈值
    const float nms_threshold = 0.4f;    // NMS阈值
    const int num_classes = 1;          // 修改为实际类别数
    const bool show_debug = true;        // 显示调试信息

    // 1. 加载模型 ----------------------------
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        if (net.empty()) throw std::runtime_error("Failed to load model");
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }

    // 2. 遍历输入目录 ------------------------
    for (const auto& entry : fs::directory_iterator(file_path)) {
        const std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Warning: Cannot read image " << image_path << std::endl;
            continue;
        }

        // 3. 图像预处理 ----------------------
        cv::Mat gray, blurred;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // 4. 旋转校正 -------------------------
        cv::Mat corrected = houguLinesDetectAndRotated(blurred);
        cv::Mat corrected_color;
        cv::cvtColor(corrected, corrected_color, cv::COLOR_GRAY2BGR);

        // 5. 模型推理 -------------------------
        cv::Mat blob;
        try {
            // 记录原始尺寸用于坐标转换
            const int orig_width = corrected_color.cols;
            const int orig_height = corrected_color.rows;

            // 存储原始图像的备份，用于可视化
            cv::Mat display_image = corrected_color.clone();

            // 直接调整大小到模型输入尺寸，保持纵横比
            float scale_x = input_size / static_cast<float>(orig_width);
            float scale_y = input_size / static_cast<float>(orig_height);
            float scale = std::min(scale_x, scale_y);

            // 计算新尺寸
            int new_width = static_cast<int>(orig_width * scale);
            int new_height = static_cast<int>(orig_height * scale);

            // 计算填充
            int dw = (input_size - new_width) / 2;
            int dh = (input_size - new_height) / 2;

            // 创建letterbox图像，先调整大小再填充
            cv::Mat resized;
            cv::resize(corrected_color, resized, cv::Size(new_width, new_height));

            // 创建画布并将调整大小的图像复制到中央
            cv::Mat letterbox_img(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
            resized.copyTo(letterbox_img(cv::Rect(dw, dh, new_width, new_height)));

            // 记录调试信息
            std::cout << "原始尺寸: " << orig_width << "x" << orig_height << std::endl;
            std::cout << "缩放因子: " << scale << std::endl;
            std::cout << "新尺寸: " << new_width << "x" << new_height << std::endl;
            std::cout << "填充: dw=" << dw << ", dh=" << dh << std::endl;

            // 创建模型输入
            blob = cv::dnn::blobFromImage(
                letterbox_img,
                1.0 / 255.0,
                cv::Size(input_size, input_size),
                cv::Scalar(),
                true,
                false
            );

            net.setInput(blob);

            // 获取输出层名称
            std::vector<std::string> output_names = net.getUnconnectedOutLayersNames();
            std::cout << "输出层数量: " << output_names.size() << std::endl;
            for (const auto& name : output_names) {
                std::cout << "输出层名称: " << name << std::endl;
            }

            // 使用输出层名称获取输出
            std::vector<cv::Mat> outputs;
            net.forward(outputs, output_names);

            // 输出基本信息
            std::cout << "输出数量: " << outputs.size() << std::endl;

            // 处理每个输出层
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;

            // YOLOv11输出可能有不同格式，我们需要确定格式并适当处理
            for (size_t i = 0; i < outputs.size(); i++) {
                std::cout << "输出 " << i << " 形状: ";
                for (int j = 0; j < outputs[i].dims; j++) {
                    std::cout << outputs[i].size[j] << " ";
                }
                std::cout << std::endl;

                // 根据形状确定如何处理输出
                if (outputs[i].dims == 3 && outputs[i].size[2] == 5 + num_classes) {
                    // 标准YOLOv8格式输出: [batch, num_boxes, 5+num_classes]
                    std::cout << "检测到标准YOLOv8输出格式" << std::endl;
                    processYOLOv8Output(outputs[i], boxes, confidences, class_ids,
                        conf_threshold, orig_width, orig_height,
                        scale, dw, dh);
                }
                else if (outputs[i].dims == 2 && outputs[i].size[1] == 5 + num_classes) {
                    // 一些YOLOv8版本输出: [num_boxes, 5+num_classes]
                    std::cout << "检测到展平的YOLOv8输出格式" << std::endl;
                    processYOLOv8FlatOutput(outputs[i], boxes, confidences, class_ids,
                        conf_threshold, orig_width, orig_height,
                        scale, dw, dh);
                }
                else if (outputs[i].dims == 4) {
                    // 可能是YOLOv11特殊格式
                    std::cout << "检测到可能的YOLOv11专用输出格式" << std::endl;
                    processYOLOv11Output(outputs[i], boxes, confidences, class_ids,
                        conf_threshold, orig_width, orig_height,
                        scale, dw, dh);
                }
                // 新增：检测YOLOv11转置输出格式 [1, 5, num_predictions]
                else if (outputs[i].dims == 3 && outputs[i].size[1] == 5) {
                    std::cout << "检测到YOLOv11转置输出格式 [1, 5, " << outputs[i].size[2] << "]" << std::endl;
                    processYOLOv11TransposedOutput(outputs[i], boxes, confidences, class_ids,
                        conf_threshold, orig_width, orig_height,
                        scale, dw, dh, num_classes);
                }
                else {
                    std::cout << "未知的输出格式，跳过" << std::endl;
                    continue;
                }
            }

            // 如果以上都失败，尝试默认处理方式
            if (boxes.empty() && !outputs.empty()) {
                std::cout << "尝试最简单的处理方式..." << std::endl;

                // 保存原始输出到文件进行调试
                std::ofstream outfile("model_output.txt");
                const cv::Mat& output = outputs[0];
                outfile << "输出形状: ";
                for (int j = 0; j < output.dims; j++) {
                    outfile << output.size[j] << " ";
                }
                outfile << std::endl;
                outfile.close();

                // 简单YOLO处理 - 假设输出是标准格式但可能被重塑了
                try {
                    // 假设重塑后的输出形式为[1, -1, 6]（6 = 5+1类）
                    cv::Mat reshaped;
                    if (output.dims == 2) {
                        // 如果是[N, 6]格式
                        reshaped = output.clone();
                    }
                    else if (output.dims == 3) {
                        // 如果是[1, N, 6]格式
                        reshaped = output.reshape(1, output.size[0] * output.size[1]);
                    }
                    else if (output.dims == 4) {
                        // 如果是复杂格式，尝试展平为2D
                        int total_preds = output.size[1] * output.size[2];
                        reshaped = output.reshape(1, total_preds);
                    }

                    if (!reshaped.empty()) {
                        processYOLOv8FlatOutput(reshaped, boxes, confidences, class_ids,
                            conf_threshold, orig_width, orig_height,
                            scale, dw, dh);
                    }
                }
                catch (const cv::Exception& e) {
                    std::cerr << "重塑输出时出错: " << e.what() << std::endl;
                }
            }

            std::cout << "检测到 " << boxes.size() << " 个目标" << std::endl;

            // 7. NMS过滤 ----------------------
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
            std::cout << "NMS后保留 " << indices.size() << " 个目标" << std::endl;

            // 8. 处理检测结果 ------------------
            for (const int idx : indices) {
                const cv::Rect& box = boxes[idx];

                if (box.area() < 20000) {
                    continue;
                }

                // 绘制检测框
                cv::rectangle(display_image, box, cv::Scalar(0, 255, 0), 2);

                // 9. 螺丝计数逻辑 --------------
                cv::Mat roi = corrected(box);  // 使用灰度图像处理
                cv::Mat thresh;
                cv::threshold(roi, thresh, 90, 255, cv::THRESH_BINARY_INV);

                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                int screw_count = 0;
                for (const auto& contour : contours) {
                    const double area = cv::contourArea(contour);
                    if (area > 5) {  // 过滤小噪点
                        screw_count++;
                    }
                }

                // 绘制计数结果
                std::string status = (screw_count == expected_screws) ?
                    "OK: " + std::to_string(screw_count) :
                    "NG: " + std::to_string(screw_count);

                cv::putText(display_image, status,
                    cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    (screw_count == expected_screws) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                    2
                );
            }

            // 显示结果
            cv::imshow("Detection Results", display_image);
            cv::waitKey(0);

        }
        catch (const cv::Exception& e) {
            std::cerr << "Inference Error: " << e.what() << std::endl;
        }
    }
    cv::destroyAllWindows();
}

// YOLOv8标准输出处理
void processYOLOv8Output(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh) {
    const int num_boxes = output.size[1];
    const int num_classes = output.size[2] - 5;

    for (int i = 0; i < num_boxes; i++) {
        const float* data_ptr = output.ptr<float>(0, i);
        float objectness = data_ptr[4];

        if (objectness < conf_threshold) continue;

        // 查找最高类别得分
        float max_class_score = 0;
        int max_class_id = 0;
        for (int j = 0; j < num_classes; j++) {
            float class_score = data_ptr[5 + j];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                max_class_id = j;
            }
        }

        float confidence = objectness * max_class_score;
        if (confidence < conf_threshold) continue;

        // 提取边界框坐标
        float x = data_ptr[0];
        float y = data_ptr[1];
        float w = data_ptr[2];
        float h = data_ptr[3];

        // 将归一化坐标转为像素坐标
        float input_size = 640; // 假设输入尺寸为640
        x *= input_size;
        y *= input_size;
        w *= input_size;
        h *= input_size;

        // 从letterbox坐标转回原图
        x = (x - dw) / scale;
        y = (y - dh) / scale;
        w = w / scale;
        h = h / scale;

        // 转为矩形坐标
        int left = std::max(0, int(x - w / 2));
        int top = std::max(0, int(y - h / 2));
        int right = std::min(img_width, int(x + w / 2));
        int bottom = std::min(img_height, int(y + h / 2));

        // 创建矩形
        cv::Rect rect(left, top, right - left, bottom - top);
        if (rect.width <= 0 || rect.height <= 0) continue;

        boxes.push_back(rect);
        confidences.push_back(confidence);
        class_ids.push_back(max_class_id);
    }
}

// YOLOv8平面输出处理
void processYOLOv8FlatOutput(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh) {
    const int num_boxes = output.size[0];
    const int dimensions = output.size[1];
    const int num_classes = dimensions - 5;

    for (int i = 0; i < num_boxes; i++) {
        const float* data_ptr = output.ptr<float>(i);
        float objectness = data_ptr[4];

        if (objectness < conf_threshold) continue;

        // 查找最高类别得分
        float max_class_score = 0;
        int max_class_id = 0;
        for (int j = 0; j < num_classes; j++) {
            float class_score = data_ptr[5 + j];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                max_class_id = j;
            }
        }

        float confidence = objectness * max_class_score;
        if (confidence < conf_threshold) continue;

        // 提取边界框坐标
        float x = data_ptr[0];
        float y = data_ptr[1];
        float w = data_ptr[2];
        float h = data_ptr[3];

        // 将归一化坐标转为像素坐标
        float input_size = 640; // 假设输入尺寸为640
        x *= input_size;
        y *= input_size;
        w *= input_size;
        h *= input_size;

        // 从letterbox坐标转回原图
        x = (x - dw) / scale;
        y = (y - dh) / scale;
        w = w / scale;
        h = h / scale;

        // 转为矩形坐标
        int left = std::max(0, int(x - w / 2));
        int top = std::max(0, int(y - h / 2));
        int right = std::min(img_width, int(x + w / 2));
        int bottom = std::min(img_height, int(y + h / 2));

        // 创建矩形
        cv::Rect rect(left, top, right - left, bottom - top);
        if (rect.width <= 0 || rect.height <= 0) continue;

        boxes.push_back(rect);
        confidences.push_back(confidence);
        class_ids.push_back(max_class_id);
    }
}

// YOLOv11特殊输出处理
void processYOLOv11Output(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh) {
    // 尝试安全地解析YOLOv11输出
    try {
        // 假设格式为 [batch, ?, num_predictions, dimensions]
        std::cout << "尝试处理4维YOLOv11输出" << std::endl;

        // 由于不确定确切格式，我们将尝试几种可能的访问模式
        const int dim0 = output.size[0]; // 通常是batch_size=1
        const int dim1 = output.size[1]; // 可能是头数或特征图尺寸
        const int dim2 = output.size[2]; // 可能是预测数
        const int dim3 = output.size[3]; // 可能是每个预测的特征数

        std::cout << "输出维度: " << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 << std::endl;

        // 假设dim3包含[x,y,w,h,obj,classes...]
        if (dim3 >= 5 + 1) { // 假设至少有一个类别
            // 遍历dim1和dim2
            for (int i = 0; i < dim1; i++) {
                for (int j = 0; j < std::min(100, dim2); j++) { // 限制处理的数量
                    try {
                        const float* data_ptr = output.ptr<float>(0, i, j);

                        // 提取置信度
                        float objectness = data_ptr[4];
                        if (objectness < conf_threshold) continue;

                        // 查找最高类别得分
                        float max_class_score = 0;
                        int max_class_id = 0;
                        for (int k = 0; k < std::min(10, dim3 - 5); k++) { // 限制类别数
                            float class_score = data_ptr[5 + k];
                            if (class_score > max_class_score) {
                                max_class_score = class_score;
                                max_class_id = k;
                            }
                        }

                        float confidence = objectness * max_class_score;
                        if (confidence < conf_threshold) continue;

                        // 提取边界框坐标
                        float x = data_ptr[0];
                        float y = data_ptr[1];
                        float w = data_ptr[2];
                        float h = data_ptr[3];

                        // 检查坐标是否合理（防止异常值）
                        if (x < 0 || x > 1 || y < 0 || y > 1 || w < 0 || w > 1 || h < 0 || h > 1) {
                            continue;
                        }

                        // 将归一化坐标转为像素坐标
                        float input_size = 640;
                        x *= input_size;
                        y *= input_size;
                        w *= input_size;
                        h *= input_size;

                        // 从letterbox坐标转回原图
                        x = (x - dw) / scale;
                        y = (y - dh) / scale;
                        w = w / scale;
                        h = h / scale;

                        // 转为矩形坐标
                        int left = std::max(0, int(x - w / 2));
                        int top = std::max(0, int(y - h / 2));
                        int right = std::min(img_width, int(x + w / 2));
                        int bottom = std::min(img_height, int(y + h / 2));

                        if (right <= left || bottom <= top) continue;

                        cv::Rect rect(left, top, right - left, bottom - top);
                        boxes.push_back(rect);
                        confidences.push_back(confidence);
                        class_ids.push_back(max_class_id);

                        std::cout << "检测到物体: 置信度=" << confidence
                            << ", 坐标=[" << left << "," << top << ","
                            << right - left << "," << bottom - top << "]" << std::endl;
                    }
                    catch (const cv::Exception& e) {
                        std::cerr << "处理YOLOv11预测时出错: " << e.what() << std::endl;
                        continue;
                    }
                }
            }
        }
    }
    catch (const cv::Exception& e) {
        std::cerr << "处理YOLOv11输出时出错: " << e.what() << std::endl;
    }
}

// 新增：处理YOLOv11转置输出 [1, 5, num_predictions]
void processYOLOv11TransposedOutput(const cv::Mat& output, std::vector<cv::Rect>& boxes,
    std::vector<float>& confidences, std::vector<int>& class_ids,
    float conf_threshold, int img_width, int img_height,
    float scale, int dw, int dh, int num_classes) {
    // 获取输出维度
    const int num_predictions = output.size[2]; // 预测框数量
    const int channels = output.size[1]; // 通道数，应为5

    std::cout << "处理转置输出：预测框数量=" << num_predictions << ", 通道数=" << channels << std::endl;

    // 获取指向数据的指针
    const float* data = (float*)output.data;

    // 遍历所有预测框
    for (int i = 0; i < num_predictions; i++) {
        // YOLOv11转置输出中，数据排列为[1, 5, 8400]
        // 即每个预测框的数据在不同的通道中
        // 需要获取第i个预测框的所有通道数据

        // 首先获取objectness得分（第5个通道）
        float objectness = data[4 * num_predictions + i]; // 获取第5通道的第i个预测

        if (objectness < conf_threshold) continue;

        // 在只有一个类别的情况下，我们直接使用objectness作为最终置信度
        float confidence = objectness;
        int class_id = 0; // 默认类别为0

        // 如果有多个类别，这里应该遍历类别通道并找到最高得分
        // 但根据输出形状，这里似乎只有5个通道(x,y,w,h,obj)，没有类别通道
        // 如果需要处理多类别，可能需要另一个输出层或不同的处理逻辑

        // 提取边界框坐标（从前4个通道）
        float x = data[0 * num_predictions + i]; // 第1通道的第i个预测
        float y = data[1 * num_predictions + i]; // 第2通道的第i个预测
        float w = data[2 * num_predictions + i]; // 第3通道的第i个预测
        float h = data[3 * num_predictions + i]; // 第4通道的第i个预测

        // 打印第一个检测框的数据，用于调试
        if (i == 0) {
            std::cout << "首个检测框: x=" << x << ", y=" << y << ", w=" << w
                << ", h=" << h << ", conf=" << objectness << std::endl;
        }

        // 检查坐标是否合理（防止异常值）
        if (x < 0 || x > 1 || y < 0 || y > 1 || w < 0 || w > 1 || h < 0 || h > 1) {
            // 如果坐标超出范围，则可能不是归一化坐标
            // 尝试将绝对坐标转换为归一化坐标
            if (x > 1 && y > 1 && w > 1 && h > 1) {
                float input_size = 640.0f;
                x /= input_size;
                y /= input_size;
                w /= input_size;
                h /= input_size;
            }
            else {
                continue; // 跳过这个预测框
            }
        }

        // 将归一化坐标转为像素坐标
        float input_size = 640.0f;
        x *= input_size;
        y *= input_size;
        w *= input_size;
        h *= input_size;

        // 从letterbox坐标转回原图
        x = (x - dw) / scale;
        y = (y - dh) / scale;
        w = w / scale;
        h = h / scale;

        // 转为矩形坐标
        int left = std::max(0, int(x - w / 2));
        int top = std::max(0, int(y - h / 2));
        int right = std::min(img_width, int(x + w / 2));
        int bottom = std::min(img_height, int(y + h / 2));

        if (right <= left || bottom <= top) continue;

        cv::Rect rect(left, top, right - left, bottom - top);
        boxes.push_back(rect);
        confidences.push_back(confidence);
        class_ids.push_back(class_id);

        std::cout << "检测到物体 #" << i << ": 置信度=" << confidence
            << ", 坐标=[" << left << "," << top << ","
            << right - left << "," << bottom - top << "]" << std::endl;
    }

    std::cout << "转置输出处理完成，检测到 " << boxes.size() << " 个物体" << std::endl;
}

int main() {
    // 配置参数
    const std::string image_dir = "E:/opencvProject/LYJ/ultralytics-main/inference/20250319_MLD_det";
    const int expected_screws = 2;  // 每个检测区域的预期螺丝数量

    try {
        detection(image_dir, expected_screws);
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}