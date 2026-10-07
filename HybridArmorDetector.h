#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

#include "traditional_fallback.h"

struct HybridArmorResult
{
    // YOLO 或 OpenCV fallback
    std::string source;

    // RED / BLUE / UNKNOWN
    std::string color;

    // 1~5
    int number = -1;

    float armor_confidence = 0.0f;
    float number_confidence = 0.0f;
    float color_score = 0.0f;

    cv::Rect box;

    // 顺序：
    // 0 左上
    // 1 左下
    // 2 右下
    // 3 右上
    std::vector<cv::Point2f> keypoints;
};


class HybridArmorDetector
{
public:
    HybridArmorDetector(
        const std::string& poseModelPath,
        const std::string& numberModelPath
    );

    bool isReady() const;

    // 对一帧图像执行：
    //
    // YOLO Pose
    // ↓
    // 如果 YOLO 完全漏检：
    // OpenCV fallback
    // ↓
    // 颜色识别
    // ↓
    // 数字识别
    //
    // draw=true 时直接在 frame 上绘制结果
    std::vector<HybridArmorResult> detect(
        cv::Mat& frame,
        bool draw = true
    );

    double lastFps() const;

    double lastTotalMs() const;

    double lastPoseMs() const;


private:
    struct PoseDetection
    {
        float confidence = 0.0f;

        cv::Rect box;

        std::vector<cv::Point2f> keypoints;
    };


private:
    cv::dnn::Net poseNet_;
    cv::dnn::Net numberNet_;

    rmarmor::TraditionalFallback fallbackDetector_;

    bool ready_ = false;

    double lastFps_ = 0.0;
    double lastTotalMs_ = 0.0;
    double lastPoseMs_ = 0.0;


private:
    cv::Mat letterbox(
        const cv::Mat& image,
        float& scale,
        int& padX,
        int& padY
    ) const;

    std::vector<PoseDetection> decodePose(
        const cv::Mat& output,
        const cv::Mat& original,
        float scale,
        int padX,
        int padY,
        float confThreshold = 0.05f,
        float nmsThreshold = 0.45f
    ) const;

    cv::Mat rectifyArmor(
        const cv::Mat& image,
        const std::vector<cv::Point2f>& keypoints
    ) const;

    std::pair<std::string, float> classifyColor(
        const cv::Mat& warped
    ) const;

    cv::Mat makeDigitImage(
        const cv::Mat& warped
    ) const;

    std::pair<int, float> classifyNumber(
        const cv::Mat& digitImage
    );

    void drawResult(
        cv::Mat& frame,
        const HybridArmorResult& result
    ) const;

    static std::vector<float> softmax(
        const std::vector<float>& values
    );
};
