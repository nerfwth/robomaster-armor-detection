#include "HybridArmorDetector.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// ============================================================
// 构造函数：加载两个 ONNX
// ============================================================

HybridArmorDetector::HybridArmorDetector(
    const std::string& poseModelPath,
    const std::string& numberModelPath)
{
    try
    {
        poseNet_ =
            cv::dnn::readNetFromONNX(
                poseModelPath
            );

        numberNet_ =
            cv::dnn::readNetFromONNX(
                numberModelPath
            );

        if (poseNet_.empty() ||
            numberNet_.empty())
        {
            std::cerr
                << "[HybridArmorDetector] model empty"
                << std::endl;

            ready_ = false;
            return;
        }

        // Ubuntu 当前先使用 OpenCV CPU
        poseNet_.setPreferableBackend(
            cv::dnn::DNN_BACKEND_OPENCV
        );

        poseNet_.setPreferableTarget(
            cv::dnn::DNN_TARGET_CPU
        );

        numberNet_.setPreferableBackend(
            cv::dnn::DNN_BACKEND_OPENCV
        );

        numberNet_.setPreferableTarget(
            cv::dnn::DNN_TARGET_CPU
        );

        ready_ = true;

        std::cout
            << "[HybridArmorDetector] models loaded successfully"
            << std::endl;
    }
    catch (const cv::Exception& e)
    {
        std::cerr
            << "[HybridArmorDetector] OpenCV error:\n"
            << e.what()
            << std::endl;

        ready_ = false;
    }
}


// ============================================================

bool HybridArmorDetector::isReady() const
{
    return ready_;
}


// ============================================================
// Letterbox 640x640
// ============================================================

cv::Mat HybridArmorDetector::letterbox(
    const cv::Mat& image,
    float& scale,
    int& padX,
    int& padY) const
{
    constexpr int inputSize = 640;

    scale =
        std::min(
            inputSize /
                static_cast<float>(image.cols),

            inputSize /
                static_cast<float>(image.rows)
        );

    int newW =
        static_cast<int>(
            std::round(
                image.cols * scale
            )
        );

    int newH =
        static_cast<int>(
            std::round(
                image.rows * scale
            )
        );

    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(newW, newH)
    );

    padX =
        (inputSize - newW) / 2;

    padY =
        (inputSize - newH) / 2;

    cv::Mat canvas(
        inputSize,
        inputSize,
        CV_8UC3,
        cv::Scalar(114, 114, 114)
    );

    resized.copyTo(
        canvas(
            cv::Rect(
                padX,
                padY,
                newW,
                newH
            )
        )
    );

    return canvas;
}


// ============================================================
// YOLOv8 Pose 解码
// 1 x 17 x 8400
// ============================================================

std::vector<HybridArmorDetector::PoseDetection>
HybridArmorDetector::decodePose(
    const cv::Mat& output,
    const cv::Mat& original,
    float scale,
    int padX,
    int padY,
    float confThreshold,
    float nmsThreshold) const
{
    std::vector<PoseDetection> result;

    if (output.dims != 3)
    {
        return result;
    }

    const int channels =
        output.size[1];

    const int numPreds =
        output.size[2];

    if (channels != 17)
    {
        std::cerr
            << "[HybridArmorDetector] unexpected pose channels: "
            << channels
            << std::endl;

        return result;
    }

    const float* data =
        reinterpret_cast<const float*>(
            output.data
        );

    auto value =
        [&](int channel, int index)
    {
        return data[
            channel * numPreds + index
        ];
    };

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    std::vector<
        std::vector<cv::Point2f>
    > keypointLists;

    for (int i = 0;
         i < numPreds;
         ++i)
    {
        float confidence =
            value(4, i);

        if (confidence <
            confThreshold)
        {
            continue;
        }

        float cx =
            value(0, i);

        float cy =
            value(1, i);

        float w =
            value(2, i);

        float h =
            value(3, i);

        float x1 =
            (cx - w * 0.5f - padX)
            / scale;

        float y1 =
            (cy - h * 0.5f - padY)
            / scale;

        float x2 =
            (cx + w * 0.5f - padX)
            / scale;

        float y2 =
            (cy + h * 0.5f - padY)
            / scale;

        x1 = std::clamp(
            x1,
            0.0f,
            static_cast<float>(
                original.cols - 1
            )
        );

        y1 = std::clamp(
            y1,
            0.0f,
            static_cast<float>(
                original.rows - 1
            )
        );

        x2 = std::clamp(
            x2,
            0.0f,
            static_cast<float>(
                original.cols - 1
            )
        );

        y2 = std::clamp(
            y2,
            0.0f,
            static_cast<float>(
                original.rows - 1
            )
        );

        if (x2 <= x1 ||
            y2 <= y1)
        {
            continue;
        }

        cv::Rect box(
            static_cast<int>(x1),
            static_cast<int>(y1),

            std::max(
                1,
                static_cast<int>(
                    x2 - x1
                )
            ),

            std::max(
                1,
                static_cast<int>(
                    y2 - y1
                )
            )
        );

        std::vector<cv::Point2f>
            keypoints;

        for (int k = 0;
             k < 4;
             ++k)
        {
            int base =
                5 + k * 3;

            float kx =
                value(base, i);

            float ky =
                value(base + 1, i);

            float ox =
                (kx - padX)
                / scale;

            float oy =
                (ky - padY)
                / scale;

            ox = std::clamp(
                ox,
                0.0f,
                static_cast<float>(
                    original.cols - 1
                )
            );

            oy = std::clamp(
                oy,
                0.0f,
                static_cast<float>(
                    original.rows - 1
                )
            );

            keypoints.emplace_back(
                ox,
                oy
            );
        }

        boxes.emplace_back(box);
        scores.emplace_back(confidence);

        keypointLists.emplace_back(
            keypoints
        );
    }

    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        scores,
        confThreshold,
        nmsThreshold,
        indices
    );

    for (int index : indices)
    {
        PoseDetection detection;

        detection.confidence =
            scores[index];

        detection.box =
            boxes[index];

        detection.keypoints =
            keypointLists[index];

        result.emplace_back(
            detection
        );
    }

    return result;
}


// ============================================================
// 透视矫正
//
// kp0 左上
// kp1 左下
// kp2 右下
// kp3 右上
// ============================================================

cv::Mat HybridArmorDetector::rectifyArmor(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& keypoints) const
{
    constexpr int W = 160;
    constexpr int H = 80;

    if (keypoints.size() != 4)
    {
        return {};
    }

    std::vector<cv::Point2f> dst =
    {
        {0.0f, 0.0f},

        {
            0.0f,
            static_cast<float>(H - 1)
        },

        {
            static_cast<float>(W - 1),
            static_cast<float>(H - 1)
        },

        {
            static_cast<float>(W - 1),
            0.0f
        }
    };

    cv::Mat M =
        cv::getPerspectiveTransform(
            keypoints,
            dst
        );

    cv::Mat warped;

    cv::warpPerspective(
        image,
        warped,
        M,
        cv::Size(W, H)
    );

    return warped;
}


// ============================================================
// RED / BLUE 判断
// ============================================================

std::pair<std::string, float>
HybridArmorDetector::classifyColor(
    const cv::Mat& warped) const
{
    if (warped.empty())
    {
        return {
            "UNKNOWN",
            0.0f
        };
    }

    const int h =
        warped.rows;

    const int w =
        warped.cols;

    const int sideWidth =
        static_cast<int>(
            w * 0.25f
        );

    if (sideWidth <= 0)
    {
        return {
            "UNKNOWN",
            0.0f
        };
    }

    cv::Mat left =
        warped(
            cv::Rect(
                0,
                0,
                sideWidth,
                h
            )
        );

    cv::Mat right =
        warped(
            cv::Rect(
                w - sideWidth,
                0,
                sideWidth,
                h
            )
        );

    long long redStrength = 0;
    long long blueStrength = 0;

    int redCount = 0;
    int blueCount = 0;

    auto processRegion =
        [&](const cv::Mat& region)
    {
        for (int y = 0;
             y < region.rows;
             ++y)
        {
            for (int x = 0;
                 x < region.cols;
                 ++x)
            {
                const cv::Vec3b& p =
                    region.at<cv::Vec3b>(
                        y,
                        x
                    );

                int b = p[0];
                int r = p[2];

                int brightness =
                    std::max(
                        r,
                        b
                    );

                if (brightness <= 80)
                {
                    continue;
                }

                int diff =
                    r - b;

                if (diff > 20)
                {
                    ++redCount;

                    redStrength +=
                        diff;
                }
                else if (diff < -20)
                {
                    ++blueCount;

                    blueStrength +=
                        -diff;
                }
            }
        }
    };

    processRegion(left);
    processRegion(right);

    double total =
        static_cast<double>(
            redStrength +
            blueStrength
        );

    if (total < 1.0)
    {
        return {
            "UNKNOWN",
            0.0f
        };
    }

    double redRatio =
        redStrength / total;

    double blueRatio =
        blueStrength / total;

    if (
        redCount >= 8 &&
        redRatio >= 0.55
    )
    {
        return {
            "RED",
            static_cast<float>(
                redRatio * 100.0
            )
        };
    }

    if (
        blueCount >= 8 &&
        blueRatio >= 0.55
    )
    {
        return {
            "BLUE",
            static_cast<float>(
                blueRatio * 100.0
            )
        };
    }

    return {
        "UNKNOWN",
        static_cast<float>(
            std::max(
                redRatio,
                blueRatio
            ) * 100.0
        )
    };
}


// ============================================================
// 构造 64x64 数字分类输入
// ============================================================

cv::Mat HybridArmorDetector::makeDigitImage(
    const cv::Mat& warped) const
{
    constexpr int W = 160;
    constexpr int H = 80;
    constexpr int OUT = 64;

    if (warped.empty())
    {
        return {};
    }

    const int x1 =
        static_cast<int>(
            W * 0.20f
        );

    const int x2 =
        static_cast<int>(
            W * 0.80f
        );

    const int y1 =
        static_cast<int>(
            H * 0.05f
        );

    const int y2 =
        static_cast<int>(
            H * 0.95f
        );

    cv::Mat crop =
        warped(
            cv::Rect(
                x1,
                y1,
                x2 - x1,
                y2 - y1
            )
        ).clone();

    cv::Mat gray;

    cv::cvtColor(
        crop,
        gray,
        cv::COLOR_BGR2GRAY
    );

    cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE(
            2.0,
            cv::Size(4, 4)
        );

    clahe->apply(
        gray,
        gray
    );

    float scale =
        std::min(
            OUT /
                static_cast<float>(
                    gray.cols
                ),

            OUT /
                static_cast<float>(
                    gray.rows
                )
        );

    int newW =
        std::max(
            1,
            static_cast<int>(
                gray.cols * scale
            )
        );

    int newH =
        std::max(
            1,
            static_cast<int>(
                gray.rows * scale
            )
        );

    cv::Mat resized;

    cv::resize(
        gray,
        resized,
        cv::Size(
            newW,
            newH
        ),
        0,
        0,
        cv::INTER_AREA
    );

    cv::Mat canvas =
        cv::Mat::zeros(
            OUT,
            OUT,
            CV_8UC1
        );

    int px =
        (OUT - newW) / 2;

    int py =
        (OUT - newH) / 2;

    resized.copyTo(
        canvas(
            cv::Rect(
                px,
                py,
                newW,
                newH
            )
        )
    );

    cv::Mat bgr;

    cv::cvtColor(
        canvas,
        bgr,
        cv::COLOR_GRAY2BGR
    );

    return bgr;
}


// ============================================================
// Softmax
// ============================================================

std::vector<float>
HybridArmorDetector::softmax(
    const std::vector<float>& values)
{
    std::vector<float> result(
        values.size()
    );

    if (values.empty())
    {
        return result;
    }

    float maxValue =
        *std::max_element(
            values.begin(),
            values.end()
        );

    float sum = 0.0f;

    for (size_t i = 0;
         i < values.size();
         ++i)
    {
        result[i] =
            std::exp(
                values[i] -
                maxValue
            );

        sum += result[i];
    }

    if (sum > 0.0f)
    {
        for (float& value :
             result)
        {
            value /= sum;
        }
    }

    return result;
}


// ============================================================
// 数字分类 1~5
// ============================================================

std::pair<int, float>
HybridArmorDetector::classifyNumber(
    const cv::Mat& digitImage)
{
    if (digitImage.empty())
    {
        return {
            -1,
            0.0f
        };
    }

    cv::Mat blob =
        cv::dnn::blobFromImage(
            digitImage,
            1.0 / 255.0,
            cv::Size(64, 64),
            cv::Scalar(),
            true,
            false
        );

    numberNet_.setInput(blob);

    cv::Mat output =
        numberNet_.forward();

    cv::Mat flat =
        output.reshape(
            1,
            1
        );

    if (flat.total() != 5)
    {
        std::cerr
            << "[HybridArmorDetector] number output != 5"
            << std::endl;

        return {
            -1,
            0.0f
        };
    }

    std::vector<float> raw(5);

    for (int i = 0;
         i < 5;
         ++i)
    {
        raw[i] =
            flat.at<float>(
                0,
                i
            );
    }

    bool alreadyProbability =
        true;

    float probabilitySum =
        0.0f;

    for (float value : raw)
    {
        if (
            value < 0.0f ||
            value > 1.0f
        )
        {
            alreadyProbability =
                false;
        }

        probabilitySum += value;
    }

    if (
        std::abs(
            probabilitySum - 1.0f
        ) > 0.05f
    )
    {
        alreadyProbability =
            false;
    }

    std::vector<float> probs;

    if (alreadyProbability)
    {
        probs = raw;
    }
    else
    {
        probs =
            softmax(raw);
    }

    int bestIndex = 0;

    for (int i = 1;
         i < 5;
         ++i)
    {
        if (
            probs[i] >
            probs[bestIndex]
        )
        {
            bestIndex = i;
        }
    }

    return {
        bestIndex + 1,
        probs[bestIndex]
    };
}


// ============================================================
// 绘制
// ============================================================

void HybridArmorDetector::drawResult(
    cv::Mat& frame,
    const HybridArmorResult& result) const
{
    cv::Scalar color;

    if (result.color == "RED")
    {
        color =
            cv::Scalar(
                0,
                0,
                255
            );
    }
    else if (
        result.color == "BLUE"
    )
    {
        color =
            cv::Scalar(
                255,
                0,
                0
            );
    }
    else
    {
        color =
            cv::Scalar(
                0,
                255,
                0
            );
    }

    cv::rectangle(
        frame,
        result.box,
        color,
        2
    );

    for (size_t i = 0;
         i < result.keypoints.size();
         ++i)
    {
        cv::circle(
            frame,
            result.keypoints[i],
            5,
            cv::Scalar(
                0,
                255,
                255
            ),
            -1
        );

        cv::putText(
            frame,
            std::to_string(i),
            result.keypoints[i]
                + cv::Point2f(
                    4,
                    -4
                ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(
                0,
                255,
                255
            ),
            1
        );
    }

    if (result.keypoints.size() == 4)
    {
        for (int i = 0;
             i < 4;
             ++i)
        {
            cv::line(
                frame,
                result.keypoints[i],
                result.keypoints[
                    (i + 1) % 4
                ],
                color,
                2
            );
        }
    }

    std::string label =
        result.source
        + " "
        + result.color
        + " "
        + std::to_string(
            result.number
        )
        + " | armor "
        + cv::format(
            "%.2f",
            result.armor_confidence
        )
        + " | num "
        + cv::format(
            "%.2f",
            result.number_confidence
        );

    cv::putText(
        frame,
        label,
        cv::Point(
            result.box.x,
            std::max(
                25,
                result.box.y - 8
            )
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    );
}


// ============================================================
// 最终 detect()
// ============================================================

std::vector<HybridArmorResult>
HybridArmorDetector::detect(
    cv::Mat& frame,
    bool draw)
{
    std::vector<HybridArmorResult>
        results;

    if (
        !ready_ ||
        frame.empty()
    )
    {
        return results;
    }

    int64 totalStart =
        cv::getTickCount();

    // ========================================================
    // YOLO Pose
    // ========================================================

    float scale = 1.0f;
    int padX = 0;
    int padY = 0;

    cv::Mat input =
        letterbox(
            frame,
            scale,
            padX,
            padY
        );

    cv::Mat blob =
        cv::dnn::blobFromImage(
            input,
            1.0 / 255.0,
            cv::Size(640, 640),
            cv::Scalar(),
            true,
            false
        );

    poseNet_.setInput(blob);

    int64 poseStart =
        cv::getTickCount();

    cv::Mat poseOutput =
        poseNet_.forward();

    lastPoseMs_ =
        (
            cv::getTickCount() -
            poseStart
        )
        * 1000.0
        / cv::getTickFrequency();

    std::vector<PoseDetection>
        poseDetections =
            decodePose(
                poseOutput,
                frame,
                scale,
                padX,
                padY
            );

    bool usedFallback =
        poseDetections.empty();

    // ========================================================
    // YOLO 完全漏检 -> OpenCV fallback
    // ========================================================

    if (usedFallback)
    {
        auto fallbackResults =
            fallbackDetector_.detect(
                frame
            );

        for (const auto& fallback :
             fallbackResults)
        {
            PoseDetection detection;

            detection.confidence =
                fallback.score;

            detection.box =
                fallback.box;

            detection.keypoints.assign(
                fallback.keypoints.begin(),
                fallback.keypoints.end()
            );

            poseDetections.emplace_back(
                detection
            );
        }
    }

    // ========================================================
    // 对检测结果做颜色 + 数字
    // ========================================================

    for (const auto& detection :
         poseDetections)
    {
        cv::Mat warped =
            rectifyArmor(
                frame,
                detection.keypoints
            );

        if (warped.empty())
        {
            continue;
        }

        auto colorResult =
            classifyColor(
                warped
            );

        cv::Mat digitImage =
            makeDigitImage(
                warped
            );

        auto numberResult =
            classifyNumber(
                digitImage
            );

        HybridArmorResult result;

        result.source =
            usedFallback
            ? "CV"
            : "YOLO";

        result.color =
            colorResult.first;

        result.color_score =
            colorResult.second;

        result.number =
            numberResult.first;

        result.number_confidence =
            numberResult.second;

        result.armor_confidence =
            detection.confidence;

        result.box =
            detection.box;

        result.keypoints =
            detection.keypoints;

        results.emplace_back(
            result
        );
    }

    // ========================================================
    // 总耗时 / FPS
    // ========================================================

    lastTotalMs_ =
        (
            cv::getTickCount() -
            totalStart
        )
        * 1000.0
        / cv::getTickFrequency();

    lastFps_ =
        lastTotalMs_ > 0.0
        ? 1000.0 /
            lastTotalMs_
        : 0.0;

    if (draw)
    {
        for (const auto& result :
             results)
        {
            drawResult(
                frame,
                result
            );
        }

        cv::putText(
            frame,
            "FPS: "
            + cv::format(
                "%.1f",
                lastFps_
            ),
            cv::Point(
                20,
                35
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(
                0,
                255,
                0
            ),
            2
        );
    }

    return results;
}


// ============================================================

double HybridArmorDetector::lastFps() const
{
    return lastFps_;
}

double HybridArmorDetector::lastTotalMs() const
{
    return lastTotalMs_;
}

double HybridArmorDetector::lastPoseMs() const
{
    return lastPoseMs_;
}
