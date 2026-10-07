#pragma once

#include <array>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace rmarmor {

enum class FallbackColor { UNKNOWN = 0, RED, BLUE };

struct FallbackDetection {
    cv::Rect box{};
    // Pose-compatible order: left-top, left-bottom, right-bottom, right-top.
    std::array<cv::Point2f, 4> keypoints{};
    FallbackColor color = FallbackColor::UNKNOWN;
    float score = 0.0f;
};

class TraditionalFallback {
public:
    std::vector<FallbackDetection> detect(const cv::Mat& bgr) const;

    static const char* colorName(FallbackColor c);

private:
    struct LightBar {
        cv::Point2f top{};
        cv::Point2f bottom{};
        cv::Point2f center{};
        float length = 0.0f;
        float width = 0.0f;
        float tilt_deg = 0.0f;
        FallbackColor color = FallbackColor::UNKNOWN;
        cv::Rect bbox{};
    };

    static cv::Mat makeColorMask(const cv::Mat& bgr, FallbackColor color);
    static std::vector<LightBar> findLights(const cv::Mat& bgr,
                                            const cv::Mat& mask,
                                            FallbackColor color);
    static bool isLight(const LightBar& light);
    static bool isArmorPair(const LightBar& a, const LightBar& b);
};

} // namespace rmarmor
