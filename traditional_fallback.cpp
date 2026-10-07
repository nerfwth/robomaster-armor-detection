#include "traditional_fallback.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace rmarmor {
namespace {
constexpr float kPi = 3.14159265358979323846f;

float tiltFromVertical(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f d = b - a;
    float n = cv::norm(d);
    if (n < 1e-6f) return 90.0f;
    d *= 1.0f / n;
    float dot = std::abs(d.y);
    dot = std::clamp(dot, 0.0f, 1.0f);
    return std::acos(dot) * 180.0f / kPi;
}

cv::Rect safeRect(const cv::Rect& r, const cv::Size& s) {
    return r & cv::Rect(0, 0, s.width, s.height);
}
}

const char* TraditionalFallback::colorName(FallbackColor c) {
    if (c == FallbackColor::RED) return "RED";
    if (c == FallbackColor::BLUE) return "BLUE";
    return "UNKNOWN";
}

cv::Mat TraditionalFallback::makeColorMask(const cv::Mat& bgr,
                                            FallbackColor color) {
    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);

    cv::Mat diff;
    if (color == FallbackColor::RED) {
        cv::subtract(ch[2], ch[0], diff); // R - B, saturated at 0
    } else {
        cv::subtract(ch[0], ch[2], diff); // B - R, saturated at 0
    }

    cv::Mat colorMask;
    cv::threshold(diff, colorMask, 60, 255, cv::THRESH_BINARY);

    cv::Mat rbMax;
    cv::max(ch[0], ch[2], rbMax);
    cv::Mat brightMask;
    cv::threshold(rbMax, brightMask, 180, 255, cv::THRESH_BINARY);

    cv::Mat mask;
    cv::bitwise_and(colorMask, brightMask, mask);

    // 不做 CLOSE，避免把灯条和旁边红色物体/手部粘连。
    // 只做轻微 OPEN 去除孤立噪声。
    cv::Mat openKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(3, 3)
    );
    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_OPEN,
        openKernel
    );

    return mask;
}

bool TraditionalFallback::isLight(const LightBar& light) {
    if (light.length < 10.0f || light.width < 1.0f) return false;

    float ratio = light.width / light.length;
    if (ratio < 0.035f || ratio > 0.55f) return false;

    // Fallback is intentionally tolerant, but still rejects near-horizontal bars.
    if (light.tilt_deg > 50.0f) return false;

    return true;
}

std::vector<TraditionalFallback::LightBar>
TraditionalFallback::findLights(const cv::Mat& bgr,
                                const cv::Mat& mask,
                                FallbackColor color) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> lights;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 20.0) continue;

        cv::RotatedRect rr = cv::minAreaRect(contour);
        cv::Point2f p[4];
        rr.points(p);

        float e01 = cv::norm(p[1] - p[0]);
        float e12 = cv::norm(p[2] - p[1]);
        float length = std::max(e01, e12);
        float width = std::min(e01, e12);
        if (length < 1e-6f || width < 1e-6f) continue;

        cv::Point2f axis = (e01 >= e12) ? (p[1] - p[0]) : (p[2] - p[1]);
        float norm = cv::norm(axis);
        if (norm < 1e-6f) continue;
        axis *= 1.0f / norm;

        cv::Point2f a = rr.center - axis * (length * 0.5f);
        cv::Point2f b = rr.center + axis * (length * 0.5f);
        if (a.y > b.y) std::swap(a, b);

        LightBar light;
        light.top = a;
        light.bottom = b;
        light.center = rr.center;
        light.length = length;
        light.width = width;
        light.tilt_deg = tiltFromVertical(a, b);
        light.color = color;
        light.bbox = safeRect(cv::boundingRect(contour), bgr.size());

        if (!isLight(light)) continue;
        lights.push_back(light);
    }

    return lights;
}

bool TraditionalFallback::isArmorPair(const LightBar& a,
                                       const LightBar& b) {
    if (a.color != b.color) return false;

    float lenRatio = std::min(a.length, b.length) / std::max(a.length, b.length);
    if (lenRatio < 0.50f) return false;

    float avgLen = (a.length + b.length) * 0.5f;
    if (avgLen < 1e-6f) return false;

    cv::Point2f diff = b.center - a.center;
    float distRatio = cv::norm(diff) / avgLen;
    if (distRatio < 0.65f || distRatio > 5.8f) return false;

    float angle = std::abs(std::atan2(diff.y, diff.x)) * 180.0f / kPi;
    angle = std::min(angle, 180.0f - angle);
    if (angle > 38.0f) return false;

    float tiltDiff = std::abs(a.tilt_deg - b.tilt_deg);
    if (tiltDiff > 25.0f) return false;

    return true;
}

std::vector<FallbackDetection>
TraditionalFallback::detect(const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.channels() != 3) return {};

    cv::Mat redMask = makeColorMask(bgr, FallbackColor::RED);
    cv::Mat blueMask = makeColorMask(bgr, FallbackColor::BLUE);

    std::vector<LightBar> lights = findLights(bgr, redMask, FallbackColor::RED);
    auto blue = findLights(bgr, blueMask, FallbackColor::BLUE);
    lights.insert(lights.end(), blue.begin(), blue.end());

    std::vector<FallbackDetection> out;

    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            if (!isArmorPair(lights[i], lights[j])) continue;

            const LightBar* left = &lights[i];
            const LightBar* right = &lights[j];
            if (left->center.x > right->center.x) std::swap(left, right);

            FallbackDetection det;
            det.keypoints = {
                left->top,
                left->bottom,
                right->bottom,
                right->top
            };
            det.color = left->color;

            std::vector<cv::Point2f> pts(det.keypoints.begin(), det.keypoints.end());
            det.box = cv::boundingRect(pts);

            float lenRatio = std::min(left->length, right->length) /
                             std::max(left->length, right->length);
            float centerAngle = std::abs(std::atan2(right->center.y - left->center.y,
                                                    right->center.x - left->center.x)) *
                                180.0f / kPi;
            centerAngle = std::min(centerAngle, 180.0f - centerAngle);
            det.score = std::clamp(0.65f * lenRatio +
                                   0.35f * (1.0f - centerAngle / 38.0f),
                                   0.0f, 1.0f);

            out.push_back(det);
        }
    }

    // Prefer stronger pairs. This fallback is only used when YOLO has no detections,
    // so keeping the strongest few is sufficient and avoids cross-pair clutter.
    std::sort(out.begin(), out.end(), [](const FallbackDetection& a,
                                         const FallbackDetection& b) {
        return a.score > b.score;
    });

    if (out.size() > 4) out.resize(4);
    return out;
}

} // namespace rmarmor
