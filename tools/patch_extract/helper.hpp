#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>

namespace blg {
cv::Mat color_code(const cv::Mat &m, int color_map = cv::COLORMAP_HOT);
void visualize(const cv::Mat &m, int color_map = cv::COLORMAP_HOT);
}

#endif /* HELPER_H */
