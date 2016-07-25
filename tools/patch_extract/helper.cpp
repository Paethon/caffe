#include "helper.hpp"
/**
 * \brief Color codes a matrix.
 *
 * The values of at given matrix are mapped to a range of 0 to 255 and
 * those values are interpreted (i.e. colored) according to a color
 * map. Useful for visualizing a.g. float matrices
 *
 * \param[in] m The matrix to be color coded \param color_map Specifies
 * the color map to be used. See <a
 * href="http://docs.opencv.org/modules/contrib/doc/facerec/colormaps.html">here</a>
 * for valid color maps.
 *
 * \return An image matrix representing the matrix m
 */
cv::Mat blg::color_code(const cv::Mat &m, int color_map) {
  double min, max;
  cv::minMaxIdx(m, &min, &max);
  std::cout << max << std::endl;
  // std::cout << fabs(min - max) << std::endl;
  auto adjMap = cv::Mat{};
  m.convertTo(adjMap, CV_8UC1, 255.0 / fabs(min - max),
              -(255 * min / fabs(min - max)));
  auto falseColorsMap = cv::Mat{};
  cv::applyColorMap(adjMap, falseColorsMap, color_map);
  return falseColorsMap;
}

/**
 * \brief Show visualization of matrix on screen
 *
 * Shows a color coded version of a matrix on the screen.
 *
 * \param[in] m Matrix to be visualized
 * \param[in] color_map Color map used for visualization
 * \return Image is displayed on screen
 */
void blg::visualize(const cv::Mat &m, int color_map) {
  namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  imshow("Display window", blg::color_code(m, color_map));
  cv::waitKey(0);
}
