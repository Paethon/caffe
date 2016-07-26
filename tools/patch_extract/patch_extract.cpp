#include <iostream>
#include <vector>
#include <dirent.h>
#include <random>
#include <iomanip>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include "random_element.hpp"

using namespace std;

/// Returns all files in a specified path
vector<string> getFilesInDirectory(string folderPath) {
  vector<string> ret;
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(folderPath.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      ret.push_back(string{ent->d_name});
    }
    closedir(dir);
  }

  return ret;
}

/// Returns a filtered version of a list of strings according to a pattern
vector<string> filterPattern(vector<string> list, string pattern) {
  vector<string> ret;

  for (const auto &e : list)
    if (e.find(pattern) != string::npos)
      ret.push_back(e);

  return ret;
}

/// Converts int to string padded with zeros
string zero_pad(int i, int padding) {
  stringstream ss;
  ss << setfill('0') << setw(padding) << i;
  return ss.str();
}

/// Sets up the command line parser and parses the command line
auto parse_commandline(int argc, char *argv[]) {
  namespace po = boost::program_options;

  // Set up named options
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message");
  desc.add_options()("size", po::value<int>()->default_value(32),
                     "size of image patches");
  desc.add_options()("samples", po::value<int>()->default_value(100),
                     "number of image patch triples to extract");
  desc.add_options()("stddev", po::value<double>()->default_value(5),
                     "standard deviation used to determine second patch");
  desc.add_options()("pattern,p", po::value<string>(),
                     "pattern used to find images for "
                     "computation. Contains directory and file "
                     "extension seperated by *. e.g. "
                     "/home/test/*.png");
  desc.add_options()("out",
                     po::value<string>()->default_value("./patch_output"),
                     "directory to write the extracted patches to");

  // Set positional options
  po::positional_options_description p;
  p.add("pattern", 1);

  // Parse command line
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(desc).positional(p).run(),
      vm);
  po::notify(vm);

  if (argc == 1 || vm.count("help")) {
    cout << "Usage: " << argv[0] << " [options]" << endl << endl;
    cout << desc << endl;
    exit(EXIT_SUCCESS);
  }

  return vm;
}

int main(int argc, char *argv[]) {
  auto vm = parse_commandline(argc, argv);

  // Get the directory and file extension pattern
  vector<string> parts;
  boost::split(parts, vm["pattern"].as<string>(), boost::is_any_of("*"));
  const string &dir = parts[0];
  const string &extension = parts[1];

  // Get vector of all files in directory matching pattern
  auto files = filterPattern(getFilesInDirectory(dir), extension);
  RandomElement<string> rnd_file(files);

  // Get all the needed parameters
  auto size = vm["size"].as<int>();
  auto samples = vm["samples"].as<int>();
  auto stddev = vm["stddev"].as<double>();

  // Set up random system
  random_device r;
  default_random_engine re(r());

  boost::progress_display progress(samples); // For the nice progress bar

  auto index_file_name1 = vm["out"].as<string>() + "/index1.txt";
  auto index_file_name2 = vm["out"].as<string>() + "/index2.txt";
  auto index_file_name3 = vm["out"].as<string>() + "/index3.txt";
  ofstream index_file1, index_file2, index_file3;
  index_file1.open(index_file_name1);
  index_file2.open(index_file_name2);
  index_file3.open(index_file_name3);

  // Extract and save all the patches
  for (int i = 0; i < samples; i++) {
    // Select two random images and load them
    auto img1 = cv::imread(dir + rnd_file.get());
    auto img2 = cv::imread(dir + rnd_file.get());
    // Determine two random patch positions for the two images
    int patch1_x = uniform_int_distribution<>(
        0 + size / 2, img1.size().width - size / 2 - 1)(re);
    int patch1_y = uniform_int_distribution<>(
        0 + size / 2, img1.size().height - size / 2 - 1)(re);
    int patch3_x = uniform_int_distribution<>(
        0 + size / 2, img2.size().width - size / 2 - 1)(re);
    int patch3_y = uniform_int_distribution<>(
        0 + size / 2, img2.size().height - size / 2 - 1)(re);
    // Determine second patch position for first image.
    // Normally distributed around the first patch
    int patch2_x = 0;
    while (patch2_x - size / 2 < 0 ||
           patch2_x + size / 2 >= img1.size().width) {
      normal_distribution<> d(patch1_x, stddev);
      patch2_x = round(d(re));
    }
    int patch2_y = 0;
    while (patch2_y - size / 2 < 0 ||
           patch2_y + size / 2 >= img1.size().height) {
      normal_distribution<> d(patch1_y, stddev);
      patch2_y = round(d(re));
    }
    // Extract patches
    auto patch1 =
        img1(cv::Rect(patch1_x - size / 2, patch1_y - size / 2, size, size));
    auto patch2 =
        img1(cv::Rect(patch2_x - size / 2, patch2_y - size / 2, size, size));
    auto patch3 =
        img2(cv::Rect(patch3_x - size / 2, patch3_y - size / 2, size, size));
    // Save patches
    constexpr int padding = 7;
    cv::imwrite(vm["out"].as<string>() + "/" + zero_pad(i, padding) +
                    "_patch_1.png",
                patch1);
    cv::imwrite(vm["out"].as<string>() + "/" + zero_pad(i, padding) +
                    "_patch_2.png",
                patch2);
    cv::imwrite(vm["out"].as<string>() + "/" + zero_pad(i, padding) +
                    "_patch_3.png",
                patch3);
    // Write to index file (similar similar different)
    index_file1 << zero_pad(i, padding) << "_patch_1.png 0" << std::endl;
    index_file2 << zero_pad(i, padding) << "_patch_2.png 0" << std::endl;
    index_file3 << zero_pad(i, padding) << "_patch_3.png 0" << std::endl;
    // Advance and refresh progress bar
    ++progress;
  }
  index_file1.close();
  index_file2.close();
  index_file3.close();
}
