#include <algorithm>
#include <vector>

#include "caffe/layers/random_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandomLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  dist = boost::random::uniform_int_distribution<>(-1000, 1000);
  return;
}

template <typename Dtype>
void RandomLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data(); // Incoming
  Dtype *top_data = top[0]->mutable_cpu_data();     // Outgoing
  const int count = bottom[0]->count();             // Number of values
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] + dist(rng)/1000;
  }
}

template <typename Dtype>
void RandomLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      // Just copy the incoming gradient since we only add random values
      bottom_diff[i] = top_diff[i];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RandomLayer);
#endif

INSTANTIATE_CLASS(RandomLayer);
REGISTER_LAYER_CLASS(Random);
} // namespace caffe
