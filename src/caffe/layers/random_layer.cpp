#include <algorithm>
#include <vector>

#include "caffe/layers/random_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandomLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data(); // Incoming
  Dtype *top_data = top[0]->mutable_cpu_data();     // Outgoing
  const int count = bottom[0]->count();             // Number of values
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void RandomLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *bottom_data = bottom[0]->cpu_data();
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      // If incoming value is negative set gradient to zero
      bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RandomLayer);
#endif

INSTANTIATE_CLASS(RandomLayer);
} // namespace caffe
