#ifndef CAFFE_RELU_LAYER_HPP_
#define CAFFE_RELU_LAYER_HPP_

#include <vector>

#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Perturbs incoming values by a random amount
 */
template <typename Dtype> class RandomLayer : public NeuronLayer<Dtype> {
public:
  explicit RandomLayer(const LayerParameter &param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Random"; }

protected:
  boost::mt19937 rng;
  boost::random::uniform_int_distribution<> dist;

  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);
};

} // namespace caffe

#endif // CAFFE_RELU_LAYER_HPP_
