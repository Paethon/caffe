#include <vector>
#include <random>

using namespace std;

template <typename T> class RandomElement {
private:
  const vector<T> &_vec;
  default_random_engine _rnd;
  uniform_int_distribution<int> _dist;

public:
  RandomElement(const vector<T> &vec) : _vec(vec) {

    random_device r;
    _rnd = default_random_engine(r());
    _dist = uniform_int_distribution<int>(0, vec.size() - 1);
  }

  /// Return reference to random element from vector
  const T &get() { return _vec[_dist(_rnd)]; }
};
