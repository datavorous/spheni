#pragma once

#include <memory>
#include <span>
#include <vector>

namespace spheni {

enum class Metric { Cosine, L2 };
enum class IndexKind { Flat, IVF };

struct IndexSpec {
  int dim = 0;
  bool normalize = true;
  int nlist = 0;
  Metric metric = Metric::Cosine;
  IndexKind kind = IndexKind::Flat;
};

struct SearchParams {
  int k = 10;
  int nprobe = 1;
};

struct SearchHit {
  long long id;
  float score;
};

class Index {
public:
  virtual ~Index() = default;
  virtual void add(std::span<const long long> ids,
                   std::span<const float> vectors) = 0;
  virtual std::vector<SearchHit> search(std::span<const float> query,
                                        const SearchParams &params) const = 0;
  virtual void train() {}
  virtual const IndexSpec &spec() const = 0;
  virtual long long size() const = 0;
  virtual int dim() const = 0;
};
std::unique_ptr<Index> make_index(const IndexSpec &spec);
} // namespace spheni
