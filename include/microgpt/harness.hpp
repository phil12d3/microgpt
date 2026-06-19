#pragma once

#include <functional>
#include <map>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {

struct HarnessIO {
  std::ostream& out;
  std::ostream& err;

  HarnessIO(std::ostream& o = std::cout, std::ostream& e = std::cerr) : out(o), err(e) {}
};

using HarnessFn = std::function<int(const std::vector<std::string>&, HarnessIO&)>;

class HarnessRegistry {
 public:
  void register_tool(std::string name, HarnessFn fn) { tools_[std::move(name)] = std::move(fn); }

  int run(const std::string& name, const std::vector<std::string>& args, HarnessIO& io) const {
    auto it = tools_.find(name);
    if (it == tools_.end()) {
      throw std::runtime_error("unknown tool: " + name);
    }
    return it->second(args, io);
  }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(tools_.size());
    for (const auto& kv : tools_) {
      out.push_back(kv.first);
    }
    return out;
  }

 private:
  std::map<std::string, HarnessFn> tools_;
};

}  // namespace microgpt
