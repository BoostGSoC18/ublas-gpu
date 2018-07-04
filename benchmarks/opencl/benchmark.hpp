#pragma once

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

class benchmark
{
  using clock = std::chrono::system_clock;
public:
  benchmark(std::string const &name) : name_(name) { file = std::ofstream(name.c_str());  }
  void print_header()
  {
    std::cout << "# benchmarking : " << name_ << '\n'
              << "# ..." << std::endl;
  }
  virtual void setup(long) {}
  virtual void operation(long) {}
  virtual void teardown() {}
  
  void run(std::vector<long> const &sizes)
  {
    print_header();
    for (auto s : sizes)
    {
      setup(s);
      auto start = clock::now();
      operation(s);
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start);
      teardown();
      file << s << ' ' << duration.count() << std::endl;
    }
  }
private:
  std::string name_;
  std::ofstream file;
};
