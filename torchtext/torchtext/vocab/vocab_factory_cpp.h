#pragma once
#include <pybind11/pybind11.h>
#include "vocab.h"

namespace py = pybind11;

namespace vocab_ext {

Vocab _build_vocab_from_text_file_using_python_tokenizer(
    const std::string& file_path,
    const int64_t min_freq,
    py::object tokenizer);

Vocab _load_vocab_from_file(
    const std::string& file_path,
    const int64_t min_freq,
    const int64_t num_cpus);

Vocab _build_vocab_from_text_file(
    const std::string& file_path,
    const int64_t min_freq,
    const int64_t num_cpus,
    torch::jit::script::Module tokenizer);

}  // namespace vocab_ext
