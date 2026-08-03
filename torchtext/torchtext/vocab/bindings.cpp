#include "vocab.h"
#include "vocab_factory_cpp.h"
#include <torch/extension.h>

namespace py = pybind11;
using namespace vocab_ext;

namespace {
// The scripted-Module-tokenizer path (torchtext's _build_vocab_from_text_file)
// depends on torch::jit::as_module() from an unstable, internal JIT-Python
// binding header. It SIGABRTs when compiled out-of-tree against a torch pip
// wheel (pybind11 cross-module type-identity mismatch), so it's intentionally
// not wired up here. The two stable-ABI paths below are unaffected.
Vocab build_vocab_from_text_file_py(
    const std::string& /*file_path*/,
    const int64_t /*min_freq*/,
    const int64_t /*num_cpus*/,
    py::object /*fn*/) {
  throw std::runtime_error(
      "_build_vocab_from_text_file (scripted-module tokenizer) is not supported "
      "by this standalone extension: it relies on unstable torch JIT-Python "
      "internals that crash when built out-of-tree. Use "
      "_build_vocab_from_text_file_using_python_tokenizer instead.");
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "_build_vocab_from_text_file_using_python_tokenizer",
      &_build_vocab_from_text_file_using_python_tokenizer);
  m.def("_load_vocab_from_file", &_load_vocab_from_file);
  m.def("_build_vocab_from_text_file", &build_vocab_from_text_file_py);

  py::class_<Vocab, c10::intrusive_ptr<Vocab>>(m, "Vocab")
      .def(py::init<StringList, c10::optional<int64_t>>())
      .def_readonly("itos_", &Vocab::itos_)
      .def_readonly("default_index_", &Vocab::default_index_)
      .def(
          "__contains__",
          [](c10::intrusive_ptr<Vocab>& self, const py::str& item) -> bool {
            Py_ssize_t length;
            const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
            return self->__contains__(c10::string_view{buffer, (size_t)length});
          })
      .def(
          "__getitem__",
          [](c10::intrusive_ptr<Vocab>& self, const py::str& item) -> int64_t {
            Py_ssize_t length;
            const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
            return self->__getitem__(c10::string_view{buffer, (size_t)length});
          })
      .def("insert_token", &Vocab::insert_token)
      .def("set_default_index", &Vocab::set_default_index)
      .def("get_default_index", &Vocab::get_default_index)
      .def("__len__", &Vocab::__len__)
      .def("append_token", &Vocab::append_token)
      .def("lookup_token", &Vocab::lookup_token)
      .def("lookup_tokens", &Vocab::lookup_tokens)
      .def(
          "lookup_indices",
          [](const c10::intrusive_ptr<Vocab>& self, const py::list& items) {
            std::vector<int64_t> indices(items.size());
            int64_t counter = 0;
            for (const auto& item : items) {
              Py_ssize_t length;
              const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
              indices[counter++] = self->__getitem__(c10::string_view{buffer, (size_t)length});
            }
            return indices;
          })
      .def("get_stoi", &Vocab::get_stoi)
      .def("get_itos", &Vocab::get_itos)
      .def(py::pickle(
          [](const c10::intrusive_ptr<Vocab>& self) -> VocabStates { return _serialize_vocab(self); },
          [](VocabStates states) -> c10::intrusive_ptr<Vocab> { return _deserialize_vocab(states); }));
}
