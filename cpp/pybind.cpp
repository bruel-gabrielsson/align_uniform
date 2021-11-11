#include <torch/extension.h>

#include "mst.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("MST", &MST, "MST");
}
