#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "spheni/engine.h"
#include "spheni/spheni.h"

namespace py = pybind11;

namespace {
void ensure_float32(const py::array& array) {
    if (!array.dtype().is(py::dtype::of<float>())) {
        throw py::type_error("Expected numpy.ndarray of dtype float32");
    }
}

void ensure_int64(const py::array& array) {
    if (!array.dtype().is(py::dtype::of<long long>())) {
        throw py::type_error("Expected numpy.ndarray of dtype int64");
    }
}

std::span<const float> as_vector_span(const py::array& array) {
    auto info = array.request();
    return std::span<const float>(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size));
}

std::span<const long long> as_id_span(const py::array& array) {
    auto info = array.request();
    return std::span<const long long>(static_cast<const long long*>(info.ptr), static_cast<size_t>(info.size));
}
}

PYBIND11_MODULE(spheni, m) {
    m.doc() = "spheni python bindings";

    py::enum_<spheni::Metric>(m, "Metric")
        .value("Cosine", spheni::Metric::Cosine)
        .value("L2", spheni::Metric::L2);

    py::enum_<spheni::IndexKind>(m, "IndexKind")
        .value("Flat", spheni::IndexKind::Flat)
        .value("IVF", spheni::IndexKind::IVF);

    py::enum_<spheni::StorageType>(m, "StorageType")
        .value("F32", spheni::StorageType::F32)
        .value("INT8", spheni::StorageType::INT8);

    py::class_<spheni::IndexSpec>(m, "IndexSpec")
        .def(py::init<int, spheni::Metric, spheni::IndexKind, bool>(),
             py::arg("dim"), py::arg("metric"), py::arg("kind"), py::arg("normalize") = true)
        .def(py::init<int, spheni::Metric, spheni::IndexKind, spheni::StorageType, bool>(),
             py::arg("dim"), py::arg("metric"), py::arg("kind"), py::arg("storage"), py::arg("normalize") = true)
        .def(py::init<int, spheni::Metric, spheni::IndexKind, int, bool>(),
             py::arg("dim"), py::arg("metric"), py::arg("kind"), py::arg("nlist"), py::arg("normalize") = true)
        .def(py::init<int, spheni::Metric, spheni::IndexKind, int, spheni::StorageType, bool>(),
             py::arg("dim"), py::arg("metric"), py::arg("kind"), py::arg("nlist"), py::arg("storage"), py::arg("normalize") = true)
        .def_readwrite("dim", &spheni::IndexSpec::dim)
        .def_readwrite("metric", &spheni::IndexSpec::metric)
        .def_readwrite("normalize", &spheni::IndexSpec::normalize)
        .def_readwrite("kind", &spheni::IndexSpec::kind)
        .def_readwrite("storage", &spheni::IndexSpec::storage)
        .def_readwrite("nlist", &spheni::IndexSpec::nlist);

    py::class_<spheni::SearchParams>(m, "SearchParams")
        .def(py::init<int>(), py::arg("k"))
        .def(py::init<int, int>(), py::arg("k"), py::arg("nprobe"))
        .def_readwrite("k", &spheni::SearchParams::k)
        .def_readwrite("nprobe", &spheni::SearchParams::nprobe);

    py::class_<spheni::SearchHit>(m, "SearchHit")
        .def(py::init<long long, float>(), py::arg("id"), py::arg("score"))
        .def_readwrite("id", &spheni::SearchHit::id)
        .def_readwrite("score", &spheni::SearchHit::score);

    py::class_<spheni::Engine>(m, "Engine")
        .def(py::init<const spheni::IndexSpec&>(), py::arg("spec"))
        .def("add",
             [](spheni::Engine& self, const py::array_t<float, py::array::c_style>& vectors) {
                 ensure_float32(vectors);
                 if (vectors.ndim() != 2) {
                     throw py::value_error("vectors must be a 2D array");
                 }
                 if (vectors.shape(1) != self.dim()) {
                     throw py::value_error("vectors second dimension must match index dim");
                 }
                 self.add(as_vector_span(vectors));
             },
             py::arg("vectors"))
        .def("add",
             [](spheni::Engine& self,
                const py::array_t<long long, py::array::c_style>& ids,
                const py::array_t<float, py::array::c_style>& vectors) {
                 ensure_int64(ids);
                 ensure_float32(vectors);
                 if (ids.ndim() != 1) {
                     throw py::value_error("ids must be a 1D array");
                 }
                 if (vectors.ndim() != 2) {
                     throw py::value_error("vectors must be a 2D array");
                 }
                 if (vectors.shape(1) != self.dim()) {
                     throw py::value_error("vectors second dimension must match index dim");
                 }
                 if (vectors.shape(0) != ids.shape(0)) {
                     throw py::value_error("ids and vectors must have the same length");
                 }
                 self.add(as_id_span(ids), as_vector_span(vectors));
             },
             py::arg("ids"), py::arg("vectors"))
        .def("train", &spheni::Engine::train)
        .def("search",
             [](const spheni::Engine& self, const py::array_t<float, py::array::c_style>& query, int k) {
                 ensure_float32(query);
                 if (query.ndim() != 1) {
                     throw py::value_error("query must be a 1D array");
                 }
                 if (query.shape(0) != self.dim()) {
                     throw py::value_error("query length must match index dim");
                 }
                 return self.search(as_vector_span(query), k);
             },
             py::arg("query"), py::arg("k"))
        .def("search",
             [](const spheni::Engine& self, const py::array_t<float, py::array::c_style>& query, int k, int nprobe) {
                 ensure_float32(query);
                 if (query.ndim() != 1) {
                     throw py::value_error("query must be a 1D array");
                 }
                 if (query.shape(0) != self.dim()) {
                     throw py::value_error("query length must match index dim");
                 }
                 return self.search(as_vector_span(query), k, nprobe);
             },
             py::arg("query"), py::arg("k"), py::arg("nprobe"))
        .def("search_batch",
             [](const spheni::Engine& self, const py::array_t<float, py::array::c_style>& queries, int k) {
                 ensure_float32(queries);
                 if (queries.ndim() != 2) {
                     throw py::value_error("queries must be a 2D array");
                 }
                 if (queries.shape(1) != self.dim()) {
                     throw py::value_error("queries second dimension must match index dim");
                 }
                 return self.search_batch(as_vector_span(queries), k);
             },
             py::arg("queries"), py::arg("k"))
        .def("search_batch",
             [](const spheni::Engine& self, const py::array_t<float, py::array::c_style>& queries, int k, int nprobe) {
                 ensure_float32(queries);
                 if (queries.ndim() != 2) {
                     throw py::value_error("queries must be a 2D array");
                 }
                 if (queries.shape(1) != self.dim()) {
                     throw py::value_error("queries second dimension must match index dim");
                 }
                 return self.search_batch(as_vector_span(queries), k, nprobe);
             },
             py::arg("queries"), py::arg("k"), py::arg("nprobe"))
        .def("save", &spheni::Engine::save, py::arg("path"))
        .def_static("load", &spheni::Engine::load, py::arg("path"));
}
