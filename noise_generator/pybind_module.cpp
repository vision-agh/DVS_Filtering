#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "noise_cpp/event2d.h"
#include "noise_cpp/noise_generator_algorithm.h"

namespace py = pybind11;
using namespace Metavision;

PYBIND11_MODULE(noise_generator_py, m) {
    py::class_<Event2d>(m, "Event2d")
        .def(py::init<unsigned short, unsigned short, short, timestamp>(),
             py::arg("x"), py::arg("y"), py::arg("p"), py::arg("t"))
        .def_readwrite("x", &Event2d::x)
        .def_readwrite("y", &Event2d::y)
        .def_readwrite("p", &Event2d::p)
        .def_readwrite("t", &Event2d::t)
        .def("__repr__", [](const Event2d &e) {
            return std::string("Event2d(x=") + std::to_string(e.x) + ", y=" + std::to_string(e.y) + ", p=" + std::to_string(e.p) + ", t=" + std::to_string(e.t) + ")";
        });

    py::class_<NoiseGeneratorAlgorithm>(m, "NoiseGeneratorAlgorithm")
        .def(py::init<std::uint32_t, std::uint32_t, double, double, uint32_t>(),
             py::arg("width") = 1280, py::arg("height") = 720, py::arg("shot_noise_rate_hz") = 1.0, py::arg("poisson_divider") = 30.0, py::arg("timestamp_resolution_us") = 1)
        .def("process_events", [](NoiseGeneratorAlgorithm &self, const std::vector<Event2d> &input_events) {
            std::vector<Event2d> output_events;
            output_events.reserve(input_events.size() * 2);
            self.process_events(input_events.begin(), input_events.end(), std::back_inserter(output_events));
            return output_events;
        });

    m.doc() = "Pybind11 wrapper for DVS noise generator";
}