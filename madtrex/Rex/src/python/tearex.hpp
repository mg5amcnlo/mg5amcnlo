#pragma once

#include <pybind11/pybind11.h>

// Binds the REX::tea (teaRex) types into the given module. Split into its own
// translation unit purely to mirror the Rex/teaRex source split
void bind_tearex(pybind11::module_ m);
