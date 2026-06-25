#include "madspace/phasespace/pdf.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

using namespace madspace;

namespace {

std::string trim(const std::string_view& str) {
    return {
        std::find_if(str.begin(), str.end(), [](auto c) { return !std::isspace(c); }),
        std::find_if(str.rbegin(), str.rend(), [](auto c) {
            return !std::isspace(c);
        }).base()
    };
}

std::vector<double> read_doubles(const std::string& str) {
    std::istringstream sstream(str);
    return {std::istream_iterator<double>(sstream), std::istream_iterator<double>{}};
}

std::vector<int> read_ints(const std::string& str) {
    std::istringstream sstream(str);
    return {std::istream_iterator<int>(sstream), std::istream_iterator<int>{}};
}

double grid_value(
    const PdfGrid& grid, std::size_t q_idx, std::size_t x_idx, std::size_t pid_idx
) {
    return grid.values.at(q_idx * grid.x.size() + x_idx).at(pid_idx);
}

double
ddx(const PdfGrid& grid, std::size_t q_idx, std::size_t x_idx, std::size_t pid_idx) {
    std::size_t x_idx_max = grid.logx.size() - 1;
    double del1 = x_idx == 0 ? 0. : grid.logx.at(x_idx) - grid.logx.at(x_idx - 1);
    double del2 =
        x_idx == x_idx_max ? 0. : grid.logx.at(x_idx + 1) - grid.logx.at(x_idx);
    if (x_idx == 0) {
        double val_mid = grid_value(grid, q_idx, x_idx, pid_idx);
        double val_high = grid_value(grid, q_idx, x_idx + 1, pid_idx);
        return (val_high - val_mid) / del2;
    } else if (x_idx == x_idx_max) {
        double val_low = grid_value(grid, q_idx, x_idx - 1, pid_idx);
        double val_mid = grid_value(grid, q_idx, x_idx, pid_idx);
        return (val_mid - val_low) / del1;
    } else {
        double val_low = grid_value(grid, q_idx, x_idx - 1, pid_idx);
        double val_mid = grid_value(grid, q_idx, x_idx, pid_idx);
        double val_high = grid_value(grid, q_idx, x_idx + 1, pid_idx);
        return 0.5 * ((val_high - val_mid) / del2 + (val_mid - val_low) / del1);
    }
}

std::array<double, 4> compute_logx_coeffs(
    const PdfGrid& grid, std::size_t q_idx, std::size_t x_idx, std::size_t pid_idx
) {
    double dlogx = grid.logx.at(x_idx + 1) - grid.logx.at(x_idx);
    double vl = grid_value(grid, q_idx, x_idx, pid_idx);
    double vh = grid_value(grid, q_idx, x_idx + 1, pid_idx);
    double vdl = ddx(grid, q_idx, x_idx, pid_idx) * dlogx;
    double vdh = ddx(grid, q_idx, x_idx + 1, pid_idx) * dlogx;
    return {vdh + vdl - 2 * vh + 2 * vl, 3 * vh - 3 * vl - 2 * vdl - vdh, vdl, vl};
}

std::array<double, 16> compute_coeffs(
    const PdfGrid& grid,
    std::size_t q_idx,
    std::size_t x_idx,
    std::size_t pid_idx,
    bool low_q,
    bool high_q
) {
    auto vl = compute_logx_coeffs(grid, q_idx, x_idx, pid_idx);
    auto vh = compute_logx_coeffs(grid, q_idx + 1, x_idx, pid_idx);
    double dlogq_0 =
        low_q ? 0. : 1. / (grid.logq2.at(q_idx) - grid.logq2.at(q_idx - 1));
    double dlogq_1 = grid.logq2.at(q_idx + 1) - grid.logq2.at(q_idx);
    double dlogq_2 =
        high_q ? 0. : 1. / (grid.logq2.at(q_idx + 2) - grid.logq2.at(q_idx + 1));
    std::array<double, 4> vdl, vdh;
    if (low_q) {
        auto vhh = compute_logx_coeffs(grid, q_idx + 2, x_idx, pid_idx);
        for (std::size_t i = 0; i < 4; ++i) {
            vdl[i] = vh[i] - vl[i];
            vdh[i] = 0.5 * (vdl[i] + (vhh[i] - vh[i]) * dlogq_1 * dlogq_2);
        }
    } else if (high_q) {
        auto vll = compute_logx_coeffs(grid, q_idx - 1, x_idx, pid_idx);
        for (std::size_t i = 0; i < 4; ++i) {
            vdh[i] = vh[i] - vl[i];
            vdl[i] = 0.5 * (vdh[i] + (vl[i] - vll[i]) * dlogq_1 * dlogq_0);
        }
    } else {
        auto vll = compute_logx_coeffs(grid, q_idx - 1, x_idx, pid_idx);
        auto vhh = compute_logx_coeffs(grid, q_idx + 2, x_idx, pid_idx);
        for (std::size_t i = 0; i < 4; ++i) {
            vdl[i] = 0.5 * (vh[i] - vl[i] + (vl[i] - vll[i]) * dlogq_1 * dlogq_0);
            vdh[i] = 0.5 * (vh[i] - vl[i] + (vhh[i] - vh[i]) * dlogq_1 * dlogq_2);
        }
    }
    return {
        vl[0],
        vl[1],
        vl[2],
        vl[3],
        vh[0],
        vh[1],
        vh[2],
        vh[3],
        vdl[0],
        vdl[1],
        vdl[2],
        vdl[3],
        vdh[0],
        vdh[1],
        vdh[2],
        vdh[3]
    };
}

void init_logq2(
    Tensor tensor,
    const std::vector<std::size_t>& region_sizes,
    const std::vector<double> logq2
) {
    // TODO: check shapes and device
    auto tensor_view = tensor.view<double, 2>()[0];
    std::size_t index_out = 1, index_in = 0;
    tensor_view[0] = -std::numeric_limits<double>::max();
    for (std::size_t region_size : region_sizes) {
        for (std::size_t i = index_in == 0 ? 0 : 1; i < region_size + 1;
             ++i, ++index_in, ++index_out) {
            tensor_view[index_out] = logq2.at(index_in);
        }
        ++index_in;
    }
    tensor_view[index_out] = std::numeric_limits<double>::max();
}

} // namespace

PdfGrid::PdfGrid(const std::string& file) {
    std::ifstream grid_file(file);
    if (!grid_file) {
        throw std::runtime_error(std::format("could not open file '{}'", file));
    }
    int line_type = 0;
    std::size_t expected_value_count = 0, x_index = 0, q_index = 0, q_start = 0;
    for (std::string line; std::getline(grid_file, line);) {
        // skip comments and remove leading and trailing spaces
        line = trim(line);
        if (line.find("#") == 0 || line == "") {
            continue;
        }

        switch (line_type) {
        case 0:
            if (line == "---") {
                line_type = 1;
            }
            break;
        case 1: {
            std::vector<double> new_x = read_doubles(line);
            if (logx.size() == 0) {
                x = new_x;
            } else if (x != new_x) {
                throw std::invalid_argument(
                    "x values for different q regions must be equal"
                );
            }
            line_type = 2;
            break;
        }
        case 2: {
            std::vector<double> new_q = read_doubles(line);
            if (q.size() != 0 && q.back() != new_q.at(0)) {
                throw std::invalid_argument("q regions must connect seamlessly");
            }
            q_start = q.size();
            q.insert(q.end(), new_q.begin(), new_q.end());
            region_sizes.push_back(new_q.size() - 1);
            line_type = 3;
            break;
        }
        case 3: {
            std::vector<int> new_pids = read_ints(line);
            if (pids.size() == 0) {
                pids = new_pids;
            } else if (pids != new_pids) {
                throw std::invalid_argument(
                    "particle ids for different q regions must be equal"
                );
            }
            expected_value_count = x.size() * (region_sizes.back() + 1);
            values.resize(values.size() + expected_value_count);
            x_index = 0;
            q_index = q_start;
            line_type = 4;
            break;
        }
        case 4:
            if (expected_value_count == 0) {
                if (line == "---") {
                    line_type = 1;
                } else {
                    throw std::invalid_argument("expected end of file or next section");
                }
            } else {
                std::vector<double> new_values = read_doubles(line);
                if (new_values.size() != pids.size()) {
                    throw std::invalid_argument(
                        "exactly one grid value must be given for every PID"
                    );
                }
                values.at(q_index * x.size() + x_index) = new_values;
                ++q_index;
                if (q_index == q.size()) {
                    q_index = q_start;
                    ++x_index;
                }
                --expected_value_count;
            }
            break;
        }
    }
    if (expected_value_count != 0) {
        throw std::invalid_argument("expected more grid values");
    }
    logx.resize(x.size());
    std::transform(x.begin(), x.end(), logx.begin(), [](auto xi) {
        return std::log(xi);
    });
    logq2.resize(q.size());
    std::transform(q.begin(), q.end(), logq2.begin(), [](auto qi) {
        return 2. * std::log(qi);
    });
}

std::size_t PdfGrid::grid_point_count() const {
    // account for padding in x and q, and duplicates in q at region boundaries
    return (q_count() + 1) * (x.size() + 1);
}

std::size_t PdfGrid::q_count() const { return q.size() - region_sizes.size() + 1; }

void PdfGrid::initialize_coefficients(Tensor tensor) const {
    // TODO: check shapes and device
    tensor.zero();
    auto tensor_view = tensor.view<double, 4>()[0];
    std::size_t node_idx = x.size() + 2, q_idx = 0;
    for (std::size_t region_size : region_sizes) {
        for (std::size_t region_idx = 0; region_idx < region_size;
             ++region_idx, ++q_idx) {
            for (std::size_t x_idx = 0; x_idx < x.size() - 1; ++x_idx, ++node_idx) {
                for (std::size_t pid_idx = 0; pid_idx < pids.size(); ++pid_idx) {
                    auto coeffs = compute_coeffs(
                        *this,
                        q_idx,
                        x_idx,
                        pid_idx,
                        region_idx == 0,
                        region_idx == region_size - 1
                    );
                    for (std::size_t coeff_idx = 0; coeff_idx < coeffs.size();
                         ++coeff_idx) {
                        tensor_view[coeff_idx][pid_idx][node_idx] =
                            coeffs.at(coeff_idx);
                    }
                }
            }
            node_idx += 2;
        }
        ++q_idx;
    }
}

void PdfGrid::initialize_logx(Tensor tensor) const {
    // TODO: check shapes and device
    auto tensor_view = tensor.view<double, 2>()[0];
    std::size_t index_out = 1;
    tensor_view[0] = -std::numeric_limits<double>::max();
    for (double val : logx) {
        tensor_view[index_out] = val;
        ++index_out;
    }
    tensor_view[index_out] = std::numeric_limits<double>::max();
}

void PdfGrid::initialize_logq2(Tensor tensor) const {
    init_logq2(tensor, region_sizes, logq2);
}

std::vector<std::size_t> PdfGrid::coefficients_shape(bool batch_dim) const {
    if (batch_dim) {
        return {1, 16, pids.size(), grid_point_count()};
    } else {
        return {16, pids.size(), grid_point_count()};
    }
}

std::vector<std::size_t> PdfGrid::logx_shape(bool batch_dim) const {
    if (batch_dim) {
        return {1, x.size() + 2};
    } else {
        return {x.size() + 2};
    }
}

std::vector<std::size_t> PdfGrid::logq2_shape(bool batch_dim) const {
    if (batch_dim) {
        return {1, q_count() + 2};
    } else {
        return {q_count() + 2};
    }
}

void PdfGrid::initialize_globals(ContextPtr context, const std::string& prefix) const {
    auto logx_tensor_global = context->define_global(
        prefixed_name(prefix, "pdf_logx"), DataType::dt_float, logx_shape()
    );
    auto q2_tensor_global = context->define_global(
        prefixed_name(prefix, "pdf_logq2"), DataType::dt_float, logq2_shape()
    );
    auto coeffs_tensor_global = context->define_global(
        prefixed_name(prefix, "pdf_coefficients"),
        DataType::dt_float,
        coefficients_shape()
    );
    bool is_cpu = context->device() == cpu_device();
    Tensor logx_tensor, q2_tensor, coeffs_tensor;
    if (is_cpu) {
        logx_tensor = logx_tensor_global;
        q2_tensor = q2_tensor_global;
        coeffs_tensor = coeffs_tensor_global;
    } else {
        logx_tensor = Tensor(DataType::dt_float, logx_shape(true));
        q2_tensor = Tensor(DataType::dt_float, logq2_shape(true));
        coeffs_tensor = Tensor(DataType::dt_float, coefficients_shape(true));
    }
    initialize_logx(logx_tensor);
    initialize_logq2(q2_tensor);
    initialize_coefficients(coeffs_tensor);
    if (!is_cpu) {
        logx_tensor_global.copy_from(logx_tensor);
        q2_tensor_global.copy_from(q2_tensor);
        coeffs_tensor_global.copy_from(coeffs_tensor);
    }
}

PartonDensity::PartonDensity(
    const PdfGrid& grid,
    const std::vector<int>& pids,
    bool dynamic_pid,
    const std::string& prefix
) :
    FunctionGenerator(
        "PartonDensity",
        [&] {
            NamedVector<Type> arg_types{{"x", batch_float}, {"q", batch_float}};
            if (dynamic_pid) {
                arg_types.push_back("flavor_index", batch_int);
            }
            return arg_types;
        }(),
        {{"pdf", dynamic_pid ? batch_float : batch_float_array(pids.size())}}
    ),
    _prefix(prefix),
    _dynamic_pid(dynamic_pid),
    _logx_shape(grid.logx_shape()),
    _logq2_shape(grid.logq2_shape()),
    _coeffs_shape(grid.coefficients_shape()) {
    for (int pid : pids) {
        auto find_pid = std::find(grid.pids.begin(), grid.pids.end(), pid);
        if (find_pid == grid.pids.end()) {
            throw std::invalid_argument(
                std::format("PID {} not found in pdf grid", pid)
            );
        }
        _pid_indices.push_back(find_pid - grid.pids.begin());
    }
}

NamedVector<Value> PartonDensity::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto x = args.at(0);
    auto q = args.at(1);
    auto grid_logx = fb.global(
        prefixed_name(_prefix, "pdf_logx"),
        DataType::dt_float,
        {_logx_shape.begin(), _logx_shape.end()}
    );
    auto grid_logq2 = fb.global(
        prefixed_name(_prefix, "pdf_logq2"),
        DataType::dt_float,
        {_logq2_shape.begin(), _logq2_shape.end()}
    );
    auto grid_coeffs = fb.global(
        prefixed_name(_prefix, "pdf_coefficients"),
        DataType::dt_float,
        {_coeffs_shape.begin(), _coeffs_shape.end()}
    );
    if (_dynamic_pid) {
        // TODO: stack/unstack always copy. add instructions to avoid that
        auto indices = fb.unsqueeze(fb.gather_int(args.at(2), _pid_indices));
        auto pdf =
            fb.interpolate_pdf(x, q, indices, grid_logx, grid_logq2, grid_coeffs);
        return {{"pdf", fb.squeeze(pdf)}};
    } else {
        return {
            {"pdf",
             fb.interpolate_pdf(x, q, _pid_indices, grid_logx, grid_logq2, grid_coeffs)}
        };
    }
}

AlphaSGrid::AlphaSGrid(const std::string& file) {
    std::ifstream grid_file(file);
    if (!grid_file) {
        throw std::runtime_error(std::format("could not open file '{}'", file));
    }
    for (std::string line; std::getline(grid_file, line);) {
        // skip comments and remove leading and trailing spaces
        line = trim(line);

        // minimalistic parser to read yaml-like data
        // does the job in practice, but makes a lot of assumptions on the
        // structure of the data
        auto colon_pos = std::find(line.begin(), line.end(), ':');
        if (colon_pos == line.end()) {
            continue;
        }
        auto key = trim({line.begin(), colon_pos});
        auto value = trim({colon_pos + 1, line.end()});
        bool is_q;
        if (key == "AlphaS_Qs") {
            is_q = true;
        } else if (key == "AlphaS_Vals") {
            is_q = false;
        } else {
            continue;
        }
        auto list_begin = std::find(value.begin(), value.end(), '[');
        if (list_begin == value.end()) {
            throw std::runtime_error("expected list of values");
        }
        line = {list_begin + 1, value.end()};
        std::stringstream all_values;
        do {
            auto list_end = std::find(line.begin(), line.end(), ']');
            if (list_end == line.end()) {
                all_values << line;
            } else {
                all_values << std::string_view{line.begin(), list_end};
                break;
            }
        } while (std::getline(grid_file, line));
        std::vector<double> val_list;
        for (auto val_range : std::views::split(all_values.str(), ',')) {
            auto val_str = trim({val_range.begin(), val_range.end()});
            if (val_str.empty()) {
                continue;
            }
            val_list.push_back(std::stod(val_str));
        }
        if (is_q) {
            q = val_list;
            region_sizes.push_back(0);
            for (auto [q1, q2] : zip(q, q | std::views::drop(1))) {
                if (q1 == q2) {
                    region_sizes.push_back(0);
                } else {
                    ++region_sizes.back();
                }
            }
            logq2.resize(q.size());
            std::transform(q.begin(), q.end(), logq2.begin(), [](auto qi) {
                return 2. * std::log(qi);
            });
        } else {
            values = val_list;
        }
    }
}

std::size_t AlphaSGrid::q_count() const { return q.size() - region_sizes.size() + 1; }

void AlphaSGrid::initialize_coefficients(Tensor tensor) const {
    // TODO: check shapes and device
    tensor.zero();
    auto tensor_view = tensor.view<double, 3>()[0];
    std::size_t grid_idx = 1, q_idx = 0;
    for (std::size_t region_size : region_sizes) {
        for (std::size_t region_idx = 0; region_idx < region_size;
             ++region_idx, ++q_idx, ++grid_idx) {
            auto diff = [&](std::size_t i) {
                return (values.at(i + 1) - values.at(i)) /
                    (logq2.at(i + 1) - logq2.at(i));
            };
            double diff0, diff1;
            if (region_idx == 0) {
                diff0 = diff(q_idx);
                diff1 = 0.5 * (diff(q_idx + 1) + diff(q_idx));
            } else if (region_idx == region_size - 1) {
                diff0 = 0.5 * (diff(q_idx) + diff(q_idx - 1));
                diff1 = diff(q_idx);
            } else {
                diff0 = 0.5 * (diff(q_idx) + diff(q_idx - 1));
                diff1 = 0.5 * (diff(q_idx + 1) + diff(q_idx));
            }
            double dlogq2 = logq2.at(q_idx + 1) - logq2.at(q_idx);
            tensor_view[0][grid_idx] = values.at(q_idx);
            tensor_view[1][grid_idx] = values.at(q_idx + 1);
            tensor_view[2][grid_idx] = diff0 * dlogq2;
            tensor_view[3][grid_idx] = diff1 * dlogq2;
        }
        ++q_idx;
    }
}

void AlphaSGrid::initialize_logq2(Tensor tensor) const {
    init_logq2(tensor, region_sizes, logq2);
}

std::vector<std::size_t> AlphaSGrid::coefficients_shape(bool batch_dim) const {
    if (batch_dim) {
        return {1, 4, q_count() + 1};
    } else {
        return {4, q_count() + 1};
    }
}

std::vector<std::size_t> AlphaSGrid::logq2_shape(bool batch_dim) const {
    if (batch_dim) {
        return {1, q_count() + 2};
    } else {
        return {q_count() + 2};
    }
}

void AlphaSGrid::initialize_globals(
    ContextPtr context, const std::string& prefix
) const {
    auto q2_tensor_global = context->define_global(
        prefixed_name(prefix, "alpha_s_logq2"), DataType::dt_float, logq2_shape()
    );
    auto coeffs_tensor_global = context->define_global(
        prefixed_name(prefix, "alpha_s_coefficients"),
        DataType::dt_float,
        coefficients_shape()
    );
    bool is_cpu = context->device() == cpu_device();
    Tensor logx_tensor, q2_tensor, coeffs_tensor;
    if (is_cpu) {
        q2_tensor = q2_tensor_global;
        coeffs_tensor = coeffs_tensor_global;
    } else {
        q2_tensor = Tensor(DataType::dt_float, logq2_shape(true));
        coeffs_tensor = Tensor(DataType::dt_float, coefficients_shape(true));
    }
    initialize_logq2(q2_tensor);
    initialize_coefficients(coeffs_tensor);
    if (!is_cpu) {
        q2_tensor_global.copy_from(q2_tensor);
        coeffs_tensor_global.copy_from(coeffs_tensor);
    }
}

RunningCoupling::RunningCoupling(const AlphaSGrid& grid, const std::string& prefix) :
    FunctionGenerator(
        "RunningCoupling", {{"q", batch_float}}, {{"alpha_s", batch_float}}
    ),
    _prefix(prefix),
    _logq2_shape(grid.logq2_shape()),
    _coeffs_shape(grid.coefficients_shape()) {}

NamedVector<Value> RunningCoupling::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto q = args.at(0);
    auto grid_logq2 = fb.global(
        prefixed_name(_prefix, "alpha_s_logq2"),
        DataType::dt_float,
        {_logq2_shape.begin(), _logq2_shape.end()}
    );
    auto grid_coeffs = fb.global(
        prefixed_name(_prefix, "alpha_s_coefficients"),
        DataType::dt_float,
        {_coeffs_shape.begin(), _coeffs_shape.end()}
    );
    return {{"alpha_s", fb.interpolate_alpha_s(q, grid_logq2, grid_coeffs)}};
}
