#include "madtrex/pdfware.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <set>
#include <stdexcept>

namespace madtrex {

PdfSet::PdfSet(
    madspace::ContextPtr context,
    const std::string& grid_file,
    std::vector<int> pids,
    const std::string& alpha_s_grid_file,
    const std::string& prefix
) :
    _context(std::move(context)), _label(prefix), _pids(std::move(pids)) {
    if (_pids.empty()) {
        throw std::invalid_argument("PdfSet: pids must not be empty");
    }
    for (std::size_t i = 0; i < _pids.size(); ++i) {
        _pid_index[_pids[i]] = static_cast<madspace::me_int_t>(i);
    }

    madspace::PdfGrid grid(grid_file);
    grid.initialize_globals(_context, prefix);
    _density.emplace(grid, _pids, /*dynamic_pid=*/true, prefix);
    _density_runtime =
        madspace::build_runtime(_density->function(), _context, /*concurrent=*/false);

    if (!alpha_s_grid_file.empty()) {
        madspace::AlphaSGrid alpha_s_grid(alpha_s_grid_file);
        alpha_s_grid.initialize_globals(_context, prefix);
        _alpha_s.emplace(alpha_s_grid, prefix);
        _alpha_s_runtime =
            madspace::build_runtime(_alpha_s->function(), _context, /*concurrent=*/false);
    }
}

PdfSet::PdfSet(
    madspace::ContextPtr context,
    const madspace::PdfGrid& grid,
    std::vector<int> pids,
    const madspace::AlphaSGrid* alpha_s_grid,
    const std::string& existing_globals_prefix,
    const std::string& label
) :
    _context(std::move(context)), _label(label), _pids(std::move(pids)) {
    if (_pids.empty()) {
        throw std::invalid_argument("PdfSet: pids must not be empty");
    }
    for (std::size_t i = 0; i < _pids.size(); ++i) {
        _pid_index[_pids[i]] = static_cast<madspace::me_int_t>(i);
    }

    // No initialize_globals() call here: existing_globals_prefix's globals
    // are assumed to already be registered on _context by the caller.
    _density.emplace(grid, _pids, /*dynamic_pid=*/true, existing_globals_prefix);
    _density_runtime =
        madspace::build_runtime(_density->function(), _context, /*concurrent=*/false);

    if (alpha_s_grid) {
        _alpha_s.emplace(*alpha_s_grid, existing_globals_prefix);
        _alpha_s_runtime =
            madspace::build_runtime(_alpha_s->function(), _context, /*concurrent=*/false);
    }
}

std::vector<double> PdfSet::eval_pdf(
    const std::vector<double>& x,
    const std::vector<double>& q,
    const std::vector<long int>& pdgs
) const {
    std::size_t count = x.size();
    if (q.size() != count || pdgs.size() != count) {
        throw std::runtime_error("PdfSet::eval_pdf: x/q/pdgs size mismatch");
    }
    if (count == 0) {
        return {};
    }

    // Built host-side (on the CPU device) since the values below come from a
    // plain C++ loop over host memory; moved onto the context's own device
    // afterward if that isn't the CPU (matching how Tensor::cpu() moves the
    // other way for reading results back below).
    madspace::Tensor x_tensor(madspace::DataType::dt_float64, madspace::Sizes{count});
    madspace::Tensor q_tensor(madspace::DataType::dt_float64, madspace::Sizes{count});
    madspace::Tensor idx_tensor(madspace::DataType::dt_int32, madspace::Sizes{count});
    auto x_view = x_tensor.view<double, 1>();
    auto q_view = q_tensor.view<double, 1>();
    auto idx_view = idx_tensor.view<madspace::me_int_t, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        x_view[i] = x[i];
        q_view[i] = q[i];
        auto found = _pid_index.find(pdgs[i]);
        if (found == _pid_index.end()) {
            throw std::runtime_error(std::format(
                "PdfSet::eval_pdf: PDG code {} is not among the PIDs '{}' was built for",
                pdgs[i],
                _label
            ));
        }
        idx_view[i] = found->second;
    }

    auto device = _context->device();
    if (device != madspace::cpu_device()) {
        x_tensor = x_tensor.copy(*device);
        q_tensor = q_tensor.copy(*device);
        idx_tensor = idx_tensor.copy(*device);
    }

    auto outputs = _density_runtime->run({x_tensor, q_tensor, idx_tensor});
    madspace::Tensor pdf_cpu = outputs.at(0).cpu();
    auto pdf_view = pdf_cpu.view<double, 1>();
    std::vector<double> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = pdf_view[i];
    }
    return result;
}

std::vector<double> PdfSet::eval_alpha_s(const std::vector<double>& q) const {
    if (!has_alpha_s()) {
        throw std::runtime_error(
            std::format("PdfSet::eval_alpha_s: '{}' was loaded without an alpha_s grid", _label)
        );
    }
    std::size_t count = q.size();
    if (count == 0) {
        return {};
    }
    madspace::Tensor q_tensor(madspace::DataType::dt_float64, madspace::Sizes{count});
    auto q_view = q_tensor.view<double, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        q_view[i] = q[i];
    }
    auto device = _context->device();
    if (device != madspace::cpu_device()) {
        q_tensor = q_tensor.copy(*device);
    }
    auto outputs = _alpha_s_runtime->run({q_tensor});
    madspace::Tensor alpha_s_cpu = outputs.at(0).cpu();
    auto as_view = alpha_s_cpu.view<double, 1>();
    std::vector<double> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = as_view[i];
    }
    return result;
}

namespace {

struct IncomingPartons {
    double x1, x2;
    long int pdg1, pdg2;
};

// Reads off the two incoming (status==-1) partons of every event in proc
// and converts their momenta into Bjorken x via the standard along-beam-axis
// relations for massless, collinear beams. Which of the two is "beam 1" (the
// +z-moving hadron) is decided by the sign of pz.
std::vector<IncomingPartons> extract_incoming(
    REX::process& proc, double beam1_energy, double beam2_energy
) {
    std::vector<IncomingPartons> result;
    result.reserve(proc.events.size());
    for (auto& ev_ptr : proc.events) {
        auto& ev = *ev_ptr;
        std::optional<std::size_t> idx1, idx2;
        for (std::size_t i = 0; i < ev.size(); ++i) {
            if (ev.status_[i] == -1) {
                if (!idx1) {
                    idx1 = i;
                } else if (!idx2) {
                    idx2 = i;
                }
            }
        }
        if (!idx1 || !idx2) {
            throw std::runtime_error(
                "make_pdf_weightor: event does not have exactly two incoming "
                "(status==-1) partons"
            );
        }
        auto p1 = ev.momenta_[*idx1];
        auto p2 = ev.momenta_[*idx2];
        bool first_is_beam1 = p1.z() >= p2.z();
        auto& beam1_p = first_is_beam1 ? p1 : p2;
        auto& beam2_p = first_is_beam1 ? p2 : p1;
        long int beam1_pdg = ev.pdg_[first_is_beam1 ? *idx1 : *idx2];
        long int beam2_pdg = ev.pdg_[first_is_beam1 ? *idx2 : *idx1];
        result.push_back(
            {(beam1_p.t() + beam1_p.z()) / (2.0 * beam1_energy),
             (beam2_p.t() - beam2_p.z()) / (2.0 * beam2_energy),
             beam1_pdg,
             beam2_pdg}
        );
    }
    return result;
}

} // namespace

std::shared_ptr<REX::tea::weightor> make_pdf_weightor(
    const PdfHypothesis& hyp, double beam1_energy, double beam2_energy, int alpha_s_power
) {
    if (!hyp.beam1_pdf_set || !hyp.beam2_pdf_set) {
        throw std::runtime_error("make_pdf_weightor: hypothesis has no PdfSet");
    }
    if (beam1_energy <= 0.0 || beam2_energy <= 0.0) {
        throw std::runtime_error(
            "make_pdf_weightor: beam energies must be positive; call "
            "PdfDriver::set_beam_energies(), or supply them via the LHEMeta passed to "
            "load_events()"
        );
    }

    auto beam1_pdf_set = hyp.beam1_pdf_set;
    auto beam2_pdf_set = hyp.beam2_pdf_set;
    ScaleChoice muF_scale = hyp.muF_scale;
    ScaleChoice muR_scale = hyp.muR_scale;

    return std::make_shared<REX::tea::weightor>(
        [beam1_pdf_set, beam2_pdf_set, muF_scale, muR_scale, alpha_s_power, beam1_energy,
         beam2_energy](REX::process& proc) -> std::shared_ptr<std::vector<double>> {
            auto incoming = extract_incoming(proc, beam1_energy, beam2_energy);
            std::size_t count = incoming.size();
            auto result = std::make_shared<std::vector<double>>(count, 0.0);
            if (count == 0) {
                return result;
            }

            auto& muF_vec = proc.get_muF();
            std::vector<double> x1(count), x2(count), muF(count);
            std::vector<long int> pdg1(count), pdg2(count);
            for (std::size_t i = 0; i < count; ++i) {
                x1[i] = incoming[i].x1;
                x2[i] = incoming[i].x2;
                pdg1[i] = incoming[i].pdg1;
                pdg2[i] = incoming[i].pdg2;
                muF[i] = muF_scale.resolve(muF_vec.at(i));
            }
            auto pdf1 = beam1_pdf_set->eval_pdf(x1, muF, pdg1);
            auto pdf2 = beam2_pdf_set->eval_pdf(x2, muF, pdg2);

            std::vector<double> alpha_s_vals;
            if (alpha_s_power != 0) {
                // Use beam1's alpha_s grid if available, otherwise beam2's.
                const auto& as_set = beam1_pdf_set->has_alpha_s() ? beam1_pdf_set : beam2_pdf_set;
                if (!as_set->has_alpha_s()) {
                    throw std::runtime_error(
                        "make_pdf_weightor: alpha_s_power != 0 but neither beam's PdfSet was "
                        "loaded with an alpha_s grid"
                    );
                }
                auto& muR_vec = proc.get_muR();
                std::vector<double> muR(count);
                for (std::size_t i = 0; i < count; ++i) {
                    muR[i] = muR_scale.resolve(muR_vec.at(i));
                }
                alpha_s_vals = as_set->eval_alpha_s(muR);
            }

            for (std::size_t i = 0; i < count; ++i) {
                double w = pdf1[i] * pdf2[i];
                if (alpha_s_power != 0) {
                    w *= std::pow(alpha_s_vals[i], alpha_s_power);
                }
                (*result)[i] = w;
            }
            return result;
        }
    );
}

PdfDriver::PdfDriver() : _context(madspace::default_context()) {}

PdfDriver::PdfDriver(madspace::ContextPtr context) : _context(std::move(context)) {}

PdfDriver& PdfDriver::load_pdf_set(
    const std::string& name,
    const std::string& grid_file,
    const std::string& alpha_s_grid_file,
    std::vector<int> pids
) {
    if (_pdf_sets.contains(name)) {
        throw std::runtime_error(
            std::format("PdfDriver::load_pdf_set: a PDF set named '{}' is already loaded", name)
        );
    }
    _pdf_sets[name] =
        std::make_shared<PdfSet>(_context, grid_file, std::move(pids), alpha_s_grid_file, name);
    return *this;
}

PdfDriver& PdfDriver::add_pdf_set(
    const std::string& name,
    const madspace::PdfGrid& grid,
    const madspace::AlphaSGrid* alpha_s_grid,
    std::vector<int> pids,
    const std::string& existing_globals_prefix
) {
    if (_pdf_sets.contains(name)) {
        throw std::runtime_error(
            std::format("PdfDriver::add_pdf_set: a PDF set named '{}' is already loaded", name)
        );
    }
    _pdf_sets[name] = std::make_shared<PdfSet>(
        _context, grid, std::move(pids), alpha_s_grid, existing_globals_prefix, name
    );
    return *this;
}

PdfDriver& PdfDriver::set_alpha_s_power(int power) {
    _alpha_s_power = power;
    return *this;
}

const std::shared_ptr<PdfSet>& PdfDriver::find_pdf_set(const std::string& name) const {
    auto found = _pdf_sets.find(name);
    if (found == _pdf_sets.end()) {
        throw std::runtime_error(std::format(
            "PdfDriver: no PDF set named '{}' has been loaded; call load_pdf_set() first", name
        ));
    }
    return found->second;
}

PdfDriver& PdfDriver::set_baseline(
    const std::string& pdf_set_name, ScaleChoice muF_scale, ScaleChoice muR_scale
) {
    auto ps = find_pdf_set(pdf_set_name);
    _baseline = PdfHypothesis{"", ps, ps, muF_scale, muR_scale};
    return *this;
}

PdfDriver& PdfDriver::set_baseline(
    const std::string& beam1_pdf_set_name,
    const std::string& beam2_pdf_set_name,
    ScaleChoice muF_scale,
    ScaleChoice muR_scale
) {
    _baseline = PdfHypothesis{
        "", find_pdf_set(beam1_pdf_set_name), find_pdf_set(beam2_pdf_set_name),
        muF_scale, muR_scale};
    return *this;
}

PdfDriver& PdfDriver::add_hypothesis(
    const std::string& launch_name,
    const std::string& pdf_set_name,
    ScaleChoice muF_scale,
    ScaleChoice muR_scale
) {
    auto ps = find_pdf_set(pdf_set_name);
    _hypotheses.push_back({launch_name, ps, ps, muF_scale, muR_scale});
    return *this;
}

PdfDriver& PdfDriver::add_hypothesis(
    const std::string& launch_name,
    const std::string& beam1_pdf_set_name,
    const std::string& beam2_pdf_set_name,
    ScaleChoice muF_scale,
    ScaleChoice muR_scale
) {
    _hypotheses.push_back(
        {launch_name, find_pdf_set(beam1_pdf_set_name), find_pdf_set(beam2_pdf_set_name),
         muF_scale, muR_scale}
    );
    return *this;
}

namespace {

// The standard "seven-point" scale-variation set: every (muR_factor,
// muF_factor) pair in {0.5, 1, 2}^2 except the two "uncorrelated" extremes
// (0.5, 2) and (2, 0.5), whose factor-of-4 simultaneous muR/muF shift is
// generally considered unphysically large to include.
const std::vector<std::pair<double, double>>& seven_point_factors() {
    static const std::vector<std::pair<double, double>> factors = {
        {1.0, 1.0}, {2.0, 1.0}, {0.5, 1.0}, {1.0, 2.0},
        {1.0, 0.5}, {2.0, 2.0}, {0.5, 0.5}
    };
    return factors;
}

// Formats a scale factor into a launch-name-safe token; every
// seven_point_factors() entry is one of these three.
std::string factor_token(double factor) {
    if (factor == 0.5) return "0p5";
    if (factor == 1.0) return "1";
    if (factor == 2.0) return "2";
    return std::format("{:g}", factor);
}

} // namespace

PdfDriver& PdfDriver::add_scale_variation_envelope(
    const std::string& group_name,
    const std::string& pdf_set_name,
    ScaleChoice muF_scale,
    ScaleChoice muR_scale
) {
    std::vector<std::string> names;
    names.reserve(seven_point_factors().size());
    for (auto& [muR_factor, muF_factor] : seven_point_factors()) {
        std::string name =
            group_name + "_muR" + factor_token(muR_factor) + "_muF" + factor_token(muF_factor);
        add_hypothesis(
            name, pdf_set_name, muF_scale.scaled_by(muF_factor), muR_scale.scaled_by(muR_factor)
        );
        names.push_back(std::move(name));
    }
    _scale_variation_groups.push_back(std::move(names));
    return *this;
}

PdfDriver& PdfDriver::add_scale_variation_envelope(
    const std::string& group_name,
    const std::string& beam1_pdf_set_name,
    const std::string& beam2_pdf_set_name,
    ScaleChoice muF_scale,
    ScaleChoice muR_scale
) {
    std::vector<std::string> names;
    names.reserve(seven_point_factors().size());
    for (auto& [muR_factor, muF_factor] : seven_point_factors()) {
        std::string name =
            group_name + "_muR" + factor_token(muR_factor) + "_muF" + factor_token(muF_factor);
        add_hypothesis(
            name, beam1_pdf_set_name, beam2_pdf_set_name, muF_scale.scaled_by(muF_factor),
            muR_scale.scaled_by(muR_factor)
        );
        names.push_back(std::move(name));
    }
    _scale_variation_groups.push_back(std::move(names));
    return *this;
}

PdfDriver& PdfDriver::set_scale_variation_output(ScaleVariationOutput mode) {
    _scale_variation_output = mode;
    return *this;
}

ScaleVariationOutput PdfDriver::scale_variation_output() const {
    return _scale_variation_output;
}

PdfDriver& PdfDriver::set_beam_energies(double beam1_energy, double beam2_energy) {
    _beam1_energy = beam1_energy;
    _beam2_energy = beam2_energy;
    _beam_energies_set = true;
    return *this;
}

PdfDriver& PdfDriver::load_events(const std::string& path, const madspace::LHEMeta& meta) {
    if (!_baseline) {
        throw std::runtime_error(
            "PdfDriver::load_events: no baseline set; call set_baseline() first"
        );
    }
    if (_hypotheses.empty()) {
        throw std::runtime_error(
            "PdfDriver::load_events: no hypotheses registered; call add_hypothesis() first"
        );
    }

    LheFormat format = detect_lhe_format(path);
    std::shared_ptr<REX::lhe> loaded =
        format == LheFormat::binary ? load_lhe_binary(path, meta) : load_lhe_xml(path);

    double beam1 = _beam1_energy, beam2 = _beam2_energy;
    if (!_beam_energies_set) {
        beam1 = loaded->eBm_[0];
        beam2 = loaded->eBm_[1];
    }
    if (beam1 <= 0.0 || beam2 <= 0.0) {
        throw std::runtime_error(
            "PdfDriver::load_events: beam energies are not known; call "
            "set_beam_energies() explicitly, or supply them via meta.beam1_energy/"
            "beam2_energy for binary event files"
        );
    }

    auto proc_rwgt = std::make_shared<REX::tea::procReweightor>();
    proc_rwgt->set_event_checker(REX::event_bool_fn([](REX::event&) { return true; }));
    proc_rwgt->set_normaliser(*make_pdf_weightor(*_baseline, beam1, beam2, _alpha_s_power));

    std::vector<REX::tea::weightor> rwgts;
    std::vector<std::string> names;
    rwgts.reserve(_hypotheses.size());
    names.reserve(_hypotheses.size());
    for (auto& hyp : _hypotheses) {
        rwgts.push_back(*make_pdf_weightor(hyp, beam1, beam2, _alpha_s_power));
        names.push_back(hyp.launch_name);
    }
    proc_rwgt->set_reweight_functions(rwgts);

    _reweightor = std::make_unique<REX::tea::reweightor>(
        std::move(*loaded), std::vector<std::shared_ptr<REX::tea::procReweightor>>{proc_rwgt}
    );
    _reweightor->set_launch_names(names);
    // Muted by default (see _verbose's member initializer); linking here
    // shares the flag both ways, so verbose().set_verbose(true) unmutes the
    // reweightor after the fact too.
    _reweightor->verbose.link(_verbose);
    _binary_output = (format == LheFormat::binary);
    return *this;
}

PdfDriver& PdfDriver::set_binary_output(bool binary) {
    _binary_output = binary;
    return *this;
}

void PdfDriver::filter_scale_variation_weights() {
    if (_scale_variation_groups.empty() || _scale_variation_output == ScaleVariationOutput::all) {
        return;
    }
    if (!_reweightor) return;

    auto& weight_ids = *_reweightor->weight_ids;
    std::set<std::size_t> to_drop;

    for (auto& group : _scale_variation_groups) {
        std::vector<std::size_t> indices;
        indices.reserve(group.size());
        for (auto& name : group) {
            auto it = std::find(weight_ids.begin(), weight_ids.end(), name);
            if (it != weight_ids.end()) {
                indices.push_back(static_cast<std::size_t>(std::distance(weight_ids.begin(), it)));
            }
        }
        // Nothing (left) of this group in weight_ids -- either
        // write_weights() has already dropped it (repeat call, harmless
        // no-op) or run() hasn't populated weight_ids yet.
        if (indices.empty()) continue;

        std::vector<std::size_t> keep;
        if (_scale_variation_output == ScaleVariationOutput::envelope) {
            std::size_t min_idx = indices.front(), max_idx = indices.front();
            for (auto idx : indices) {
                if (idx >= _reweightor->rwgt_xSec.size()) continue;
                if (_reweightor->rwgt_xSec[idx] < _reweightor->rwgt_xSec[min_idx]) min_idx = idx;
                if (_reweightor->rwgt_xSec[idx] > _reweightor->rwgt_xSec[max_idx]) max_idx = idx;
            }
            keep.push_back(min_idx);
            if (max_idx != min_idx) keep.push_back(max_idx);
        }
        // ScaleVariationOutput::none: keep stays empty, so every index of
        // this group below gets dropped.

        for (auto idx : indices) {
            if (std::find(keep.begin(), keep.end(), idx) == keep.end()) {
                to_drop.insert(idx);
            }
        }
    }

    if (to_drop.empty()) return;

    // Highest index first, so erasing one never invalidates the positions
    // still queued for removal.
    std::vector<std::size_t> drop_sorted(to_drop.begin(), to_drop.end());
    std::sort(drop_sorted.rbegin(), drop_sorted.rend());

    for (auto idx : drop_sorted) {
        if (idx < weight_ids.size()) weight_ids.erase(weight_ids.begin() + idx);
    }
    for (auto& ev : _reweightor->events) {
        if (!ev) continue;
        for (auto idx : drop_sorted) {
            if (idx < ev->wgts_.size()) ev->wgts_.erase(ev->wgts_.begin() + idx);
        }
    }
    for (auto idx : drop_sorted) {
        if (idx < _reweightor->rwgt_xSec.size()) {
            _reweightor->rwgt_xSec.erase(_reweightor->rwgt_xSec.begin() + idx);
        }
        if (idx < _reweightor->rwgt_xErr.size()) {
            _reweightor->rwgt_xErr.erase(_reweightor->rwgt_xErr.begin() + idx);
        }
    }
}

PdfDriver& PdfDriver::keep_weights(const std::vector<std::string>& keep_names) {
    if (!_reweightor) {
        throw std::runtime_error(
            "PdfDriver::keep_weights: reweightor has not been built yet; call load_events() "
            "first"
        );
    }
    auto& weight_ids = *_reweightor->weight_ids;
    std::vector<std::size_t> drop_sorted;
    for (std::size_t i = 0; i < weight_ids.size(); ++i) {
        if (std::find(keep_names.begin(), keep_names.end(), weight_ids[i]) == keep_names.end()) {
            drop_sorted.push_back(i);
        }
    }
    if (drop_sorted.empty()) return *this;

    // Highest index first, so erasing one never invalidates the positions
    // still queued for removal (drop_sorted is already ascending from the
    // loop above, so this just reverses it).
    std::sort(drop_sorted.rbegin(), drop_sorted.rend());

    for (auto idx : drop_sorted) {
        weight_ids.erase(weight_ids.begin() + idx);
    }
    for (auto& ev : _reweightor->events) {
        if (!ev) continue;
        for (auto idx : drop_sorted) {
            if (idx < ev->wgts_.size()) ev->wgts_.erase(ev->wgts_.begin() + idx);
        }
    }
    for (auto idx : drop_sorted) {
        if (idx < _reweightor->rwgt_xSec.size()) {
            _reweightor->rwgt_xSec.erase(_reweightor->rwgt_xSec.begin() + idx);
        }
        if (idx < _reweightor->rwgt_xErr.size()) {
            _reweightor->rwgt_xErr.erase(_reweightor->rwgt_xErr.begin() + idx);
        }
    }
    return *this;
}

PdfDriver& PdfDriver::write_weights(const std::string& path) {
    if (!_reweightor) {
        throw std::runtime_error(
            "PdfDriver::write_weights: reweightor has not been built yet; call "
            "load_events() first"
        );
    }
    filter_scale_variation_weights();
    if (_binary_output) {
        save_weights_binary(*_reweightor, path);
    } else {
        REX::write_lhef(*_reweightor, path, true);
    }
    return *this;
}

PdfDriver& PdfDriver::reweight(
    const std::string& events_path, const std::string& output_path, const madspace::LHEMeta& meta
) {
    load_events(events_path, meta);
    _reweightor->run();
    write_weights(output_path);
    return *this;
}

REX::tea::reweightor& PdfDriver::get() {
    if (!_reweightor) {
        throw std::runtime_error("PdfDriver::get: reweightor has not been built yet");
    }
    return *_reweightor;
}

const REX::tea::reweightor& PdfDriver::get() const {
    if (!_reweightor) {
        throw std::runtime_error("PdfDriver::get: reweightor has not been built yet");
    }
    return *_reweightor;
}

std::vector<int> PdfDriver::default_pids() {
    return {21, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
}

} // namespace madtrex
