#include "madspace/driver/systematics.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <map>
#include <set>
#include <sstream>

#include "madspace/driver/logger.hpp"

using namespace madspace;
using json = nlohmann::json;

namespace {

bool layout_has(const DataLayout& layout, const std::string& name) {
    for (auto& field : layout.event_layout()) {
        if (field.name == name) {
            return true;
        }
    }
    return false;
}

std::string sanitize_description(const std::string& descr) {
    std::string out;
    for (std::size_t i = 0; i < descr.size(); ++i) {
        if (descr.compare(i, 2, "=>") == 0) {
            out += ";";
            ++i;
        } else if (descr[i] == '>') {
            out += ".gt.";
        } else if (descr[i] == '<') {
            out += ".lt.";
        } else if (descr[i] == '\n') {
            out += " ";
        } else {
            out += descr[i];
        }
    }
    return out;
}

// key of a distinct alpha_s value entering the matrix element re-evaluation
struct AlphaSKey {
    std::size_t alpha_s_index;
    double mur;
    int dyn;
    double alps;
    bool operator<(const AlphaSKey& other) const {
        return std::tie(alpha_s_index, mur, dyn, alps) <
            std::tie(other.alpha_s_index, other.mur, other.dyn, other.alps);
    }
};

bool particle_layout_has(const DataLayout& layout, const std::string& name) {
    for (auto& field : layout.particle_layout()) {
        if (field.name == name) {
            return true;
        }
    }
    return false;
}

} // namespace

std::string SystematicsCalculator::format_number(double value) {
    std::string str = std::format("{}", value);
    if (str.find('.') == std::string::npos && str.find('e') == std::string::npos &&
        str.find("inf") == std::string::npos && str.find("nan") == std::string::npos) {
        str += ".0";
    }
    return str;
}

std::string SystematicsCalculator::dyn_scale_name(int dyn) {
    switch (dyn) {
    case 1:
        return "sum pt";
    case 2:
        return "HT";
    case 3:
        return "HT/2";
    case 4:
        return "sqrts";
    default:
        return std::format("{}", dyn);
    }
}

double SystematicsCalculator::dynamical_scale(
    int dyn, const std::vector<std::array<double, 4>>& momenta
) {
    // same definitions as kernels/scale.hpp (incoming particles are the first two)
    switch (dyn) {
    case 1: {
        double et_sum = 0.;
        for (std::size_t i = 2; i < momenta.size(); ++i) {
            auto& p = momenta[i];
            double pt2 = p[1] * p[1] + p[2] * p[2];
            double p2 = pt2 + p[3] * p[3];
            if (p2 > 0.) {
                et_sum += p[0] * std::sqrt(pt2 / p2);
            }
        }
        return et_sum;
    }
    case 2:
    case 3: {
        double mt_sum = 0.;
        for (std::size_t i = 2; i < momenta.size(); ++i) {
            auto& p = momenta[i];
            mt_sum += std::sqrt(std::max(0., p[0] * p[0] - p[3] * p[3]));
        }
        return dyn == 2 ? mt_sum : 0.5 * mt_sum;
    }
    case 4: {
        if (momenta.size() < 2) {
            return 0.;
        }
        double e_tot = momenta[0][0] + momenta[1][0];
        double pz_tot = momenta[0][3] + momenta[1][3];
        return std::sqrt(std::max(0., e_tot * e_tot - pz_tot * pz_tot));
    }
    default:
        throw std::invalid_argument(
            std::format("invalid dynamical scale choice {}", dyn)
        );
    }
}

namespace {

// copy of `grid` restricted to the given PIDs (smaller coefficient tensor)
PdfGrid reduce_grid(const PdfGrid& grid, const std::vector<int>& pids) {
    PdfGrid reduced = grid;
    std::vector<std::size_t> columns;
    for (int pid : pids) {
        auto search = std::find(grid.pids.begin(), grid.pids.end(), pid);
        if (search == grid.pids.end()) {
            throw std::invalid_argument(
                std::format("PID {} not found in pdf grid", pid)
            );
        }
        columns.push_back(search - grid.pids.begin());
    }
    reduced.pids = pids;
    for (auto& row : reduced.values) {
        std::vector<double> new_row;
        for (std::size_t column : columns) {
            new_row.push_back(row.at(column));
        }
        row = new_row;
    }
    return reduced;
}

std::atomic<std::size_t> calculator_counter{0};

} // namespace

SystematicsCalculator::PdfEvaluator SystematicsCalculator::make_pdf_evaluator(
    const PdfGrid& grid,
    const std::vector<int>& pids,
    const std::string& name,
    std::size_t alpha_s_index
) {
    PdfGrid reduced = reduce_grid(grid, pids);
    std::string prefix = std::format("{}.{}", _prefix, name);
    reduced.initialize_globals(_context, prefix);
    PartonDensity density(reduced, pids, true, prefix);
    return PdfEvaluator{
        .runtime = build_runtime(density.function(), _context, false),
        .pids = pids,
        .alpha_s_index = alpha_s_index,
    };
}

SystematicsCalculator::SystematicsCalculator(
    const SystematicsConfig& config,
    const std::vector<SubprocessSystArgs>& subproc_args,
    const std::optional<PdfGrid>& nominal_pdf,
    const std::optional<AlphaSGrid>& nominal_alpha_s,
    ContextPtr context,
    const std::vector<std::optional<MatrixElement>>& matrix_elements,
    const nested_vector2<me_int_t>& me_flavor_remap,
    const std::optional<PdfGrid>& nominal_pdf2
) :
    _config(config),
    _subproc_args(subproc_args),
    _context(context ? context : std::make_shared<Context>(cpu_device(), 1)),
    _prefix(std::format("systematics{}", calculator_counter++)) {
    if (!nominal_alpha_s) {
        throw std::invalid_argument(
            "SystematicsCalculator requires the nominal alpha_s grid"
        );
    }
    if (_config.has_pdf && !nominal_pdf) {
        throw std::invalid_argument(
            "SystematicsCalculator requires the nominal PDF grid for hadronic beams"
        );
    }
    for (int dyn : _config.dyn_scales) {
        if (dyn < 1 || dyn > 4) {
            throw std::invalid_argument(
                std::format("invalid dynamical scale choice {} (must be 1-4)", dyn)
            );
        }
    }

    // alpha_s of the nominal set: batched RunningCoupling on the context
    auto add_alpha_s = [&](const AlphaSGrid& grid) {
        std::size_t index = _alpha_s_runtimes.size();
        std::string prefix = std::format("{}.alpha_s{}", _prefix, index);
        grid.initialize_globals(_context, prefix);
        RunningCoupling coupling(grid, prefix);
        _alpha_s_runtimes.push_back(
            build_runtime(coupling.function(), _context, false)
        );
        return index;
    };
    add_alpha_s(nominal_alpha_s.value());

    // the PIDs the events need: both beams of every flavor of every subprocess
    std::vector<int> pids;
    for (auto& args : _subproc_args) {
        for (auto& pdgs : args.beam_pdgs) {
            for (int pdg : pdgs) {
                if (std::find(pids.begin(), pids.end(), pdg) == pids.end()) {
                    pids.push_back(pdg);
                }
            }
        }
    }
    if (_config.pdf_ladder && nominal_pdf) {
        // a ladder step carries whichever parton the beam line had there
        for (int pdg : nominal_pdf->pids) {
            bool parton = pdg == 21 || (pdg != 0 && std::abs(pdg) <= 5);
            if (parton && std::find(pids.begin(), pids.end(), pdg) == pids.end()) {
                pids.push_back(pdg);
            }
        }
    }
    std::sort(pids.begin(), pids.end());
    if (_config.has_pdf) {
        _nominal_pdf = make_pdf_evaluator(nominal_pdf.value(), pids, "nominal", 0);
        if (nominal_pdf2) {
            _nominal_pdf2 =
                make_pdf_evaluator(nominal_pdf2.value(), pids, "nominal2", 0);
        }
    }

    // matrix elements for the subprocesses with mixed alpha_s powers
    _matrix_elements.resize(_subproc_args.size());
    if (!matrix_elements.empty()) {
        if (matrix_elements.size() != _subproc_args.size()) {
            throw std::invalid_argument(
                "SystematicsCalculator: one matrix element per subprocess expected"
            );
        }
        for (std::size_t i = 0; i < matrix_elements.size(); ++i) {
            auto& me = matrix_elements.at(i);
            if (!me || _subproc_args.at(i).qcd_power >= 0) {
                continue;
            }
            std::vector<me_int_t> remap;
            if (i < me_flavor_remap.size()) {
                remap = me_flavor_remap.at(i);
            }
            if (remap.size() < _subproc_args.at(i).beam_pdgs.size()) {
                throw std::invalid_argument(
                    std::format(
                        "SystematicsCalculator: flavor remap of subprocess {} has {} "
                        "entries, {} flavors expected",
                        i,
                        remap.size(),
                        _subproc_args.at(i).beam_pdgs.size()
                    )
                );
            }
            _matrix_elements.at(i) = MatrixElementData{
                .runtime = build_runtime(me->function(), _context, false),
                .particle_count = me->particle_count(),
                .flavor_remap = remap,
            };
        }
    }

    _mur_supported = true;
    for (std::size_t i = 0; i < _subproc_args.size(); ++i) {
        if (_subproc_args.at(i).qcd_power < 0 && !_matrix_elements.at(i)) {
            _mur_supported = false;
        }
    }

    // the varied PDF members; the alpha_s grid of a set is shared by its members
    std::map<std::string, std::size_t> alpha_s_indices;
    for (auto& spec : _config.pdf_members) {
        if (!_config.has_pdf) {
            _warnings.push_back(
                std::format(
                    "PDF variation {} member {} ignored: the beams have no PDF",
                    spec.set_name,
                    spec.member
                )
            );
            continue;
        }
        if (_nominal_pdf2) {
            // a member of one set cannot stand in for two different sets
            _warnings.push_back(
                std::format(
                    "PDF variation {} member {} ignored: the two beams use different "
                    "PDF sets",
                    spec.set_name,
                    spec.member
                )
            );
            continue;
        }
        bool same_set = spec.set_lhaid == _config.nominal_lhaid;
        if (!same_set && !_mur_supported) {
            _warnings.push_back(
                std::format(
                    "PDF variation {} member {} ignored: the events have no uniform "
                    "alpha_s power and no matrix element to re-evaluate, so the "
                    "alpha_s "
                    "of a different PDF set cannot be applied",
                    spec.set_name,
                    spec.member
                )
            );
            continue;
        }
        std::size_t alpha_s_index;
        if (same_set) {
            alpha_s_index = 0;
        } else if (auto search = alpha_s_indices.find(spec.info_file);
                   search != alpha_s_indices.end()) {
            alpha_s_index = search->second;
        } else {
            alpha_s_index = add_alpha_s(AlphaSGrid(spec.info_file));
            alpha_s_indices[spec.info_file] = alpha_s_index;
        }
        _member_pdfs.push_back(make_pdf_evaluator(
            PdfGrid(spec.grid_file),
            pids,
            std::format("member{}", _members.size()),
            alpha_s_index
        ));
        _members.push_back(spec);
    }

    build_variations();
    _variation_sums.assign(_variations.size(), 0.);
}

void SystematicsCalculator::build_variations() {
    _variations.clear();
    int id = _config.first_id;

    // scale variations: the (mur, muf) grid at the generation scale, then the
    // same grid (including the central point) for every alternative dynamical
    // scale, as systematics.py does with together = (mur, muf, dyn)
    std::vector<std::pair<double, double>> scales;
    if (_config.together) {
        for (double mur : _config.mur) {
            for (double muf : _config.muf) {
                scales.push_back({mur, muf});
            }
        }
    } else {
        scales.push_back({1., 1.});
        for (double mur : _config.mur) {
            if (mur != 1.) {
                scales.push_back({mur, 1.});
            }
        }
        for (double muf : _config.muf) {
            if (muf != 1.) {
                scales.push_back({1., muf});
            }
        }
    }
    std::vector<int> dyns{-1};
    dyns.insert(dyns.end(), _config.dyn_scales.begin(), _config.dyn_scales.end());
    bool mur_dropped = false;
    for (int dyn : dyns) {
        for (auto [mur, muf] : scales) {
            if (dyn == -1 && mur == 1. && muf == 1.) {
                continue; // the nominal point
            }
            bool changes_mur = mur != 1. || dyn != -1;
            if (changes_mur && !_mur_supported) {
                mur_dropped = true;
                continue;
            }
            if (muf != 1. && !_config.has_pdf) {
                continue;
            }
            _variations.push_back({id++, mur, muf, -1, dyn});
        }
    }
    if (mur_dropped) {
        _warnings.push_back(
            "renormalisation scale variations dropped: the power of alpha_s in |M|^2 "
            "is not the same for all diagrams of at least one subprocess (qcd_power "
            "= -1) and no matrix element is available to re-evaluate it. Only "
            "factorisation scale and PDF member variations are written."
        );
    }

    // emission scale variations of an MLM merged sample (madevent's alpsfact),
    // at the nominal mu_R, mu_F and PDF, as systematics.py writes them
    for (double alps : _config.alps) {
        if (alps != 1.) {
            _variations.push_back({id++, 1., 1., -1, -1, alps});
        }
    }

    // PDF variations: when members of the nominal set are requested, the group
    // starts with the nominal member itself so that the group is complete
    bool has_nominal_set_members =
        std::any_of(_members.begin(), _members.end(), [&](auto& spec) {
            return spec.set_lhaid == _config.nominal_lhaid;
        });
    if (has_nominal_set_members) {
        _variations.push_back({id++, 1., 1., -1, -1});
    }
    for (std::size_t i = 0; i < _members.size(); ++i) {
        _variations.push_back({id++, 1., 1., static_cast<int>(i), -1});
    }
}

std::vector<int> SystematicsCalculator::weight_ids() const {
    std::vector<int> ids;
    for (auto& var : _variations) {
        ids.push_back(var.id);
    }
    return ids;
}

std::vector<std::size_t> SystematicsCalculator::scale_variation_indices() const {
    std::vector<std::size_t> indices;
    for (std::size_t k = 0; k < _variations.size(); ++k) {
        if (_variations.at(k).is_scale()) {
            indices.push_back(k);
        }
    }
    return indices;
}

std::vector<std::size_t> SystematicsCalculator::emission_variation_indices() const {
    std::vector<std::size_t> indices;
    for (std::size_t k = 0; k < _variations.size(); ++k) {
        if (_variations.at(k).is_emission()) {
            indices.push_back(k);
        }
    }
    return indices;
}

std::vector<PdfGroupInfo> SystematicsCalculator::pdf_groups() const {
    std::map<int, PdfGroupInfo> groups;
    for (std::size_t k = 0; k < _variations.size(); ++k) {
        auto& var = _variations.at(k);
        if (var.is_scale() || var.is_emission()) {
            continue;
        }
        int set_lhaid, member;
        std::string set_name, error_type;
        if (var.pdf_index == -1) {
            set_lhaid = _config.nominal_lhaid;
            set_name = _config.nominal_set_name;
            error_type = _config.nominal_error_type;
            member = 0;
        } else {
            auto& spec = _members.at(var.pdf_index);
            set_lhaid = spec.set_lhaid;
            set_name = spec.set_name;
            error_type = spec.error_type;
            member = spec.member;
        }
        auto& group = groups[set_lhaid];
        group.set_name = set_name;
        group.set_lhaid = set_lhaid;
        group.error_type = error_type;
        group.members.push_back({member, k});
    }
    std::vector<PdfGroupInfo> result;
    for (auto& [lhaid, group] : groups) {
        std::sort(group.members.begin(), group.members.end());
        result.push_back(group);
    }
    return result;
}

std::tuple<double, double, double> SystematicsCalculator::pdf_uncertainty(
    const std::string& error_type,
    std::optional<double> central_opt,
    const std::vector<std::pair<int, double>>& member_values
) {
    double nan = std::nan("");
    std::vector<double> others;
    std::optional<double> member0;
    for (auto& [member, value] : member_values) {
        if (member == 0) {
            member0 = value;
        } else {
            others.push_back(value);
        }
    }
    double central = member0 ? member0.value() : central_opt.value_or(nan);
    if (error_type == "replicas") {
        if (others.empty()) {
            return {central, nan, nan};
        }
        double mean = 0.;
        for (double v : others) {
            mean += v;
        }
        mean /= others.size();
        double var = 0.;
        for (double v : others) {
            var += (v - mean) * (v - mean);
        }
        double err = others.size() > 1 ? std::sqrt(var / (others.size() - 1)) : 0.;
        return {mean, err, err};
    }
    if (std::isnan(central)) {
        return {nan, nan, nan};
    }
    if (error_type == "hessian") {
        // asymmetric errors from the (2i-1, 2i) eigenvector pairs
        double up = 0., down = 0.;
        for (std::size_t i = 0; i + 1 < others.size(); i += 2) {
            double dp = others[i] - central, dm = others[i + 1] - central;
            double u = std::max({dp, dm, 0.}), d = std::max({-dp, -dm, 0.});
            up += u * u;
            down += d * d;
        }
        return {central, std::sqrt(up), std::sqrt(down)};
    }
    // symmhessian and anything else: add in quadrature
    double sum = 0.;
    for (double v : others) {
        sum += (v - central) * (v - central);
    }
    return {central, std::sqrt(sum), std::sqrt(sum)};
}

std::vector<double> SystematicsCalculator::evaluate_pdf(
    const PdfEvaluator& evaluator,
    const std::vector<double>& x,
    const std::vector<double>& q,
    const std::vector<me_int_t>& slots
) const {
    std::size_t count = x.size();
    std::vector<double> values(count, 0.);
    if (count == 0) {
        return values;
    }
    Tensor x_tensor(DataType::dt_float64, {count});
    Tensor q_tensor(DataType::dt_float64, {count});
    Tensor slot_tensor(DataType::dt_int32, {count});
    auto x_view = x_tensor.view<double, 1>();
    auto q_view = q_tensor.view<double, 1>();
    auto slot_view = slot_tensor.view<me_int_t, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        x_view[i] = x[i];
        q_view[i] = q[i];
        slot_view[i] = slots[i];
    }
    TensorVec outputs;
    {
        std::lock_guard<std::mutex> lock(_runtime_mutex);
        outputs = evaluator.runtime->run({x_tensor, q_tensor, slot_tensor});
    }
    Tensor result = outputs.at(0).cpu().contiguous();
    auto result_view = result.view<double, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = result_view[i];
    }
    return values;
}

std::vector<double> SystematicsCalculator::evaluate_alpha_s(
    std::size_t alpha_s_index, const std::vector<double>& q
) const {
    std::size_t count = q.size();
    std::vector<double> values(count, 0.);
    if (count == 0) {
        return values;
    }
    Tensor q_tensor(DataType::dt_float64, {count});
    auto q_view = q_tensor.view<double, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        q_view[i] = q[i];
    }
    TensorVec outputs;
    {
        std::lock_guard<std::mutex> lock(_runtime_mutex);
        outputs = _alpha_s_runtimes.at(alpha_s_index)->run({q_tensor});
    }
    Tensor result = outputs.at(0).cpu().contiguous();
    auto result_view = result.view<double, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = result_view[i];
    }
    return values;
}

std::vector<std::array<double, 4>> SystematicsCalculator::event_momenta(
    EventBuffer& buffer, std::size_t event_index
) const {
    std::vector<std::array<double, 4>> momenta;
    for (std::size_t j = 0; j < buffer.particle_count(); ++j) {
        auto particle = buffer.particle(event_index, j);
        double e = particle.energy();
        if (e == 0. && j >= 2) {
            break; // padding
        }
        momenta.push_back({e, particle.px(), particle.py(), particle.pz()});
    }
    return momenta;
}

std::vector<double> SystematicsCalculator::matrix_elements(
    int subproc,
    EventBuffer& buffer,
    const std::vector<std::size_t>& indices,
    const std::vector<double>& alpha_s
) const {
    auto& me_data = _matrix_elements.at(subproc).value();
    std::size_t count = indices.size();
    std::size_t pc = me_data.particle_count;
    Tensor momenta(DataType::dt_float64, {count, pc, 4});
    Tensor alpha_s_tensor(DataType::dt_float64, {count});
    Tensor flavor(DataType::dt_int32, {count});
    auto mom_view = momenta.view<double, 3>();
    auto alpha_view = alpha_s_tensor.view<double, 1>();
    auto flavor_view = flavor.view<me_int_t, 1>();
    bool has_subproc = layout_has(buffer.layout(), "subprocess_index");
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t event_index = indices.at(i);
        auto event = buffer.event(event_index);
        for (std::size_t j = 0; j < pc; ++j) {
            auto particle = buffer.particle(event_index, j);
            mom_view[i][j][0] = particle.energy();
            mom_view[i][j][1] = particle.px();
            mom_view[i][j][2] = particle.py();
            mom_view[i][j][3] = particle.pz();
        }
        alpha_view[i] = alpha_s.at(i);
        flavor_view[i] = me_data.flavor_remap.at(event.flavor_index());
        (void)has_subproc;
    }
    TensorVec outputs;
    {
        std::lock_guard<std::mutex> lock(_runtime_mutex);
        outputs = me_data.runtime->run({momenta, alpha_s_tensor, flavor});
    }
    Tensor result = outputs.at(0).cpu().contiguous();
    auto result_view = result.view<double, 1>();
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = result_view[i];
    }
    return values;
}

LOReweightInfo SystematicsCalculator::reweight_info(
    EventBuffer& buffer, std::size_t event_index
) const {
    auto& layout = buffer.layout();
    bool has_beam1 = layout_has(layout, "fact_scale1");
    bool has_beam2 = layout_has(layout, "fact_scale2");
    bool has_subproc = layout_has(layout, "subprocess_index");
    auto event = buffer.event(event_index);
    int subproc = has_subproc ? event.subprocess_index().value() : 0;
    auto& args = _subproc_args.at(subproc);
    auto& pdgs = args.beam_pdgs.at(event.flavor_index());
    LOReweightInfo info{
        .qcd_power = args.qcd_power,
        .ren_scale = event.ren_scale(),
        .has_beam1 = has_beam1,
        .has_beam2 = has_beam2,
        .pdg1 = pdgs.at(0),
        .pdg2 = pdgs.at(1),
        .x1 = has_beam1 ? event.x1().value() : 0.,
        .x2 = has_beam2 ? event.x2().value() : 0.,
        .fact_scale1 = has_beam1 ? event.fact_scale1().value() : 0.,
        .fact_scale2 = has_beam2 ? event.fact_scale2().value() : 0.,
    };
    // the MLM alpha_s vertices and pdf ladder, as madevent books them
    if (particle_layout_has(layout, "alphas_scale")) {
        for (std::size_t j = 2; j < buffer.particle_count(); ++j) {
            double scale = buffer.particle(event_index, j).alphas_scale();
            if (scale > 0.) {
                info.asrwt.push_back(scale);
            }
        }
    }
    if (particle_layout_has(layout, "ladder_x") && has_beam1 && has_beam2) {
        std::array<double, 2> lead_q{
            buffer.particle(event_index, 0).ladder_lead_q(),
            buffer.particle(event_index, 1).ladder_lead_q()
        };
        if (lead_q[0] > 0. || lead_q[1] > 0.) {
            std::array<int, 2> pdg{info.pdg1, info.pdg2};
            std::array<double, 2> x{info.x1, info.x2};
            for (std::size_t b = 0; b < 2; ++b) {
                info.ladder_pdgs[b].push_back(pdg[b]);
                info.ladder_x[b].push_back(x[b]);
                info.ladder_q[b].push_back(lead_q[b]);
            }
            for (std::size_t j = 0; j < buffer.particle_count(); ++j) {
                auto particle = buffer.particle(event_index, j);
                int step_info = static_cast<int>(particle.ladder_info());
                if (!(step_info & 1)) {
                    continue;
                }
                int b = (step_info >> 1) & 1;
                info.ladder_pdgs[b].push_back(static_cast<int>(particle.ladder_pdg()));
                info.ladder_x[b].push_back(particle.ladder_x());
                info.ladder_q[b].push_back(particle.ladder_q());
            }
        }
    }
    // beam 1 moves along +z: leg 1 of a mirrored event belongs to beam 2
    if (has_beam1 && has_beam2 && buffer.particle(event_index, 0).pz() < 0.) {
        std::swap(info.pdg1, info.pdg2);
        std::swap(info.x1, info.x2);
        std::swap(info.fact_scale1, info.fact_scale2);
        std::swap(info.ladder_pdgs[0], info.ladder_pdgs[1]);
        std::swap(info.ladder_x[0], info.ladder_x[1]);
        std::swap(info.ladder_q[0], info.ladder_q[1]);
    }
    return info;
}

void SystematicsCalculator::compute(
    EventBuffer& buffer, std::vector<double>& weights
) const {
    std::size_t count = buffer.event_count();
    std::size_t var_count = _variations.size();
    weights.resize(count * var_count);
    if (var_count == 0 || count == 0) {
        return;
    }
    auto& layout = buffer.layout();
    bool has_beam1 = layout_has(layout, "fact_scale1");
    bool has_beam2 = layout_has(layout, "fact_scale2");
    bool has_partial = layout_has(layout, "partial_weight_product");
    bool has_subproc = layout_has(layout, "subprocess_index");
    bool use_pdf = _config.has_pdf && _nominal_pdf && (has_beam1 || has_beam2);
    bool has_dyn = !_config.dyn_scales.empty();
    bool has_vertices = particle_layout_has(layout, "alphas_scale");
    bool has_ladder = particle_layout_has(layout, "ladder_x");

    // per-event inputs
    struct EventInput {
        // the scales of the MLM clustering vertices whose alpha_s is evaluated
        // at their own scale rather than at mu_R (empty if none)
        std::vector<double> vertex_scales;
        // the MLM pdf reweighting ladder of each beam, as madevent books it
        // for its systematics (pdfrwt): the beam parton at x_beam and its
        // leading scale, then for every step the line's flavour, its momentum
        // fraction before the step and the scale the step takes it to
        struct LadderEntry {
            double x, q;
            me_int_t slot;
        };
        std::array<std::vector<LadderEntry>, 2> ladder;
        bool has_ladder = false;
        double w0, ren_scale, x1, x2, muf1, muf2, alpha_s, nominal_product;
        int subproc, qcd_power;
        me_int_t slot1, slot2;
        bool use_me;
        // leg 1 comes from beam 2 (mirrored event); only tracked when the
        // beams have different PDFs
        bool swapped;
        std::array<double, 5> dyn_scale;
    };
    std::vector<EventInput> inputs(count);
    auto pid_slot = [&](int pdg) -> me_int_t {
        auto& pids = _nominal_pdf->pids;
        return std::find(pids.begin(), pids.end(), pdg) - pids.begin();
    };
    for (std::size_t i = 0; i < count; ++i) {
        auto event = buffer.event(i);
        auto& in = inputs[i];
        in.w0 = event.weight();
        in.subproc = has_subproc ? event.subprocess_index().value() : 0;
        auto& args = _subproc_args.at(in.subproc);
        in.qcd_power = args.qcd_power;
        in.use_me = args.qcd_power < 0 && _matrix_elements.at(in.subproc).has_value();
        in.ren_scale = event.ren_scale();
        in.alpha_s = event.alpha_qcd();
        in.x1 = has_beam1 ? event.x1().value() : 0.;
        in.x2 = has_beam2 ? event.x2().value() : 0.;
        in.muf1 = has_beam1 ? event.fact_scale1().value() : 0.;
        in.muf2 = has_beam2 ? event.fact_scale2().value() : 0.;
        in.slot1 = in.slot2 = 0;
        in.nominal_product = 1.;
        in.swapped = _nominal_pdf2 && buffer.particle(i, 0).pz() < 0.;
        if (use_pdf) {
            auto& pdgs = args.beam_pdgs.at(event.flavor_index());
            in.slot1 = pid_slot(pdgs.at(0));
            in.slot2 = pid_slot(pdgs.at(1));
            if (has_partial) {
                in.nominal_product = event.partial_weight_product();
            }
        }
        if (has_vertices) {
            // A vertex the clustering did not reweight carries 0, as does an
            // unused slot: its alpha_s is one of the mu_R powers.
            for (std::size_t j = 2; j < buffer.particle_count(); ++j) {
                double scale = buffer.particle(i, j).alphas_scale();
                if (scale > 0.) {
                    in.vertex_scales.push_back(scale);
                }
            }
        }
        if (has_ladder && use_pdf) {
            std::array<double, 2> lead_q{
                buffer.particle(i, 0).ladder_lead_q(),
                buffer.particle(i, 1).ladder_lead_q()
            };
            // an event of a channel without the ladder carries zeros
            in.has_ladder = lead_q[0] > 0. || lead_q[1] > 0.;
            if (in.has_ladder) {
                in.ladder[0].push_back({in.x1, lead_q[0], in.slot1});
                in.ladder[1].push_back({in.x2, lead_q[1], in.slot2});
            }
            // the steps in clustering order, which is slot order
            for (std::size_t j = 0; in.has_ladder && j < buffer.particle_count(); ++j) {
                auto particle = buffer.particle(i, j);
                int info = static_cast<int>(particle.ladder_info());
                if (!(info & 1)) {
                    continue;
                }
                in.ladder[(info >> 1) & 1].push_back({
                    particle.ladder_x(),
                    particle.ladder_q(),
                    pid_slot(static_cast<int>(particle.ladder_pdg())),
                });
            }
        }
        in.dyn_scale.fill(0.);
        if (has_dyn) {
            auto momenta = event_momenta(buffer, i);
            for (int dyn : _config.dyn_scales) {
                in.dyn_scale[dyn] =
                    _config.scale_factor * dynamical_scale(dyn, momenta);
            }
        }
    }
    auto scale_of = [&](const EventInput& in, int dyn, double generated) {
        return dyn == -1 ? generated : in.dyn_scale[dyn];
    };
    // x f(x, q) at the given points, point n evaluated with `second` if
    // on_second[n] and with `first` otherwise. One batched call when both are
    // the same grid.
    auto evaluate_beams =
        [&](const PdfEvaluator& first,
            const PdfEvaluator& second,
            const std::vector<double>& x,
            const std::vector<double>& q,
            const std::vector<me_int_t>& slots,
            const std::vector<char>& on_second) {
            if (&first == &second) {
                return evaluate_pdf(first, x, q, slots);
            }
            std::array<std::vector<double>, 2> grid_x, grid_q;
            std::array<std::vector<me_int_t>, 2> grid_slots;
            for (std::size_t n = 0; n < x.size(); ++n) {
                grid_x[on_second[n]].push_back(x[n]);
                grid_q[on_second[n]].push_back(q[n]);
                grid_slots[on_second[n]].push_back(slots[n]);
            }
            std::array<std::vector<double>, 2> grid_values{
                evaluate_pdf(first, grid_x[0], grid_q[0], grid_slots[0]),
                evaluate_pdf(second, grid_x[1], grid_q[1], grid_slots[1]),
            };
            std::vector<double> values(x.size());
            std::array<std::size_t, 2> next{0, 0};
            for (std::size_t n = 0; n < x.size(); ++n) {
                values[n] = grid_values[on_second[n]][next[on_second[n]]++];
            }
            return values;
        };
    const PdfEvaluator* nominal_pdf2 = _nominal_pdf2
        ? &_nominal_pdf2.value()
        : (_nominal_pdf ? &_nominal_pdf.value() : nullptr);

    // Nominal PDF product when the events do not carry it: one batched call
    if (use_pdf && !has_partial) {
        std::vector<double> x, q;
        std::vector<me_int_t> slots;
        std::vector<char> on_second;
        for (auto& in : inputs) {
            if (has_beam1) {
                x.push_back(in.x1), q.push_back(in.muf1), slots.push_back(in.slot1);
                on_second.push_back(in.swapped);
            }
            if (has_beam2) {
                x.push_back(in.x2), q.push_back(in.muf2), slots.push_back(in.slot2);
                on_second.push_back(!in.swapped);
            }
        }
        auto values =
            evaluate_beams(_nominal_pdf.value(), *nominal_pdf2, x, q, slots, on_second);
        for (std::size_t i = 0, n = 0; i < count; ++i) {
            double product = 1.;
            if (has_beam1) {
                product *= values[n++];
            }
            if (has_beam2) {
                product *= values[n++];
            }
            inputs[i].nominal_product = product;
        }
    }

    // PDF part: R_pdf[event][variation], one batched call per grid
    // (nominal grid: every scale variation; member grids: the nominal scales)
    std::vector<std::vector<double>> r_pdf(count, std::vector<double>(var_count, 1.));
    if (use_pdf) {
        auto evaluate_variations =
            [&](const PdfEvaluator& evaluator,
                const PdfEvaluator& evaluator2,
                const std::vector<std::size_t>& var_indices) {
                std::vector<double> x, q;
                std::vector<me_int_t> slots;
                std::vector<char> on_second;
                for (auto& in : inputs) {
                    for (std::size_t k : var_indices) {
                        auto& var = _variations.at(k);
                        if (has_beam1) {
                            x.push_back(in.x1);
                            q.push_back(var.muf * scale_of(in, var.dyn, in.muf1));
                            slots.push_back(in.slot1);
                            on_second.push_back(in.swapped);
                        }
                        if (has_beam2) {
                            x.push_back(in.x2);
                            q.push_back(var.muf * scale_of(in, var.dyn, in.muf2));
                            slots.push_back(in.slot2);
                            on_second.push_back(!in.swapped);
                        }
                    }
                }
                auto values =
                    evaluate_beams(evaluator, evaluator2, x, q, slots, on_second);
                for (std::size_t i = 0, n = 0; i < count; ++i) {
                    for (std::size_t k : var_indices) {
                        double product = 1.;
                        if (has_beam1) {
                            product *= values[n++];
                        }
                        if (has_beam2) {
                            product *= values[n++];
                        }
                        r_pdf[i][k] = inputs[i].nominal_product == 0.
                            ? 0.
                            : product / inputs[i].nominal_product;
                    }
                }
            };
        std::vector<std::size_t> nominal_grid_vars;
        std::vector<std::vector<std::size_t>> member_vars(_member_pdfs.size());
        for (std::size_t k = 0; k < var_count; ++k) {
            auto& var = _variations.at(k);
            if (var.is_nominal()) {
                continue;
            }
            if (var.pdf_index == -1) {
                if (var.muf != 1. || var.dyn != -1) {
                    nominal_grid_vars.push_back(k);
                }
            } else {
                member_vars.at(var.pdf_index).push_back(k);
            }
        }
        if (!nominal_grid_vars.empty()) {
            evaluate_variations(_nominal_pdf.value(), *nominal_pdf2, nominal_grid_vars);
        }
        for (std::size_t m = 0; m < _member_pdfs.size(); ++m) {
            if (!member_vars[m].empty()) {
                evaluate_variations(_member_pdfs[m], _member_pdfs[m], member_vars[m]);
            }
        }
    }

    // MLM pdf reweighting ladder, as madevent's systematics.py rebuilds it
    // from the pdfrwt entries (q_0..q_N-1, x_0..x_N-1) of each beam:
    //   f(x_N-1, k S) prod_i<N-1 f_i(x_i, Q_i) / f_i(x_i+1, Q_i),
    //   Q_i = min(alpsfact q_i, k S),
    // S being the last entry's scale (mu_F where the ladder reaches it) or
    // the alternative dynamical scale. A PDF member evaluates all of it. The
    // nominal is rebuilt the same way, so the central point is exactly one.
    std::vector<std::size_t> ladder_events;
    for (std::size_t i = 0; use_pdf && i < count; ++i) {
        if (inputs[i].has_ladder) {
            ladder_events.push_back(i);
        }
    }
    if (!ladder_events.empty()) {
        double low = _config.ladder_min_scale;
        double high = _config.ladder_max_scale > 0. ? _config.ladder_max_scale : 1e30;
        auto clamp = [&](double q) { return std::min(std::max(q, low), high); };
        std::array<bool, 2> has_beam{has_beam1, has_beam2};
        auto ladder_weights = [&](const PdfEvaluator& evaluator, const Variation& var) {
            std::vector<double> x, q;
            std::vector<me_int_t> slots;
            for (std::size_t i : ladder_events) {
                auto& in = inputs[i];
                for (int b = 0; b < 2; ++b) {
                    auto& entries = in.ladder[b];
                    if (!has_beam[b] || entries.empty()) {
                        continue;
                    }
                    double top = var.muf *
                        (var.dyn == -1 ? entries.back().q : in.dyn_scale[var.dyn]);
                    x.push_back(entries.back().x);
                    q.push_back(clamp(top));
                    slots.push_back(entries.back().slot);
                    for (std::size_t e = 0; e + 1 < entries.size(); ++e) {
                        double scale = clamp(std::min(var.alps * entries[e].q, top));
                        x.push_back(entries[e].x);
                        q.push_back(scale);
                        slots.push_back(entries[e].slot);
                        x.push_back(entries[e + 1].x);
                        q.push_back(scale);
                        slots.push_back(entries[e].slot);
                    }
                }
            }
            auto values = evaluate_pdf(evaluator, x, q, slots);
            std::vector<double> result;
            std::size_t n = 0;
            for (std::size_t i : ladder_events) {
                auto& in = inputs[i];
                double w = 1.;
                for (int b = 0; b < 2; ++b) {
                    auto& entries = in.ladder[b];
                    if (!has_beam[b] || entries.empty()) {
                        continue;
                    }
                    w *= values[n++];
                    for (std::size_t e = 0; e + 1 < entries.size(); ++e) {
                        double numerator = values[n++];
                        double denominator = values[n++];
                        w = denominator >= 1e-10 ? w * numerator / denominator : 0.;
                    }
                }
                result.push_back(w);
            }
            return result;
        };
        Variation central{0, 1., 1., -1, -1, 1.};
        auto nominal = ladder_weights(_nominal_pdf.value(), central);
        for (std::size_t k = 0; k < var_count; ++k) {
            auto& var = _variations.at(k);
            bool varies_pdf =
                var.pdf_index >= 0 || var.muf != 1. || var.dyn != -1 || var.alps != 1.;
            if (!varies_pdf) {
                continue;
            }
            auto& evaluator = var.pdf_index >= 0
                ? _member_pdfs.at(var.pdf_index)
                : _nominal_pdf.value();
            auto varied = ladder_weights(evaluator, var);
            for (std::size_t n = 0; n < ladder_events.size(); ++n) {
                r_pdf[ladder_events[n]][k] =
                    nominal[n] == 0. ? 0. : varied[n] / nominal[n];
            }
        }
    }

    // alpha_s part: the distinct (alpha_s grid, mur, dyn) combinations, one
    // batched alpha_s call each; the matrix element is re-evaluated for the
    // events of the mixed-order subprocesses
    std::vector<AlphaSKey> keys;
    std::vector<std::vector<std::size_t>> key_vars;
    for (std::size_t k = 0; k < var_count; ++k) {
        auto& var = _variations.at(k);
        std::size_t alpha_s_index =
            var.pdf_index >= 0 ? _member_pdfs.at(var.pdf_index).alpha_s_index : 0;
        if (var.mur == 1. && var.dyn == -1 && alpha_s_index == 0 && var.alps == 1.) {
            continue;
        }
        AlphaSKey key{alpha_s_index, var.mur, var.dyn, var.alps};
        std::size_t pos = 0;
        for (; pos < keys.size(); ++pos) {
            if (!(keys[pos] < key) && !(key < keys[pos])) {
                break;
            }
        }
        if (pos == keys.size()) {
            keys.push_back(key);
            key_vars.emplace_back();
        }
        key_vars[pos].push_back(k);
    }
    std::vector<std::vector<double>> r_alpha(count, std::vector<double>(var_count, 1.));
    std::map<int, std::vector<std::size_t>> me_events;
    std::map<int, std::vector<double>> me_nominal;
    for (std::size_t i = 0; i < count; ++i) {
        if (inputs[i].use_me) {
            me_events[inputs[i].subproc].push_back(i);
        }
    }
    // alpha_s at the nominal vertex scales, with the nominal set: what the
    // events were generated with
    std::vector<double> vertex_nominal;
    if (has_vertices) {
        std::vector<double> q;
        for (auto& in : inputs) {
            q.insert(q.end(), in.vertex_scales.begin(), in.vertex_scales.end());
        }
        vertex_nominal = evaluate_alpha_s(0, q);
    }
    bool needs_me = std::any_of(keys.begin(), keys.end(), [](auto& key) {
        return key.mur != 1. || key.dyn != -1 || key.alpha_s_index != 0;
    });
    if (needs_me) {
        for (auto& [subproc, indices] : me_events) {
            std::vector<double> alpha_s(indices.size());
            for (std::size_t n = 0; n < indices.size(); ++n) {
                alpha_s[n] = inputs[indices[n]].alpha_s;
            }
            me_nominal[subproc] = matrix_elements(subproc, buffer, indices, alpha_s);
        }
    }
    for (std::size_t p = 0; p < keys.size(); ++p) {
        auto& key = keys[p];
        std::vector<double> q(count);
        for (std::size_t i = 0; i < count; ++i) {
            q[i] = key.mur * scale_of(inputs[i], key.dyn, inputs[i].ren_scale);
        }
        // an emission scale variation alone leaves the mu_R powers as they are
        bool varies_mur = key.mur != 1. || key.dyn != -1 || key.alpha_s_index != 0;
        auto varied = varies_mur
            ? evaluate_alpha_s(key.alpha_s_index, q)
            : std::vector<double>(count, 0.);
        std::vector<double> ratio(count, 1.);
        if (varies_mur) {
            for (std::size_t i = 0; i < count; ++i) {
                auto& in = inputs[i];
                if (in.use_me || in.qcd_power == 0) {
                    continue;
                }
                ratio[i] = in.alpha_s == 0.
                    ? 0.
                    : std::pow(varied[i] / in.alpha_s, in.qcd_power);
            }
            for (auto& [subproc, indices] : me_events) {
                std::vector<double> alpha_s(indices.size());
                for (std::size_t n = 0; n < indices.size(); ++n) {
                    alpha_s[n] = varied[indices[n]];
                }
                auto me = matrix_elements(subproc, buffer, indices, alpha_s);
                auto& nominal = me_nominal.at(subproc);
                for (std::size_t n = 0; n < indices.size(); ++n) {
                    ratio[indices[n]] = nominal[n] == 0. ? 0. : me[n] / nominal[n];
                }
            }
        }
        // MLM vertices, as madevent's systematics.py: the weight is
        // alpha_s(mu_R)^(n - n_v) prod_v alpha_s(q_v), and a variation takes the
        // n - n_v powers at k mu_R but every vertex at alpsfact q_v. The ratio
        // above moved all n powers to k mu_R, so the vertices are taken back
        // out of it and put in at their own scales:
        //   prod_v alpha'(alps q_v) / alpha(q_v) * (alpha(mu_R) / alpha'(k mu_R))^n_v
        // which holds for the matrix element re-evaluation too, since that
        // ratio is the same (alpha'(k mu_R) / alpha(mu_R))^n.
        if (has_vertices) {
            std::vector<double> vq;
            for (auto& in : inputs) {
                for (double scale : in.vertex_scales) {
                    vq.push_back(key.alps * scale);
                }
            }
            auto vertex_varied = evaluate_alpha_s(key.alpha_s_index, vq);
            for (std::size_t i = 0, n = 0; i < count; ++i) {
                auto& in = inputs[i];
                if (in.vertex_scales.empty()) {
                    continue;
                }
                double factor = 1.;
                for (std::size_t v = 0; v < in.vertex_scales.size(); ++v, ++n) {
                    factor *= vertex_nominal[n] == 0.
                        ? 0.
                        : vertex_varied[n] / vertex_nominal[n];
                    if (varies_mur) {
                        factor *= varied[i] == 0. ? 0. : in.alpha_s / varied[i];
                    }
                }
                ratio[i] *= factor;
            }
        }
        for (std::size_t k : key_vars[p]) {
            for (std::size_t i = 0; i < count; ++i) {
                r_alpha[i][k] = ratio[i];
            }
        }
    }

    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t k = 0; k < var_count; ++k) {
            weights[i * var_count + k] = inputs[i].w0 * r_alpha[i][k] * r_pdf[i][k];
        }
    }
}

void SystematicsCalculator::accumulate(
    EventBuffer& buffer, const std::vector<double>& weights
) {
    std::size_t count = buffer.event_count();
    std::size_t var_count = _variations.size();
    double nominal_sum = 0.;
    std::vector<double> sums(var_count, 0.);
    for (std::size_t i = 0; i < count; ++i) {
        nominal_sum += buffer.event(i).weight();
        for (std::size_t k = 0; k < var_count; ++k) {
            sums[k] += weights.at(i * var_count + k);
        }
    }
    std::lock_guard<std::mutex> lock(_accumulate_mutex);
    _nominal_sum += nominal_sum;
    for (std::size_t k = 0; k < var_count; ++k) {
        _variation_sums[k] += sums[k];
    }
    _event_count += count;
}

std::string SystematicsCalculator::initrwgt() const {
    std::string text;
    bool in_scale = false;
    bool in_emission = false;
    int in_pdf = -1;
    for (auto& var : _variations) {
        bool is_scale = var.is_scale();
        bool is_emission = var.is_emission();
        if (is_scale) {
            if (!in_scale) {
                text += "<weightgroup name=\"Central scale variation\" "
                        "combine=\"envelope\">\n";
                in_scale = true;
            }
        } else if (in_scale) {
            text += "</weightgroup> # scale\n";
            in_scale = false;
        }
        // systematics.py's name for the alpsfact group of a matched sample
        if (is_emission) {
            if (!in_emission) {
                text += "<weightgroup name=\"Emission scale variation\" "
                        "combine=\"envelope\">\n";
                in_emission = true;
            }
        } else if (in_emission) {
            text += "</weightgroup> # ALPS\n";
            in_emission = false;
        }

        std::string tag, info;
        int pdf_lhaid;
        if (is_emission) {
            pdf_lhaid = _config.nominal_lhaid;
            info = std::format("ALPSFACT={} ", format_number(var.alps));
        } else if (!is_scale) {
            // PDF group: the nominal member (pdf_index -1) belongs to the
            // nominal set's group
            int set_lhaid;
            std::string set_name, error_type, description;
            if (var.pdf_index == -1) {
                set_lhaid = _config.nominal_lhaid;
                set_name = _config.nominal_set_name;
                error_type = _config.nominal_error_type;
                description = _config.nominal_description;
                pdf_lhaid = set_lhaid;
            } else {
                auto& spec = _members.at(var.pdf_index);
                set_lhaid = spec.set_lhaid;
                set_name = spec.set_name;
                error_type = spec.error_type;
                description = spec.description;
                pdf_lhaid = spec.set_lhaid + spec.member;
                info = std::format("PDF={} MemberID={}", spec.set_lhaid, spec.member);
            }
            if (in_pdf != set_lhaid) {
                if (in_pdf != -1) {
                    text += "</weightgroup> # PDFSET to PDFSET\n";
                }
                text += std::format(
                    "<weightgroup name=\"{}\" combine=\"{}\"> # {}: {}\n",
                    set_name,
                    error_type,
                    set_lhaid,
                    sanitize_description(description)
                );
                in_pdf = set_lhaid;
            }
        } else {
            if (in_pdf != -1) {
                text += "</weightgroup> # PDF\n";
                in_pdf = -1;
            }
            pdf_lhaid = _config.nominal_lhaid;
            if (var.mur != 1.) {
                info += std::format("MUR={} ", format_number(var.mur));
            }
            if (var.muf != 1.) {
                info += std::format("MUF={} ", format_number(var.muf));
            }
            if (var.dyn != -1) {
                info += std::format("dyn_scale_choice={} ", dyn_scale_name(var.dyn));
            }
        }
        tag = std::format(
            "MUR=\"{}\" MUF=\"{}\" ", format_number(var.mur), format_number(var.muf)
        );
        if (var.dyn != -1) {
            tag += std::format("DYN_SCALE=\"{}\" ", var.dyn);
        }
        if (var.alps != 1.) {
            tag += std::format("ALPSFACT=\"{}\" ", format_number(var.alps));
        }
        tag += std::format("PDF=\"{}\" ", pdf_lhaid);
        text += std::format("<weight id=\"{}\" {}> {} </weight>\n", var.id, tag, info);
    }
    if (in_scale || in_emission || in_pdf != -1) {
        text += "</weightgroup>\n";
    }
    return text;
}

json SystematicsCalculator::summary() const {
    json variations = json::array();
    for (std::size_t k = 0; k < _variations.size(); ++k) {
        auto& var = _variations.at(k);
        json j = var;
        if (var.pdf_index == -1) {
            j["pdf_set"] = _config.nominal_set_name;
            j["pdf_lhaid"] = _config.nominal_lhaid;
            j["pdf_member"] = 0;
        } else {
            auto& spec = _members.at(var.pdf_index);
            j["pdf_set"] = spec.set_name;
            j["pdf_lhaid"] = spec.set_lhaid;
            j["pdf_member"] = spec.member;
        }
        if (_event_count > 0) {
            // the combined events carry "average" normalised weights (each event
            // weight estimates the total cross section), so the cross section of
            // a variation is the mean of its weights
            j["cross_section"] = _variation_sums.at(k) / _event_count;
        }
        variations.push_back(j);
    }

    json result{
        {"variations", variations},
        {"nominal",
         {{"pdf_set", _config.nominal_set_name}, {"pdf_lhaid", _config.nominal_lhaid}}},
        {"event_count", _event_count},
        {"warnings", _warnings},
    };
    if (_event_count == 0) {
        return result;
    }
    double nominal_xsec = _nominal_sum / _event_count;
    result["nominal"]["cross_section"] = nominal_xsec;
    std::vector<double> variation_xsecs(_variation_sums.size());
    for (std::size_t k = 0; k < _variation_sums.size(); ++k) {
        variation_xsecs[k] = _variation_sums[k] / _event_count;
    }

    // scale envelope
    auto scale_indices = scale_variation_indices();
    if (!scale_indices.empty()) {
        json scale_ids = json::array();
        double scale_min = nominal_xsec, scale_max = nominal_xsec;
        for (std::size_t k : scale_indices) {
            scale_ids.push_back(_variations.at(k).id);
            scale_min = std::min(scale_min, variation_xsecs.at(k));
            scale_max = std::max(scale_max, variation_xsecs.at(k));
        }
        result["scale"] = {
            {"weight_ids", scale_ids},
            {"min", scale_min},
            {"max", scale_max},
        };
    }

    // emission scale envelope (MLM alpsfact)
    auto emission_indices = emission_variation_indices();
    if (!emission_indices.empty()) {
        json ids = json::array();
        double low = nominal_xsec, high = nominal_xsec;
        for (std::size_t k : emission_indices) {
            ids.push_back(_variations.at(k).id);
            low = std::min(low, variation_xsecs.at(k));
            high = std::max(high, variation_xsecs.at(k));
        }
        result["emission"] = {
            {"weight_ids", ids},
            {"min", low},
            {"max", high},
        };
    }

    // PDF uncertainty per set
    json pdf_sets = json::array();
    for (auto& group : pdf_groups()) {
        std::vector<std::pair<int, double>> member_values;
        json ids = json::array();
        for (auto [member, k] : group.members) {
            member_values.push_back({member, variation_xsecs.at(k)});
            ids.push_back(_variations.at(k).id);
        }
        std::optional<double> central;
        if (group.set_lhaid == _config.nominal_lhaid) {
            central = nominal_xsec;
        }
        auto [c, up, down] = pdf_uncertainty(group.error_type, central, member_values);
        json entry{
            {"pdf_set", group.set_name},
            {"pdf_lhaid", group.set_lhaid},
            {"error_type", group.error_type},
            {"weight_ids", ids},
        };
        if (!std::isnan(c)) {
            entry["central"] = c;
        }
        if (!std::isnan(up)) {
            entry["uncertainty_up"] = up;
            entry["uncertainty_down"] = down;
        }
        pdf_sets.push_back(entry);
    }
    result["pdf"] = pdf_sets;
    return result;
}

// ---------------------------------------------------------------------------
// JSON (de)serialisation
// ---------------------------------------------------------------------------

void madspace::to_json(json& j, const PdfMemberSpec& spec) {
    j = json{
        {"set_name", spec.set_name},
        {"set_lhaid", spec.set_lhaid},
        {"member", spec.member},
        {"grid_file", spec.grid_file},
        {"info_file", spec.info_file},
        {"error_type", spec.error_type},
        {"description", spec.description},
    };
}

void madspace::from_json(const json& j, PdfMemberSpec& spec) {
    spec.set_name = j.at("set_name").get<std::string>();
    spec.set_lhaid = j.at("set_lhaid").get<int>();
    spec.member = j.at("member").get<int>();
    spec.grid_file = j.at("grid_file").get<std::string>();
    spec.info_file = j.at("info_file").get<std::string>();
    spec.error_type = j.value("error_type", "");
    spec.description = j.value("description", "");
}

void madspace::to_json(json& j, const SystematicsConfig& config) {
    j = json{
        {"mur", config.mur},
        {"muf", config.muf},
        {"alps", config.alps},
        {"pdf_ladder", config.pdf_ladder},
        {"ladder_min_scale", config.ladder_min_scale},
        {"ladder_max_scale", config.ladder_max_scale},
        {"together", config.together},
        {"dyn_scales", config.dyn_scales},
        {"scale_factor", config.scale_factor},
        {"pdf_members", config.pdf_members},
        {"nominal_set_name", config.nominal_set_name},
        {"nominal_lhaid", config.nominal_lhaid},
        {"nominal_error_type", config.nominal_error_type},
        {"nominal_description", config.nominal_description},
        {"has_pdf", config.has_pdf},
        {"write_inputs", config.write_inputs},
        {"first_id", config.first_id},
    };
}

void madspace::from_json(const json& j, SystematicsConfig& config) {
    config.mur = j.at("mur").get<std::vector<double>>();
    config.muf = j.at("muf").get<std::vector<double>>();
    config.alps = j.value("alps", std::vector<double>{1.});
    config.pdf_ladder = j.value("pdf_ladder", false);
    config.ladder_min_scale = j.value("ladder_min_scale", 0.);
    config.ladder_max_scale = j.value("ladder_max_scale", 0.);
    config.together = j.value("together", true);
    config.dyn_scales = j.value("dyn_scales", std::vector<int>{});
    config.scale_factor = j.value("scale_factor", 1.);
    config.pdf_members = j.value("pdf_members", std::vector<PdfMemberSpec>{});
    config.nominal_set_name = j.value("nominal_set_name", "");
    config.nominal_lhaid = j.value("nominal_lhaid", 0);
    config.nominal_error_type = j.value("nominal_error_type", "");
    config.nominal_description = j.value("nominal_description", "");
    config.has_pdf = j.value("has_pdf", true);
    config.write_inputs = j.value("write_inputs", false);
    config.first_id = j.value("first_id", 1);
}

void madspace::to_json(json& j, const SubprocessSystArgs& args) {
    j = json{{"qcd_power", args.qcd_power}, {"beam_pdgs", args.beam_pdgs}};
}

void madspace::from_json(const json& j, SubprocessSystArgs& args) {
    args.qcd_power = j.at("qcd_power").get<int>();
    args.beam_pdgs = j.at("beam_pdgs").get<nested_vector2<int>>();
}

void madspace::to_json(json& j, const Variation& variation) {
    j = json{
        {"id", variation.id},
        {"mur", variation.mur},
        {"muf", variation.muf},
        {"dyn", variation.dyn},
        {"alps", variation.alps},
        {"pdf_index", variation.pdf_index},
    };
}
