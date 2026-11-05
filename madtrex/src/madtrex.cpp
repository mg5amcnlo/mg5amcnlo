#include "madtrex/madtrex.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <set>
#include <stdexcept>

namespace madtrex {

LheFormat detect_lhe_format(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("detect_lhe_format: failed to open " + path);
    }
    char magic[6] = {0};
    in.read(magic, sizeof(magic));
    if (in.gcount() == static_cast<std::streamsize>(sizeof(magic)) &&
        std::memcmp(magic, "\x93NUMPY", sizeof(magic)) == 0) {
        return LheFormat::binary;
    }
    return LheFormat::xml;
}

std::shared_ptr<REX::lhe> load_lhe_xml(const std::string& path) {
    return std::make_shared<REX::lhe>(REX::load_lhef(path));
}

std::shared_ptr<REX::lhe> load_lhe_xml(std::istream& stream) {
    return std::make_shared<REX::lhe>(REX::load_lhef(stream));
}

std::shared_ptr<REX::lhe> load_lhe(const std::string& path) {
    if (detect_lhe_format(path) == LheFormat::binary) {
        return load_lhe_binary(path);
    }
    return load_lhe_xml(path);
}

SplitResult split_events(
    const std::string& path,
    const std::string& output_prefix,
    std::size_t n_shards,
    std::size_t particle_count,
    std::size_t batch_size,
    const madspace::LHEMeta& meta
) {
    if (n_shards == 0) {
        throw std::runtime_error("split_events: n_shards must be > 0");
    }

    auto shard_path = [&](std::size_t i) {
        return output_prefix + "_shard" + std::to_string(i) + ".npy";
    };
    auto make_sink = [&](std::size_t i) -> REX::eventSink {
        return open_lhe_sink_binary(shard_path(i), particle_count);
    };

    REX::eventSource source = detect_lhe_format(path) == LheFormat::xml
        ? REX::open_lhe_source(path)
        : open_lhe_source_binary(path, meta);
    std::vector<std::size_t> counts = REX::split_source(source, n_shards, make_sink, batch_size);

    SplitResult result;
    result.counts = std::move(counts);
    result.paths.resize(n_shards);
    for (std::size_t i = 0; i < n_shards; ++i) {
        if (result.counts[i] > 0) {
            result.paths[i] = shard_path(i);
        }
    }
    return result;
}

REX::tea::reweightor combine_shards(
    const std::vector<std::string>& shard_event_paths,
    const std::vector<std::string>& shard_weight_prefixes,
    const std::vector<std::string>& launch_names,
    const std::string& output_path,
    bool binary_output,
    const madspace::LHEMeta& meta
) {
    if (shard_event_paths.empty()) {
        throw std::runtime_error("combine_shards: no shards given");
    }
    if (shard_event_paths.size() != shard_weight_prefixes.size()) {
        throw std::runtime_error(
            "combine_shards: shard_event_paths/shard_weight_prefixes size mismatch"
        );
    }
    if (launch_names.empty()) {
        throw std::runtime_error("combine_shards: no launch_names given");
    }

    // meta only informs the combined lhe's own init block (beam ids/
    // energies, xsec, PDF set); every other shard only contributes events,
    // so it's consulted at most once below, for whichever shard seeds
    // `combined`, and nowhere else. Shards with an empty path (see
    // SplitResult::paths) contributed zero events at split_events() time
    // and are silently skipped here the same way.
    std::shared_ptr<REX::lhe> combined;
    for (std::size_t s = 0; s < shard_event_paths.size(); ++s) {
        if (shard_event_paths[s].empty()) continue;
        if (!combined) {
            combined = load_lhe_binary(shard_event_paths[s], meta);
            load_weights_binary(combined->events, launch_names, shard_weight_prefixes[s]);
            continue;
        }
        auto shard = load_lhe_binary(shard_event_paths[s]);
        load_weights_binary(shard->events, launch_names, shard_weight_prefixes[s]);
        combined->events.insert(
            combined->events.end(), shard->events.begin(), shard->events.end()
        );
    }
    if (!combined) {
        throw std::runtime_error("combine_shards: every shard is empty; nothing to combine");
    }

    combined->set_weight_ids(launch_names);
    combined->sync_weight_ids();

    REX::tea::reweightor result(std::move(*combined));
    result.set_launch_names(launch_names);
    // The one and only place in this workflow that computes cross
    // sections/errors -- see this function's doc comment for why that
    // matters -- and it does so over every event from every shard at once.
    result.calc_xSecs();
    result.calc_xErrs();

    if (binary_output) {
        save_weights_binary(result, output_path);
    } else {
        REX::write_lhef(result, output_path, true);
    }

    return result;
}

namespace {

std::vector<long int> read_pdg_array(const json& arr) {
    std::vector<long int> result;
    result.reserve(arr.size());
    for (auto& v : arr) result.push_back(v.get<long int>());
    return result;
}

std::vector<FlavorChannel> expand_flavor_entry(const json& entry) {
    int index = entry.at("index").get<int>();
    bool mirror = entry.value("mirror", false);

    std::vector<FlavorChannel> result;
    for (auto& option : entry.at("options")) {
        auto pdgs = read_pdg_array(option);
        result.push_back(FlavorChannel{index, pdgs, false});
        if (mirror) {
            if (pdgs.size() < 2) {
                throw std::runtime_error(
                    "parse_subprocesses_json: flavor option has mirror=true but fewer "
                    "than 2 external legs"
                );
            }
            auto swapped = pdgs;
            std::swap(swapped[0], swapped[1]);
            result.push_back(FlavorChannel{index, std::move(swapped), true});
        }
    }
    return result;
}

// Concatenates an event's incoming-leg PDGs (status == -1, in event storage
// order) followed by its outgoing-leg PDGs (status == +1, in event storage
// order). Legs with any other status (eg intermediate resonances) are not
// part of this tuple.
std::vector<long int> ordered_pdgs(const REX::event& e) {
    std::vector<long int> incoming, outgoing;
    for (std::size_t i = 0; i < e.size(); ++i) {
        if (e.status_[i] == -1) incoming.push_back(e.pdg_[i]);
        else if (e.status_[i] == 1) outgoing.push_back(e.pdg_[i]);
    }
    incoming.insert(incoming.end(), outgoing.begin(), outgoing.end());
    return incoming;
}

std::vector<double> ordered_spins(const REX::event& e) {
    std::vector<double> incoming, outgoing;
    for (std::size_t i = 0; i < e.size(); ++i) {
        if (e.status_[i] == -1) incoming.push_back(e.spin_[i]);
        else if (e.status_[i] == 1) outgoing.push_back(e.spin_[i]);
    }
    incoming.insert(incoming.end(), outgoing.begin(), outgoing.end());
    return incoming;
}

const FlavorChannel* match_flavor(const SubProcessSpec& spec, const REX::event& e) {
    auto pdgs = ordered_pdgs(e);
    for (auto& fc : spec.flavors) {
        if (fc.pdgs == pdgs) return &fc;
    }
    return nullptr;
}

// A fixed helicity in the LHE SPINUP column: -1, +1, or 0 (scalars, and the
// longitudinal helicity of a massive vector), as in the helicity table of
// subprocesses.json. Anything else -- 9 for "unknown/summed" -- is not.
bool is_fixed_helicity(double spin) {
    return std::abs(spin - std::round(spin)) < 1e-6 && std::abs(spin) < 1.5;
}

// Explicit-key fallback used when the library doesn't export the optional
// umami_supported_inputs/required_inputs introspection symbols. Momenta and
// alpha_s are requested unconditionally; flavor index only when the subprocess
// distinguishes more than one channel (so single-channel libraries that lack
// UMAMI_IN_FLAVOR_INDEX support still work). Helicity index only with
// helicity reweighting (see Driver::set_helicity_reweighting()), and only added when the
// subprocess actually has helicity combinations to index into: older
// generated libraries hard-error on UMAMI_IN_HELICITY_INDEX when it isn't
// genuinely supported, and this fallback -- unlike default_input_keys() --
// has no supported_inputs() to check that against, since introspection
// itself is what failed to land us here in the first place; helicity
// reweighting is the implementation's explicit assertion that it knows better.
// SubProcessSpec's own helicity_index (via make_event_checker) populates
// event::helicity_ whenever fixed helicities are present regardless of this
// flag, ready for use once helicity reweighting is turned on.
// Why the helicity index of spec's events can't be mapped onto api's
// helicities, or "" if it can: the subprocess needs a helicity table (from
// subprocesses.json, which also defines MG7's SPINUP values) that enumerates
// exactly the helicities the library does (UMAMI_META_HELICITY_COUNT), with
// one helicity per external leg. UMAMI exposes the library's helicity count
// but not its table, so the count is what can be checked.
std::string helicity_map_problem(const SubProcessSpec& spec, const madspace::MatrixElementApi& api) {
    if (spec.helicities.empty()) {
        return "its subprocesses.json entry has no helicity table";
    }
    std::size_t legs = spec.incoming.size() + spec.outgoing.size();
    for (auto& row : spec.helicities) {
        if (row.size() != legs) {
            return std::format(
                "its helicity table has rows of {} helicities for {} external legs",
                row.size(), legs
            );
        }
    }
    std::size_t library_count;
    try {
        library_count = api.helicity_count();
    } catch (const std::runtime_error&) {
        return "the library does not report its helicity count";
    }
    if (library_count != spec.helicities.size()) {
        return std::format(
            "its helicity table has {} helicity combinations, the library {}",
            spec.helicities.size(), library_count
        );
    }
    return "";
}

// Whether api accepts UMAMI_IN_HELICITY_INDEX, or true if it doesn't say (the
// input_keys_for() fallback below then decides).
bool accepts_helicity_index(const madspace::MatrixElementApi& api) {
    try {
        auto supported = api.supported_inputs();
        return UMAMI_IN_HELICITY_INDEX < supported.size() && supported[UMAMI_IN_HELICITY_INDEX];
    } catch (const std::runtime_error&) {
        return true;
    }
}

std::vector<UmamiInputKey> input_keys_for(const SubProcessSpec& spec, bool helicity_reweighting) {
    std::vector<UmamiInputKey> keys{UMAMI_IN_MOMENTA, UMAMI_IN_ALPHA_S};
    std::set<int> distinct_indices;
    for (auto& fc : spec.flavors) distinct_indices.insert(fc.index);
    if (distinct_indices.size() > 1) keys.push_back(UMAMI_IN_FLAVOR_INDEX);
    if (helicity_reweighting && !spec.helicities.empty()) keys.push_back(UMAMI_IN_HELICITY_INDEX);
    return keys;
}

// The subprocess of the original process directory (normaliser_directory,
// parsed into candidates) that spec, from a process regenerated in another
// model, is normalised against: the one describing exactly the same matrix
// element call -- the same incoming and outgoing legs in the same order,
// the same flavor channels (UMAMI flavor index, PDG ids in leg order,
// mirroring) and the same helicity table -- so that an event's flavor and
// helicity indices mean the same in both libraries. Throws if there is none.
const SubProcessSpec& matching_normaliser_spec(
    const SubProcessSpec& spec,
    const std::vector<SubProcessSpec>& candidates,
    const std::string& normaliser_directory
) {
    auto same_flavors = [](const SubProcessSpec& a, const SubProcessSpec& b) {
        return std::equal(a.flavors.begin(), a.flavors.end(), b.flavors.begin(), b.flavors.end(),
                          [](const FlavorChannel& x, const FlavorChannel& y) {
                              return x.index == y.index && x.pdgs == y.pdgs && x.mirrored == y.mirrored;
                          });
    };
    for (const auto& candidate : candidates) {
        if (candidate.incoming == spec.incoming && candidate.outgoing == spec.outgoing &&
            same_flavors(candidate, spec) && candidate.helicities == spec.helicities) {
            return candidate;
        }
    }
    throw std::runtime_error(std::format(
        "Driver::load_process: subprocess {} has no identical counterpart (same legs, "
        "flavor channels and helicities) in {}, so the matrix element its events were "
        "generated with is not known exactly",
        spec.me_path_template, normaliser_directory
    ));
}

} // namespace

std::vector<SubProcessSpec> parse_subprocesses_json(const std::string& subprocesses_json_path) {
    std::ifstream in(subprocesses_json_path);
    if (!in) {
        throw std::runtime_error(
            "parse_subprocesses_json: failed to open " + subprocesses_json_path
        );
    }
    json data;
    in >> data;
    if (!data.is_array()) {
        throw std::runtime_error(
            "parse_subprocesses_json: expected a top-level JSON array in " +
            subprocesses_json_path
        );
    }

    std::vector<SubProcessSpec> specs;
    specs.reserve(data.size());
    for (auto& entry : data) {
        SubProcessSpec spec;
        spec.incoming = read_pdg_array(entry.at("incoming"));
        spec.outgoing = read_pdg_array(entry.at("outgoing"));
        spec.me_path_template = entry.at("me_path").get<std::string>();
        spec.source_path = entry.value("path", std::string());
        spec.diagram_count = entry.value("diagram_count", std::size_t(0));

        for (auto& flavor_entry : entry.at("flavors")) {
            auto expanded = expand_flavor_entry(flavor_entry);
            spec.flavors.insert(spec.flavors.end(), expanded.begin(), expanded.end());
        }

        if (entry.contains("helicities")) {
            for (auto& hel : entry.at("helicities")) {
                spec.helicities.push_back(hel.get<std::vector<int>>());
            }
        }

        specs.push_back(std::move(spec));
    }
    return specs;
}

REX::eventBelongs make_event_checker(const SubProcessSpec& spec) {
    REX::eventBelongs checker;

    // One template event per un-mirrored flavor option: the default
    // external_legs_comparator matches purely on PDG multiset per status,
    // which mirroring doesn't change, so mirrored variants don't need their
    // own template.
    for (auto& fc : spec.flavors) {
        if (fc.mirrored) continue;
        std::size_t n = fc.pdgs.size();
        REX::event templ(n);
        std::vector<short int> status(n, 1);
        for (std::size_t i = 0; i < spec.incoming.size() && i < n; ++i) status[i] = -1;
        templ.set_pdg(fc.pdgs);
        templ.set_status(status);
        checker.add_event(templ);
    }

    checker.flavor_index = [spec](REX::event& e) -> std::size_t {
        auto* fc = match_flavor(spec, e);
        if (!fc) {
            throw std::runtime_error(
                "make_event_checker: event matched the subprocess by PDG content but "
                "not positionally by any known flavor channel; the assumed event leg "
                "ordering (incoming, then outgoing, both in event storage order) may "
                "not hold for this event"
            );
        }
        return static_cast<std::size_t>(fc->index);
    };

    // The index of the event's helicity configuration in the subprocess's
    // helicity table -- the table the matrix element enumerates its
    // helicities in (UMAMI_IN_HELICITY_INDEX), and the one MG7 writes SPINUP
    // from -- read from SPINUP in the order of the matched flavor channel's
    // legs (an MG7 mirror is a rotation, keeping the leg order; a beam-
    // swapped "mirrored" channel has its two incoming spins swapped back).
    // REX::npos (-1) without fixed helicities, helicity_not_in_table for a
    // configuration missing from the table; both only matter, and are
    // reported, with helicity reweighting (see TupperWare's evaluate()).
    if (spec.helicities.empty()) return checker; // no table, no index
    checker.helicity_index = [spec](REX::event& e) -> std::size_t {
        auto* fc = match_flavor(spec, e);
        if (!fc) {
            throw std::runtime_error(
                "make_event_checker: event matched the subprocess by PDG content but "
                "not positionally by any known flavor channel"
            );
        }
        auto spins = ordered_spins(e);
        if (fc->mirrored && spins.size() >= 2) std::swap(spins[0], spins[1]);

        for (double s : spins) {
            if (!is_fixed_helicity(s)) return REX::npos; // no fixed helicity: sum over helicities
        }
        std::vector<int> rounded;
        rounded.reserve(spins.size());
        for (double s : spins) rounded.push_back(static_cast<int>(std::lround(s)));

        for (std::size_t i = 0; i < spec.helicities.size(); ++i) {
            if (spec.helicities[i] == rounded) return i;
        }
        return static_cast<std::size_t>(helicity_not_in_table);
    };

    return checker;
}

namespace {

// Scans dir for regular files whose name starts with prefix and ends with
// suffix, returning the (possibly empty) middle portion of each match --
// ie each one's resolved placeholder value(s), still concatenated together
// if the caller is matching more than one placeholder at once.
std::vector<std::string> scan_placeholder_values(
    const std::filesystem::path& dir, const std::string& prefix, const std::string& suffix
) {
    namespace fs = std::filesystem;
    std::vector<std::string> found;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() < prefix.size() + suffix.size()) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        found.push_back(name.substr(prefix.size(), name.size() - prefix.size() - suffix.size()));
    }
    return found;
}

// Picks one of found (each a candidate value for one placeholder) via
// priority: the first priority entry with a match wins; if none of the
// priority entries match but exactly one value was found, that one is used
// regardless; otherwise throws, listing what was found, so the caller can
// pass an explicit priority and try again. what/setter_hint name the
// placeholder kind and the Driver setter to use, for the error message
// only. found must outlive the returned reference.
const std::string& pick_priority(
    const std::vector<std::string>& found,
    const std::vector<std::string>& priority,
    const std::string& what,
    const std::string& setter_hint
) {
    for (auto& preferred : priority) {
        auto it = std::find(found.begin(), found.end(), preferred);
        if (it != found.end()) return *it;
    }
    if (found.size() == 1) {
        return found.front();
    }
    std::string found_list;
    for (auto& f : found) {
        if (!found_list.empty()) found_list += ", ";
        found_list += f;
    }
    throw std::runtime_error(
        "resolve_me_path: multiple " + what + " variants found (" + found_list +
        ") and none is listed in the " + what + " priority; call " + setter_hint +
        " to disambiguate"
    );
}

} // namespace

std::string resolve_me_path(
    const std::string& process_directory,
    const std::string& me_path_template,
    const std::vector<std::string>& device_priority,
    const std::vector<std::string>& fptype_priority
) {
    namespace fs = std::filesystem;
    fs::path full_template = fs::path(process_directory) / me_path_template;
    std::string filename_template = full_template.filename().string();
    fs::path dir = full_template.parent_path();

    static const std::string device_placeholder = "{device}";
    static const std::string fptype_placeholder = "{fptype}";
    auto device_pos = filename_template.find(device_placeholder);
    auto fptype_pos = filename_template.find(fptype_placeholder);

    if (device_pos == std::string::npos && fptype_pos == std::string::npos) {
        if (!fs::exists(full_template)) {
            throw std::runtime_error(
                "resolve_me_path: " + full_template.string() + " does not exist"
            );
        }
        return full_template.string();
    }

    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error(
            "resolve_me_path: directory " + dir.string() + " does not exist"
        );
    }

    if (device_pos != std::string::npos && fptype_pos != std::string::npos) {
        // Both placeholders present: {device} is expected to come first,
        // matching every me_path template MadMatrix itself ever generates
        // (see me_lib_format in madmatrix/output.py / MADMATRIX_LIB in
        // madmatrix.mk).
        if (fptype_pos < device_pos) {
            throw std::runtime_error(
                "resolve_me_path: me_path template has {fptype} before {device}, which is "
                "not supported: " + filename_template
            );
        }
        std::string prefix = filename_template.substr(0, device_pos);
        std::string middle_sep = filename_template.substr(
            device_pos + device_placeholder.size(),
            fptype_pos - (device_pos + device_placeholder.size())
        );
        std::string suffix = filename_template.substr(fptype_pos + fptype_placeholder.size());
        if (middle_sep.empty()) {
            throw std::runtime_error(
                "resolve_me_path: {device} and {fptype} must be separated by at least one "
                "literal character: " + filename_template
            );
        }

        std::vector<std::pair<std::string, std::string>> found; // (device, fptype)
        for (auto& value : scan_placeholder_values(dir, prefix, suffix)) {
            // middle_sep (eg "_") is expected to separate the device and
            // fptype values within value; search from the end since fptype
            // is the short, trailing token (one of SUPPORTED_FPTYPES in
            // madmatrix.mk: "d", "f", "m"), so this is robust even if a
            // device name ever happened to contain middle_sep itself.
            auto sep_pos = value.rfind(middle_sep);
            if (sep_pos == std::string::npos) continue;
            found.emplace_back(value.substr(0, sep_pos), value.substr(sep_pos + middle_sep.size()));
        }
        if (found.empty()) {
            throw std::runtime_error(
                "resolve_me_path: no library matching " + prefix + "*" + middle_sep + "*" +
                suffix + " found in " + dir.string()
            );
        }

        // Deduplicated (a device with several fptype variants on disk must
        // not look "ambiguous" on the device axis alone -- pick_priority's
        // found.size()==1 auto-select needs found to hold distinct values).
        std::vector<std::string> devices;
        std::set<std::string> seen_devices;
        for (auto& [d, f] : found) {
            if (seen_devices.insert(d).second) devices.push_back(d);
        }
        const std::string& chosen_device =
            pick_priority(devices, device_priority, "device", "Driver::set_device_priority(...)");

        std::vector<std::string> fptypes_for_device;
        std::set<std::string> seen_fptypes;
        for (auto& [d, f] : found) {
            if (d == chosen_device && seen_fptypes.insert(f).second) fptypes_for_device.push_back(f);
        }
        const std::string& chosen_fptype = pick_priority(
            fptypes_for_device, fptype_priority, "floating-point type",
            "Driver::set_fptype_priority(...)"
        );

        return (dir / (prefix + chosen_device + middle_sep + chosen_fptype + suffix)).string();
    }

    // Exactly one of the two placeholders is present (either {device}
    // alone -- the pre-existing convention -- or {fptype} alone).
    bool is_device = device_pos != std::string::npos;
    std::size_t pos = is_device ? device_pos : fptype_pos;
    std::size_t placeholder_len = (is_device ? device_placeholder : fptype_placeholder).size();
    std::string prefix = filename_template.substr(0, pos);
    std::string suffix = filename_template.substr(pos + placeholder_len);

    auto found = scan_placeholder_values(dir, prefix, suffix);
    if (found.empty()) {
        throw std::runtime_error(
            "resolve_me_path: no library matching " + prefix + "*" + suffix + " found in " +
            dir.string()
        );
    }
    const std::string& chosen = is_device
        ? pick_priority(found, device_priority, "device", "Driver::set_device_priority(...)")
        : pick_priority(
              found, fptype_priority, "floating-point type", "Driver::set_fptype_priority(...)"
          );
    return (dir / (prefix + chosen + suffix)).string();
}

Driver::Driver() : _context(madspace::default_context()) {}

Driver::Driver(madspace::ContextPtr context) : _context(std::move(context)) {}

Driver& Driver::set_device_priority(std::vector<std::string> priority) {
    _device_priority = std::move(priority);
    return *this;
}

const std::vector<std::string>& Driver::device_priority() const { return _device_priority; }

Driver& Driver::set_fptype_priority(std::vector<std::string> priority) {
    _fptype_priority = std::move(priority);
    return *this;
}

const std::vector<std::string>& Driver::fptype_priority() const { return _fptype_priority; }

Driver& Driver::set_normaliser_fptype_priority(std::vector<std::string> priority) {
    _normaliser_fptype_priority = std::move(priority);
    _normaliser_fptype_priority_set = true;
    return *this;
}

const std::vector<std::string>& Driver::normaliser_fptype_priority() const {
    return _normaliser_fptype_priority_set ? _normaliser_fptype_priority : _fptype_priority;
}

Driver& Driver::set_helicity_reweighting(bool helicity_reweighting) {
    _helicity_reweighting = helicity_reweighting;
    return *this;
}

bool Driver::helicity_reweighting() const { return _helicity_reweighting; }

Driver& Driver::set_normaliser_process_directory(
    const std::string& process_directory, const std::string& param_card
) {
    _normaliser_process_directory = process_directory;
    _normaliser_param_card = param_card;
    return *this;
}

const std::string& Driver::normaliser_process_directory() const {
    return _normaliser_process_directory;
}

Driver& Driver::set_normaliser_param_card(const std::string& param_card) {
    _normaliser_param_card = param_card;
    return *this;
}

const std::string& Driver::normaliser_param_card() const { return _normaliser_param_card; }

Driver& Driver::set_use_stored_normalisation(bool use_stored_normalisation) {
    _use_stored_normalisation = use_stored_normalisation;
    return *this;
}

bool Driver::use_stored_normalisation() const { return _use_stored_normalisation; }

Driver& Driver::set_use_helicity(bool use_helicity) { return set_helicity_reweighting(use_helicity); }

bool Driver::use_helicity() const { return _helicity_reweighting; }

REX::verbosity& Driver::verbose() { return _warehouse.verbose(); }

const REX::verbosity& Driver::verbose() const { return _warehouse.verbose(); }

Driver& Driver::set_verbose(bool v) {
    _warehouse.set_verbose(v);
    return *this;
}

bool Driver::is_verbose() const { return _warehouse.is_verbose(); }

Driver& Driver::set_max_batch(std::size_t max_batch) {
    _warehouse.set_max_batch(max_batch);
    return *this;
}

std::size_t Driver::max_batch() const { return _warehouse.max_batch(); }

Driver& Driver::set_threads(std::size_t threads) {
    _warehouse.set_threads(threads);
    return *this;
}

std::size_t Driver::threads() const { return _warehouse.threads(); }

Driver& Driver::set_min_batch(std::size_t min_batch) {
    _warehouse.set_min_batch(min_batch);
    return *this;
}

std::size_t Driver::min_batch() const { return _warehouse.min_batch(); }

Driver& Driver::set_max_deferred_iterations(std::size_t max_deferred_iterations) {
    _warehouse.set_max_deferred_iterations(max_deferred_iterations);
    return *this;
}

std::size_t Driver::max_deferred_iterations() const {
    return _warehouse.max_deferred_iterations();
}

Driver& Driver::set_streaming(bool enable, std::size_t batch_size) {
    _streaming = enable;
    _stream_batch_size = batch_size;
    return *this;
}

bool Driver::streaming() const { return _streaming; }

std::size_t Driver::stream_batch_size() const { return _stream_batch_size; }

Driver& Driver::load_process(
    const std::string& subprocesses_json_path, const std::string& param_card
) {
    namespace fs = std::filesystem;
    fs::path json_path(subprocesses_json_path);
    fs::path process_directory = json_path.parent_path().parent_path();

    std::string resolved_param_card = param_card;
    if (resolved_param_card.empty()) {
        resolved_param_card = (process_directory / "Cards" / "param_card.dat").string();
    }

    auto specs = parse_subprocesses_json(subprocesses_json_path);
    // The normaliser is the matrix element the events were generated with:
    // this process directory's own, unless set_normaliser_process_directory()
    // names the original one (a "change model" reweighting), whose
    // subprocesses must then be the same ones (see matching_normaliser_spec()).
    fs::path normaliser_directory = process_directory;
    std::string normaliser_param_card = resolved_param_card;
    std::vector<SubProcessSpec> normaliser_specs;
    if (!_normaliser_process_directory.empty()) {
        normaliser_directory = _normaliser_process_directory;
        normaliser_param_card = _normaliser_param_card.empty()
            ? (normaliser_directory / "Cards" / "param_card.dat").string()
            : _normaliser_param_card;
        normaliser_specs = parse_subprocesses_json(
            (normaliser_directory / "SubProcesses" / "subprocesses.json").string()
        );
    } else if (!_normaliser_param_card.empty()) {
        normaliser_param_card = _normaliser_param_card;
    }
    // See set_normaliser_fptype_priority()'s own doc comment: only load a
    // genuinely separate normaliser api when it would actually resolve to a
    // *different* library than the reweight-target api -- when they're
    // equal (the common case: no explicit call, or an explicit call that
    // happens to match), sharing one api for both roles is both correct and
    // cheaper (skips a redundant library load/context registration).
    const auto& normaliser_priority = normaliser_fptype_priority();
    bool separate_normaliser = normaliser_priority != _fptype_priority ||
                               !_normaliser_process_directory.empty() ||
                               normaliser_param_card != resolved_param_card;
    for (auto& spec : specs) {
        std::string me_path = resolve_me_path(
            process_directory.string(), spec.me_path_template, _device_priority, _fptype_priority
        );
        const auto& api = _context->load_matrix_element(me_path, resolved_param_card);
        detail::forget_api_parameters(api);
        const SubProcessSpec& normaliser_spec = _normaliser_process_directory.empty()
            ? spec
            : matching_normaliser_spec(spec, normaliser_specs, normaliser_directory.string());
        // Normaliser must reflect the sample's ORIGINAL configuration
        // (see set_normaliser_fptype_priority()), so when separate it gets
        // its own, independently-loaded api -- a different compiled library
        // (different fptype, or the original process directory's) --
        // rather than reusing the reweight-target api above. Deliberately
        // NOT passed to add_api(): it must never be touched by
        // ParamHandler's per-launch apply_launch() overrides, always
        // reflecting the untouched baseline every launch is reweighted
        // relative to.
        std::string normaliser_me_path;
        const madspace::MatrixElementApi* normaliser_api = &api;
        if (separate_normaliser) {
            normaliser_me_path = resolve_me_path(
                normaliser_directory.string(), normaliser_spec.me_path_template,
                _device_priority, normaliser_priority
            );
            normaliser_api =
                &_context->load_matrix_element(normaliser_me_path, normaliser_param_card);
            detail::forget_api_parameters(*normaliser_api);
        }
        auto checker = make_event_checker(spec);
        // Helicity reweighting for this subprocess needs both a library that
        // accepts UMAMI_IN_HELICITY_INDEX and a helicity map from the events'
        // SPINUP values onto the library's helicity enumeration
        // (helicity_map_problem()) -- for the reweighted matrix element and
        // the normaliser alike, which are always evaluated the same way:
        // both at the event's helicity, or both summed over helicities.
        bool helicity_reweighting = _helicity_reweighting;
        for (const auto* lib : {&api, normaliser_api}) {
            if (!helicity_reweighting) break;
            std::string problem = accepts_helicity_index(*lib)
                ? helicity_map_problem(spec, *lib)
                : "it does not support per-event helicities (UMAMI_IN_HELICITY_INDEX)";
            if (!problem.empty()) {
                detail::warn_once(
                    "no-helicity-map:" + lib->file_name(),
                    std::format(
                        "helicity reweighting is on, but the events of {} cannot be "
                        "mapped onto its helicities ({}); using helicity-summed matrix "
                        "elements for it",
                        lib->file_name(), problem
                    )
                );
                helicity_reweighting = false;
            }
        }
        auto keys_for = [&](const madspace::MatrixElementApi& lib, const SubProcessSpec& lib_spec) {
            std::pair<std::vector<UmamiInputKey>, std::vector<UmamiOutputKey>> keys;
            try {
                keys = {default_input_keys(lib, helicity_reweighting), default_output_keys(lib)};
            } catch (const std::runtime_error&) {
                keys = {input_keys_for(lib_spec, helicity_reweighting), {UMAMI_OUT_MATRIX_ELEMENT}};
            }
            return keys;
        };
        auto [input_keys, output_keys] = keys_for(api, spec);
        // Same (context, me_path, resolved_param_card) that api was just
        // loaded from, so TupperWare::build() can reload an independent
        // instance per max_batch overflow batch instead of every batch
        // sharing this one -- but only once set_max_batch() actually asks
        // for that (see WareHouse::set_max_batch()/TupperWare::build());
        // registering this unconditionally here costs nothing when it's
        // not.
        ReloadSpec reload{reload_context(), me_path, resolved_param_card};

        auto& tw = _warehouse.add_process()
            .add_api(api, input_keys, output_keys, reload)
            .set_event_checker(std::move(checker));
        if (_use_stored_normalisation) {
            tw.set_stored_normalisation(stored_matrix_element_key);
        }
        // Beam-swapped events of mirrored flavor channels go back into the
        // matrix element's leg order (see IncomingSwapped).
        if (std::any_of(spec.flavors.begin(), spec.flavors.end(), [](auto& fc) { return fc.mirrored; })) {
            tw.set_incoming_swapped([spec](const REX::event& e) {
                auto* fc = match_flavor(spec, e);
                return fc != nullptr && fc->mirrored;
            });
        }

        if (separate_normaliser) {
            auto [normaliser_input_keys, normaliser_output_keys] =
                keys_for(*normaliser_api, normaliser_spec);
            auto has_helicity = [](const std::vector<UmamiInputKey>& keys) {
                return std::find(keys.begin(), keys.end(), UMAMI_IN_HELICITY_INDEX) != keys.end();
            };
            if (has_helicity(normaliser_input_keys) != has_helicity(input_keys)) {
                throw std::runtime_error(std::format(
                    "Driver::load_process: {} and its normaliser {} would not be evaluated "
                    "with the same helicity treatment",
                    api.file_name(), normaliser_api->file_name()
                ));
            }
            ReloadSpec normaliser_reload{reload_context(), normaliser_me_path, normaliser_param_card};
            tw.set_normaliser(*normaliser_api, normaliser_input_keys, normaliser_output_keys, normaliser_reload);
        } else {
            // Explicit normaliser, using the same resolved keys as
            // add_api(): procReweightor::initialise() can only fall back to
            // an *immediate* (already-computed) slot of reweight_functions
            // when no normaliser is set, and a cuda/hip api registered here
            // lands in a genuinely deferred slot instead (see
            // TupperWare::build()), which can't stand in as an implicit
            // normaliser (see teaRex.cc). Passing the same input_keys/
            // output_keys explicitly also avoids set_normaliser()
            // re-deriving them via default_input_keys()/
            // default_output_keys(), which would throw for real MadMatrix
            // libraries that don't implement the optional UMAMI
            // introspection symbols -- the same reason add_api() above
            // needed the try/catch fallback in the first place.
            tw.set_normaliser(api, input_keys, output_keys, reload);
        }
        _specs.push_back(std::move(spec));
    }
    return *this;
}

Driver& Driver::load_process_directory(
    const std::string& process_directory, const std::string& param_card
) {
    namespace fs = std::filesystem;
    fs::path json_path = fs::path(process_directory) / "SubProcesses" / "subprocesses.json";
    return load_process(json_path.string(), param_card);
}

Driver& Driver::load_param_reweighting(const std::string& rwgt_path) {
    _param_handler = ParamHandler();
    _param_handler.add_apis(_warehouse).read_rwgt_card(rwgt_path);
    _warehouse.set_iterators(_param_handler.iterators());
    _warehouse.set_launch_names(_param_handler.launch_names());
    return *this;
}

ParamHandler& Driver::param_handler() { return _param_handler; }

const ParamHandler& Driver::param_handler() const { return _param_handler; }

Driver& Driver::load_events(const std::string& path, const madspace::LHEMeta& meta) {
    LheFormat format = detect_lhe_format(path);
    if (_streaming) {
        auto source = std::make_shared<REX::eventSource>(
            format == LheFormat::binary ? open_lhe_source_binary(path, meta)
                                         : REX::open_lhe_source(path)
        );
        _warehouse.build(REX::lhe::open_streaming(source, _stream_batch_size));
        _binary_output = (format == LheFormat::binary);
        return *this;
    }
    std::shared_ptr<REX::lhe> loaded =
        format == LheFormat::binary ? load_lhe_binary(path, meta) : load_lhe_xml(path);
    _warehouse.build(std::move(*loaded));
    _binary_output = (format == LheFormat::binary);
    return *this;
}

bool Driver::binary_output() const { return _binary_output; }

Driver& Driver::set_binary_output(bool binary) {
    _binary_output = binary;
    return *this;
}

Driver& Driver::write_weights(const std::string& path) {
    if (!_warehouse.built()) {
        throw std::runtime_error(
            "Driver::write_weights: WareHouse has not been built yet; call load_events() first"
        );
    }
    auto& built = _warehouse.get();
    if (_binary_output) {
        save_weights_binary(built, path);
    } else {
        // the stored matrix elements of binary input are MadtRex's own, not
        // part of the events (REX::event::print() writes every extra)
        for (auto& event : built.events) {
            if (event) event->extra.erase(stored_matrix_element_key);
        }
        REX::write_lhef(built, path, true);
    }
    return *this;
}

Driver& Driver::run_streaming(const std::string& output_path, bool include_ids) {
    if (!_streaming) {
        throw std::runtime_error(
            "Driver::run_streaming: streaming is not enabled; call set_streaming(true) "
            "before load_events()"
        );
    }
    if (!_warehouse.built()) {
        throw std::runtime_error(
            "Driver::run_streaming: WareHouse has not been built yet; call load_events() "
            "first"
        );
    }
    if (_binary_output) {
        // A streaming source never carries a header (see
        // open_lhe_source_binary()/open_lhe_source()'s own header-less
        // XML case), so reweightor::run_streaming()'s own weight_ids
        // resolution -- pre-existing header ids, if any, plus
        // launch_names, in that order -- reduces to just launch_names
        // here; open_weights_sink_binary() needs that same final list up
        // front, unlike REX::open_lhe_sink()'s XML sink, which discovers
        // it from write_open()'s own initNode/header argument instead.
        const auto& launch_names = _warehouse.launch_names();
        if (launch_names.empty()) {
            throw std::runtime_error(
                "Driver::run_streaming: binary_output() requires launch_names to be set "
                "(eg via load_param_reweighting() or warehouse().set_launch_names()), "
                "since binary weight files are named per weight id"
            );
        }
        auto sink = open_weights_sink_binary(launch_names, output_path);
        _warehouse.get().run_streaming(sink, include_ids);
        return *this;
    }
    auto sink = REX::open_lhe_sink(output_path, include_ids);
    _warehouse.get().run_streaming(sink, include_ids);
    return *this;
}

Driver& Driver::reweight(
    const std::string& process_directory,
    const std::string& events_path,
    const std::string& output_path,
    const std::string& rwgt_path,
    const std::string& param_card,
    const madspace::LHEMeta& meta
) {
    load_process_directory(process_directory, param_card);
    if (!rwgt_path.empty()) {
        load_param_reweighting(rwgt_path);
    }
    load_events(events_path, meta);
    if (_streaming) {
        run_streaming(output_path);
    } else {
        _warehouse.get().run();
        write_weights(output_path);
    }
    return *this;
}

madspace::ContextPtr Driver::context() const { return _context; }

madspace::ContextPtr Driver::reload_context() {
    // The instances reloaded per batch (ReloadSpec) are only evaluated from
    // REX's worker threads, ie through process instance 0: a context of one
    // thread initializes just that one per reload, rather than one per
    // thread of _context
    if (!_reload_context) {
        _reload_context = std::make_shared<madspace::Context>(_context->device(), 1);
    }
    return _reload_context;
}

WareHouse& Driver::warehouse() { return _warehouse; }

const WareHouse& Driver::warehouse() const { return _warehouse; }

const std::vector<SubProcessSpec>& Driver::specs() const { return _specs; }

} // namespace madtrex
