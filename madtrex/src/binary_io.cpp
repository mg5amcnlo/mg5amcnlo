#include "madtrex/binary_io.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace madtrex {

namespace {

using madspace::DataLayout;
using madspace::EventBuffer;
using madspace::EventFile;
using madspace::EventRecord;
using madspace::LHEEvent;
using madspace::LHEMeta;
using madspace::LHEParticle;
using madspace::ParticleRecord;

// The layout madtrex writes its own binary LHE files with (open_lhe_sink_
// binary(), save_weights_binary()'s sibling for events): always the same,
// fixed shape, since madtrex controls that format end to end -- unlike
// detect_lhe_event_layout() below, which reads a file this code did *not*
// write.
DataLayout lhe_event_layout() {
    return DataLayout(
        EventRecord::layout(EventRecord::f_lhe_event),
        ParticleRecord::layout(ParticleRecord::f_lhe_particle)
    );
}

// The layout to open path (an externally-produced binary LHE-format event
// file, eg MadGraph7's own events.npy from EventGenerator::
// combine_to_lhe_npy()) for *reading* with. Unlike lhe_event_layout()
// above, this isn't a fixed shape: combine_to_lhe_npy() additionally stores
// each event's own x1/fact_scale1 (beam 1) and x2/fact_scale2 (beam 2)
// whenever run_card.toml's [generation] systematics is enabled, growing
// every per-event record by that much. extract_lhe_event() never actually
// reads those values (only process_id/weight/scale/alpha_qed/alpha_qcd/
// particles), but the reader still needs to know they're there in order to
// compute correct byte offsets for everything that comes after them in the
// same record (in particular every particle's own data) -- so this peeks
// the file's own header first and builds a layout that actually matches,
// rather than assuming systematics was off. An MLM run's events also
// carry each particle's cluster_scale (ParticleRecord::f_clustering), which
// the reader skips over the same way.
DataLayout detect_lhe_event_layout(const std::string& path) {
    auto names = madspace::peek_npy_field_names(path);
    auto has = [&](const std::string& name) {
        return std::find(names.begin(), names.end(), name) != names.end();
    };
    int event_flags = EventRecord::f_lhe_event;
    if (has("fact_scale1")) event_flags |= EventRecord::f_beam1;
    if (has("fact_scale2")) event_flags |= EventRecord::f_beam2;
    if (has("partial_weight_product")) event_flags |= EventRecord::f_partial_weights;
    auto event_layout = EventRecord::layout(event_flags);
    // The weight columns after those: the systematics' rwgt_<id> (written
    // with [systematics] enabled, the default) and the reweighting's own
    // rwgt_<name> columns (added to events.npy in place by MadtRex's
    // MadGraph-side driver), in file order, then the bias. Missing them
    // shifted every later field, and the file was refused as "Invalid
    // header for event file".
    for (const auto& name : names) {
        if (name.rfind("rwgt_", 0) == 0) {
            event_layout.push_back({name, madspace::FieldLayout::f64, 7});
        }
    }
    if (has("bias")) {
        event_layout.push_back({"bias", madspace::FieldLayout::f64, 8});
    }
    // the generation-time matrix element madspace stores on request, for
    // stored_matrix_element_key
    if (has("matrix_element")) {
        event_layout.push_back({"matrix_element", madspace::FieldLayout::f64, 9});
    }
    int particle_flags = ParticleRecord::f_lhe_particle;
    if (has("part1_cluster_scale")) particle_flags |= ParticleRecord::f_clustering;
    return DataLayout(event_layout, ParticleRecord::layout(particle_flags));
}

// Extracts event index i out of buffer (laid out with EventRecord::
// f_lhe_event | ParticleRecord::f_lhe_particle, max_particles particles per
// event) as an LHEEvent. Particles with status_code == 0 are dropped: real
// LHE particles never carry status 0 (incoming is -1, outgoing +1,
// intermediate resonance 2), so a 0 marks the zero-filled padding
// EventGenerator::combine_to_lhe_npy (or open_lhe_sink_binary(), for a
// shard written by split_events()) appends past an event's actual particle
// count, up to the file's fixed particle_count(). Shared between
// read_binary_events() (whole-file) and open_lhe_source_binary()'s pull()
// (streamed, chunk at a time) -- the two differ only in how many events'
// worth of buffer they hand this one event at a time.
LHEEvent extract_lhe_event(EventBuffer& buffer, std::size_t i, std::size_t max_particles) {
    auto rec = buffer.event(i);
    LHEEvent evt;
    evt.process_id = rec.lhe_process_id();
    evt.weight = rec.lhe_weight();
    evt.scale = rec.lhe_scale();
    evt.alpha_qed = rec.lhe_alpha_qed();
    evt.alpha_qcd = rec.lhe_alpha_qcd();
    evt.particles.reserve(max_particles);
    for (std::size_t j = 0; j < max_particles; ++j) {
        auto prec = buffer.particle(i, j);
        int status = prec.lhe_status_code();
        if (status == 0) break; // padding: always a trailing run
        LHEParticle part;
        part.pdg_id = prec.lhe_pdg_id();
        part.status_code = status;
        part.mother1 = prec.lhe_mother1();
        part.mother2 = prec.lhe_mother2();
        part.color = prec.lhe_color();
        part.anti_color = prec.lhe_anti_color();
        part.px = prec.lhe_px();
        part.py = prec.lhe_py();
        part.pz = prec.lhe_pz();
        part.energy = prec.lhe_energy();
        part.mass = prec.lhe_mass();
        part.lifetime = prec.lhe_lifetime();
        part.spin = prec.lhe_spin();
        evt.particles.push_back(part);
    }
    return evt;
}

bool has_matrix_element(const DataLayout& layout) {
    const auto& fields = layout.event_layout();
    return std::any_of(fields.begin(), fields.end(), [](const auto& field) {
        return field.name == "matrix_element";
    });
}

// Reads every event out of an already-open, load-mode EventFile. Uses
// file.layout() (whatever it was actually opened with -- see
// detect_lhe_event_layout()) rather than assuming a fixed one, so this
// works regardless of which optional extra fields that layout does or
// doesn't include.
// The file's stored matrix elements, if it has them, go to
// matrix_elements, one per event.
std::vector<LHEEvent> read_binary_events(EventFile& file, std::vector<double>& matrix_elements) {
    // Named local: EventBuffer stores its DataLayout by reference, so the
    // layout must outlive every buffer built from it -- file.layout()
    // lives exactly as long as file does, which already outlives this
    // function's own scope (owned by the caller).
    const DataLayout& layout = file.layout();
    std::size_t max_particles = file.particle_count();
    EventBuffer buffer(0, max_particles, layout);

    std::vector<LHEEvent> events;
    events.reserve(file.event_count());
    bool has_me = has_matrix_element(layout);
    matrix_elements.clear();

    constexpr std::size_t chunk = 4096;
    while (file.read(buffer, chunk)) {
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            events.push_back(extract_lhe_event(buffer, i, max_particles));
            if (has_me) matrix_elements.push_back(buffer.event(i).matrix_element());
        }
    }
    return events;
}

REX::parton lhe_particle_to_parton(const LHEParticle& p) {
    REX::arr2<short int> mother{
        static_cast<short int>(p.mother1), static_cast<short int>(p.mother2)
    };
    REX::arr2<short int> icol{
        static_cast<short int>(p.color), static_cast<short int>(p.anti_color)
    };
    return REX::parton(
        REX::arr4<double>{p.energy, p.px, p.py, p.pz},
        p.mass,
        p.lifetime,
        p.spin,
        static_cast<long int>(p.pdg_id),
        static_cast<short int>(p.status_code),
        mother,
        icol
    );
}

std::shared_ptr<REX::event> lhe_event_to_event(const LHEEvent& raw) {
    std::vector<REX::parton> partons;
    partons.reserve(raw.particles.size());
    for (auto& p : raw.particles) partons.push_back(lhe_particle_to_parton(p));

    auto evt = std::make_shared<REX::event>(std::move(partons));
    evt->set_proc_id(raw.process_id);
    evt->set_weight(raw.weight);
    evt->set_scale(raw.scale);
    evt->set_alphaEW(raw.alpha_qed);
    evt->set_alphaS(raw.alpha_qcd);
    return evt;
}

REX::initNode lhe_meta_to_init(const LHEMeta& meta) {
    // Deliberately the default (nProc_ == 0, xSec_/xSecErr_/xMax_/lProc_ all
    // empty) constructor, not initNode(nproc): that overload pre-sizes those
    // four vectors to nproc entries, which the add_xSec()/etc. push-back loop
    // below would then grow past nproc, tripping validate_init()'s
    // nProc_-matches-size checks. set_nProc() records the count separately,
    // the same way the LHEF XML reader (read_lhef_body() in Rex.cc) does it.
    REX::initNode init;
    init.set_nProc(static_cast<short unsigned int>(meta.processes.size()));
    init.set_idBm(
        static_cast<long int>(meta.beam1_pdg_id), static_cast<long int>(meta.beam2_pdg_id)
    );
    init.set_eBm(meta.beam1_energy, meta.beam2_energy);
    init.set_pdfG(
        static_cast<short int>(meta.beam1_pdf_authors),
        static_cast<short int>(meta.beam2_pdf_authors)
    );
    init.set_pdfS(
        static_cast<long int>(meta.beam1_pdf_id), static_cast<long int>(meta.beam2_pdf_id)
    );
    init.set_idWgt(static_cast<short int>(meta.weight_mode));
    for (auto& proc : meta.processes) {
        init.add_xSec(proc.cross_section);
        init.add_xSecErr(proc.cross_section_error);
        init.add_xMax(proc.max_weight);
        init.add_lProc(proc.process_id);
    }
    return init;
}

// Inverse of lhe_particle_to_parton()/lhe_event_to_event(): reads back out
// of a REX::event exactly the fields those two populated in, so
// open_lhe_sink_binary()'s round trip through load_lhe_binary() is
// lossless for everything the binary format itself carries (it never
// carried mass_/vtim_/spin_ separately from momenta on the *event* struct
// to begin with -- those are per-particle, read from p below).
LHEParticle parton_to_lhe_particle(const REX::event& e, std::size_t i) {
    LHEParticle p;
    p.pdg_id = static_cast<int>(e.pdg_[i]);
    p.status_code = e.status_[i];
    p.mother1 = e.mother_[i][0];
    p.mother2 = e.mother_[i][1];
    p.color = e.icol_[i][0];
    p.anti_color = e.icol_[i][1];
    p.energy = e.momenta_[i].t();
    p.px = e.momenta_[i].x();
    p.py = e.momenta_[i].y();
    p.pz = e.momenta_[i].z();
    p.mass = e.mass_[i];
    p.lifetime = e.vtim_[i];
    p.spin = e.spin_[i];
    return p;
}

std::string sanitize_token(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        out.push_back(
            (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') ? c : '_'
        );
    }
    return out.empty() ? std::string("weight") : out;
}

} // namespace

std::shared_ptr<REX::lhe> load_lhe_binary(EventFile& file, const LHEMeta& meta) {
    std::vector<double> matrix_elements;
    auto raw_events = read_binary_events(file, matrix_elements);

    REX::lheReader<LHEMeta, LHEEvent> reader;
    reader.set_init_translator(lhe_meta_to_init);
    reader.set_event_translator(lhe_event_to_event);

    auto result = reader.read(meta, raw_events);
    if (!matrix_elements.empty() && matrix_elements.size() == result.events.size()) {
        for (std::size_t i = 0; i < result.events.size(); ++i) {
            result.events[i]->extra[stored_matrix_element_key] = matrix_elements[i];
        }
    }
    return std::make_shared<REX::lhe>(std::move(result));
}

std::shared_ptr<REX::lhe> load_lhe_binary(const std::string& path, const LHEMeta& meta) {
    EventFile file(path, detect_lhe_event_layout(path), 0, EventFile::load);
    return load_lhe_binary(file, meta);
}

REX::eventSource open_lhe_source_binary(const std::string& path, const LHEMeta& meta) {
    // Holds the open EventFile (and, via its own layout(), the DataLayout
    // EventBuffer only ever borrows a reference to -- see
    // extract_lhe_event()'s doc comment) for as long as the returned
    // eventSource's pull() closure is alive. Forward-only, same as
    // open_lhe_source(): no rewind is offered.
    struct State {
        EventFile file;
        std::size_t max_particles;
        bool has_me;
        explicit State(const std::string& p) :
            file(p, detect_lhe_event_layout(p), 0, EventFile::load),
            max_particles(file.particle_count()),
            has_me(has_matrix_element(file.layout())) {}
    };
    auto state = std::make_shared<State>(path);

    REX::eventSource src;
    src.read_init = [meta]() -> REX::initNode { return lhe_meta_to_init(meta); };
    src.pull = [state](std::size_t n) -> std::vector<std::shared_ptr<REX::event>> {
        EventBuffer buffer(0, state->max_particles, state->file.layout());
        if (!state->file.read(buffer, n)) {
            return {};
        }
        std::vector<std::shared_ptr<REX::event>> result;
        result.reserve(buffer.event_count());
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            result.push_back(
                lhe_event_to_event(extract_lhe_event(buffer, i, state->max_particles))
            );
            if (state->has_me) {
                result.back()->extra[stored_matrix_element_key] =
                    static_cast<double>(buffer.event(i).matrix_element());
            }
        }
        return result;
    };
    return src;
}

void save_weights_binary(const REX::lhe& lhe, const std::string& path_prefix) {
    if (!lhe.weight_ids || lhe.weight_ids->empty()) {
        throw std::runtime_error("save_weights_binary: lhe has no weight_ids to write");
    }
    for (std::size_t k = 0; k < lhe.weight_ids->size(); ++k) {
        EventBuffer buffer(lhe.events.size(), 0, madspace::weight_file_layout);
        for (std::size_t i = 0; i < lhe.events.size(); ++i) {
            const auto& evt = lhe.events[i];
            if (!evt || evt->wgts_.size() <= k) {
                throw std::runtime_error(
                    "save_weights_binary: event " + std::to_string(i) +
                    " is missing weight '" + (*lhe.weight_ids)[k] + "'"
                );
            }
            buffer.event(i).weight() = evt->wgts_[k];
        }
        std::string file_name =
            path_prefix + "_" + sanitize_token((*lhe.weight_ids)[k]) + ".npy";
        EventFile out(file_name, madspace::weight_file_layout, 0, EventFile::create);
        out.write(buffer);
    }
}

REX::eventSink open_weights_sink_binary(
    const std::vector<std::string>& weight_ids, const std::string& path_prefix
) {
    if (weight_ids.empty()) {
        throw std::runtime_error("open_weights_sink_binary: weight_ids must not be empty");
    }
    // One owner (this State) of every per-weight-id EventFile, reached
    // identically by every closure below -- same rationale as
    // open_lhe_sink_binary()'s State: write_close()'s clear() is then the
    // sole, deterministic trigger for finalizing every file's on-disk
    // header, not whichever copy of a shared_ptr happens to die last.
    struct State {
        std::vector<std::unique_ptr<EventFile>> files;
    };
    auto state = std::make_shared<State>();

    REX::eventSink sink;
    sink.write_open = [state, weight_ids, path_prefix](const REX::initNode&, const std::any&) {
        state->files.clear();
        state->files.reserve(weight_ids.size());
        for (auto& id : weight_ids) {
            std::string file_name = path_prefix + "_" + sanitize_token(id) + ".npy";
            state->files.push_back(std::make_unique<EventFile>(
                file_name, madspace::weight_file_layout, 0, EventFile::create
            ));
        }
    };
    sink.write_event = [state, weight_ids](std::shared_ptr<REX::event> ev) {
        if (!ev) return;
        if (ev->wgts_.size() < weight_ids.size()) {
            throw std::runtime_error(
                "open_weights_sink_binary: event is missing one or more of the " +
                std::to_string(weight_ids.size()) + " expected weights"
            );
        }
        for (std::size_t k = 0; k < weight_ids.size(); ++k) {
            EventBuffer buffer(1, 0, madspace::weight_file_layout);
            buffer.event(0).weight() = ev->wgts_[k];
            state->files.at(k)->write(buffer);
        }
    };
    sink.write_close = [state]() {
        state->files.clear(); // destructs every EventFile, finalizing each header
    };
    return sink;
}

void load_weights_binary(
    std::vector<std::shared_ptr<REX::event>>& events,
    const std::vector<std::string>& weight_ids,
    const std::string& path_prefix
) {
    for (auto& id : weight_ids) {
        std::string file_name = path_prefix + "_" + sanitize_token(id) + ".npy";
        EventFile file(file_name, madspace::weight_file_layout, 0, EventFile::load);
        if (file.event_count() != events.size()) {
            throw std::runtime_error(
                "load_weights_binary: " + file_name + " has " +
                std::to_string(file.event_count()) + " rows, expected " +
                std::to_string(events.size()) + " (one per event)"
            );
        }
        EventBuffer buffer(0, 0, madspace::weight_file_layout);
        constexpr std::size_t chunk = 4096;
        std::size_t offset = 0;
        while (file.read(buffer, chunk)) {
            for (std::size_t i = 0; i < buffer.event_count(); ++i) {
                auto& evt = events.at(offset + i);
                if (!evt) {
                    throw std::runtime_error(
                        "load_weights_binary: events[" + std::to_string(offset + i) +
                        "] is null"
                    );
                }
                evt->wgts_.push_back(buffer.event(i).weight());
            }
            offset += buffer.event_count();
        }
    }
}

REX::eventSink open_lhe_sink_binary(const std::string& path, std::size_t particle_count) {
    // Single owner (unique_ptr) of the EventFile, reached identically by
    // every closure below through the one shared State -- unlike each
    // closure separately capturing its own shared_ptr<EventFile> copy, this
    // makes write_close()'s reset() the sole, deterministic trigger for
    // EventFile::~EventFile() (which finalizes the on-disk shape/header),
    // rather than leaving that to whenever every copy happens to go out of
    // scope.
    struct State {
        std::string path;
        std::size_t particle_count;
        std::unique_ptr<EventFile> file;
    };
    auto state = std::make_shared<State>();
    state->path = path;
    state->particle_count = particle_count;

    REX::eventSink sink;
    sink.write_open = [](const REX::initNode&, const std::any&) {
        // The binary format carries no init-block information; see
        // load_lhe_binary()'s own doc comment.
    };
    sink.write_event = [state](std::shared_ptr<REX::event> ev) {
        if (!ev) return;
        std::size_t n = ev->size();
        if (!state->file) {
            // Establishes this shard's fixed particle-slot capacity from
            // whichever event lands here first, if the caller didn't pin
            // one down explicitly; see open_lhe_sink_binary()'s doc
            // comment.
            std::size_t pc = state->particle_count != 0 ? state->particle_count : n;
            state->file = std::make_unique<EventFile>(
                state->path, lhe_event_layout(), pc, EventFile::create
            );
            state->particle_count = pc;
        }
        if (n > state->particle_count) {
            throw std::runtime_error(
                "open_lhe_sink_binary: event has " + std::to_string(n) +
                " particles, more than this shard's fixed capacity (" +
                std::to_string(state->particle_count) + ")"
            );
        }

        DataLayout layout = lhe_event_layout();
        EventBuffer buffer(1, state->particle_count, layout);
        LHEEvent raw;
        raw.process_id = static_cast<int>(ev->proc_id_);
        raw.weight = ev->weight_;
        raw.scale = ev->scale_;
        raw.alpha_qed = ev->alphaEW_;
        raw.alpha_qcd = ev->alphaS_;
        buffer.event(0).from_lhe_event(raw);
        for (std::size_t j = 0; j < n; ++j) {
            buffer.particle(0, j).from_lhe_particle(parton_to_lhe_particle(*ev, j));
        }
        // particles [n, state->particle_count) stay zero-filled
        // (status_code == 0, among everything else), matching
        // read_binary_events()'s own padding convention.
        state->file->write(buffer);
    };
    sink.write_close = [state]() {
        // Finalizes the on-disk header/shape (EventFile::~EventFile()); a
        // no-op, leaving no file on disk at all, if write_event() was never
        // called (an empty shard).
        state->file.reset();
    };
    return sink;
}

} // namespace madtrex
