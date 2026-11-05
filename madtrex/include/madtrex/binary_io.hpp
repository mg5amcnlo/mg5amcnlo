// MadtRex binary I/O: builds/writes REX::lhe objects against MadGraph7's
// internal binary event-file format (madspace::EventFile, io.hpp) instead of
// LHEF XML, without teaching Rex itself anything about that format.
//
// Rex already factors "assemble/sort a REX::lhe from some raw format" into a
// generic REX::lheReader<InitRaw, EventRaw, HeaderRaw>/REX::lheWriter<...>
// pair driven by plain translator functions (see Rex.h); REX::xml_reader()/
// xml_writer() are simply that template instantiated with shared_ptr<xmlNode>
// as the raw type. This header instantiates the same machinery with
// madspace::LHEMeta/LHEEvent (io.hpp's lhe_output.hpp raw structs, which are
// already field-for-field identical to the binary format's
// EventRecord::f_lhe_event/ParticleRecord::f_lhe_particle layout) as the raw
// types, so all translation/sorting bookkeeping is still Rex's, and only the
// binary<->LHEEvent/LHEMeta plumbing lives here.

#pragma once

#include "tupperware.hpp"

#include "madspace/driver/io.hpp"
#include "madspace/driver/lhe_output.hpp"

#include <memory>
#include <string>

namespace madtrex {

// Reads a MadGraph7 internal binary event file (an EventFile opened in
// EventFile::load mode, expected to have been written with
// EventRecord::f_lhe_event | ParticleRecord::f_lhe_particle layout, eg via
// EventGenerator::combine_to_lhe_npy) into a REX::lhe.
//
// The binary format itself carries no init-block information (beam ids/
// energies, PDF set, per-process cross sections): MadGraph7's own driver
// constructs that separately (see madspace::LHEMeta) and does not persist it
// alongside the binary event file. meta is therefore an optional, caller-
// supplied source for that information; the resulting REX::lhe's initNode is
// left at its default (all-zero) values for whichever fields meta doesn't
// specify. This has no effect on reweighting itself: procReweightor::
// evaluate() only ever reads per-event data (momenta, pdg, status, flavor_/
// helicity_ indices, alphaS), never initNode fields.
//
// When the file has a "matrix_element" column (the generation-time squared
// matrix elements madspace stores on request, EventRecord::f_matrix_element),
// each event's value is kept as extra[stored_matrix_element_key], for
// TupperWare::set_stored_normalisation(). open_lhe_source_binary() does the
// same; open_lhe_sink_binary() does not write it back out.
std::shared_ptr<REX::lhe> load_lhe_binary(
    const std::string& path, const madspace::LHEMeta& meta = {}
);
// As above, given an already-open EventFile (must be in EventFile::load
// mode, with a layout compatible with EventRecord::f_lhe_event |
// ParticleRecord::f_lhe_particle).
std::shared_ptr<REX::lhe> load_lhe_binary(
    madspace::EventFile& file, const madspace::LHEMeta& meta = {}
);

// Streaming counterpart to load_lhe_binary(const std::string&, ...): opens
// path as a load-mode EventFile and returns a REX::eventSource whose
// pull(n) reads up to n further events at a time straight off disk (via
// EventFile::read()) rather than loading the whole file up front --
// otherwise translating each event exactly as load_lhe_binary() does.
// read_init() returns exactly what load_lhe_binary(path, meta)'s own init
// would have been (meta is consulted once, there, and nowhere else); the
// binary format has no header, so no read_header is set. Forward-only, like
// open_lhe_source(): no rewind is offered, open this again for a second
// pass. The returned source keeps its EventFile open for as long as its
// pull closure is kept alive.
REX::eventSource open_lhe_source_binary(
    const std::string& path, const madspace::LHEMeta& meta = {}
);

// Writes the per-event weights recorded in lhe.events[i]->wgts_ (ie the
// output of running a REX::tea::reweightor over lhe) back out as one
// madspace weight-only binary file per entry of lhe.weight_ids, using
// madspace::weight_file_layout (EventRecord::f_weight, no particle data) --
// the same index-aligned, kinematics-free weight-file convention already
// used for per-channel weights in ChannelEventGenerator. Files are named
// "<path_prefix>_<weight_id>.npy"; weight_id is sanitized to a
// filesystem-safe token if needed. Throws if an event's wgts_ is shorter
// than lhe.weight_ids (ie the reweighting loop didn't populate that weight
// id for every event).
void save_weights_binary(const REX::lhe& lhe, const std::string& path_prefix);

// Streaming counterpart to save_weights_binary(): builds a REX::eventSink
// that writes one weight-only binary EventFile per entry of weight_ids
// (same "<path_prefix>_<weight_id>.npy" convention, same
// madspace::weight_file_layout), one event's worth of rows at a time via
// write_event(), rather than requiring the whole REX::lhe to already be
// fully populated in memory the way save_weights_binary() does. Intended
// for Driver::run_streaming()'s binary_output() path -- weight_ids must
// already be final by the time this is called (write_open() only opens the
// files this constructs from weight_ids up front; it can't discover more
// weight ids later). write_event() throws if a given event's wgts_ is
// shorter than weight_ids.
REX::eventSink open_weights_sink_binary(
    const std::vector<std::string>& weight_ids, const std::string& path_prefix
);

// Reads back what save_weights_binary(lhe, path_prefix) wrote for exactly
// weight_ids (same order, same path_prefix), appending one entry to every
// events[i]->wgts_ per weight id, in weight_ids order -- ie after this
// call, events[i]->wgts_[k] is weight_ids[k]'s value for events[i], for
// every k. events[i]->weight_ids is left untouched; call
// REX::lhe::set_weight_ids()/sync_weight_ids() (or just assign it directly)
// separately once the surrounding REX::lhe is assembled. Throws if a
// "<path_prefix>_<weight_id>.npy" file is missing, or its row count doesn't
// match events.size().
void load_weights_binary(
    std::vector<std::shared_ptr<REX::event>>& events,
    const std::vector<std::string>& weight_ids,
    const std::string& path_prefix
);

// Builds a REX::eventSink that writes full events (not just weights, unlike
// save_weights_binary()) to a fresh madspace binary EventFile at path, laid
// out the way load_lhe_binary() expects to read them back
// (EventRecord::f_lhe_event | ParticleRecord::f_lhe_particle) -- so a shard
// written by this sink is, on its own, a valid load_lhe_binary() source.
// Intended for use with REX::split_source() to shard a large reweighting
// job's input across independent per-shard jobs (see madtrex::split_events()),
// always writing the binary format for the shards regardless of the
// original source's own format.
//
// The file's particle-count is fixed once and for all the first time
// write_event() is actually called (EventFile's own on-disk shape can't
// grow afterward): if particle_count is left at 0, it is taken from that
// first event's own particle count; passed explicitly, it instead fixes an
// upper bound up front (useful to guarantee sibling shards end up with a
// matching, or at least a known, capacity). Either way, every event this
// sink is asked to write from then on must have at most that many
// particles -- fewer are padded with zero-filled, status_code==0 particles
// up to it (mirroring the padding load_lhe_binary() already strips back
// out on read), more throws. write_open()'s initNode/header are accepted
// (so this satisfies REX::eventSink) but otherwise ignored: the binary
// format carries no init-block information, same as load_lhe_binary()'s
// own caveat. If write_event() is never called (eg an empty shard), no
// file is ever created on disk.
REX::eventSink open_lhe_sink_binary(const std::string& path, std::size_t particle_count = 0);

} // namespace madtrex
