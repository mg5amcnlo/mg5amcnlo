// ParamHandler: UMAMI-direct SLHA parameter reweighting.
//
// REX::tea::param_rwgt (teaRex.h) drives SLHA parameter reweighting by
// rewriting param_card.dat between iterations and relying on the matrix
// element library to reread it from scratch on its next call. Where the
// UMAMI implementation exports umami_set_parameter/umami_get_parameter (see
// umami.h), that reread is unnecessary: the already-loaded instance's
// independent parameters can be updated in place. ParamHandler builds
// REX::tea::iterator objects that do exactly that, reusing
// REX::tea::rwgt_slha only for parsing the same "# launch / # set BLOCK_NAME
// PARAM_ID PARAM_VALUE" reweight-card format param_rwgt itself reads.

#pragma once

#include "tupperware.hpp"

#include <istream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace madtrex {

class ParamHandler {
public:
    ParamHandler() = default;

    // Registers apis this handler's iterators should set parameters on.
    // Typically every madspace::MatrixElementApi loaded for one physical
    // MadGraph7 process (ie every TupperWare::apis() in one WareHouse/
    // Driver), since they share one param_card.dat and so should be kept in
    // sync with each other. Duplicate apis (by pointer) are ignored. Throws
    // if api does not report supports_set_parameter() -- for such libraries,
    // use REX::tea::param_rwgt's file-based approach instead.
    ParamHandler& add_api(const madspace::MatrixElementApi& api);
    ParamHandler& add_apis(const std::vector<const madspace::MatrixElementApi*>& apis);
    // Convenience: add_apis(tw.apis()) for every TupperWare in house.
    ParamHandler& add_apis(const WareHouse& house);
    const std::vector<const madspace::MatrixElementApi*>& apis() const { return _apis; }

    // Parses a reweight card in the same format REX::tea::rwgt_slha reads
    // (see teaRex.h). Replaces any previously parsed card.
    ParamHandler& read_rwgt_card(std::istream& rwgt_in);
    ParamHandler& read_rwgt_card(const std::string& rwgt_path);

    std::size_t launch_count() const;

    // Applies launch idx's parameter changes to every registered api via
    // MatrixElementApi::set_parameter(), first resetting every parameter any
    // apply_launch() call on this handler has previously touched back to the
    // value it had the first time this handler read it (its "nominal"
    // value), so each launch is an independent perturbation from that
    // baseline rather than compounding on top of the previous launch's
    // changes -- mirrors REX::tea::rwgt_slha::write_rwgt_card's own
    // orig_params restore-then-clear step. Returns true on success; matches
    // REX::tea::iterator's signature so it can be wrapped directly (see
    // iterators()). Throws if idx is out of range or no apis are registered.
    bool apply_launch(std::size_t idx);

    // One REX::tea::iterator per parsed launch, in card order, each calling
    // apply_launch() for its own index; ready to hand to
    // WareHouse::set_iterators() (or REX::tea::reweightor::set_iterators()
    // directly). Each iterator captures this by pointer -- the ParamHandler
    // must outlive them, exactly as REX::tea::rwgt_slha::get_card_writers()
    // requires of its own rwgt_slha.
    std::vector<REX::tea::iterator> iterators();
    // Forwarded from the underlying REX::tea::rwgt_slha; see
    // rwgt_slha::get_launch_names()/get_rwgt_commands().
    std::vector<std::string> launch_names();
    std::vector<std::string> weight_context();

private:
    std::vector<const madspace::MatrixElementApi*> _apis;
    REX::tea::rwgt_slha _card_iter;
    // (api, "BLOCK_NAME PARAM_ID") -> value found the first time this
    // handler read that pair, ie the un-reweighted baseline every launch's
    // changes are applied relative to.
    std::unordered_map<
        const madspace::MatrixElementApi*,
        std::unordered_map<std::string, std::pair<double, double>>>
        _nominal;
};

} // namespace madtrex
