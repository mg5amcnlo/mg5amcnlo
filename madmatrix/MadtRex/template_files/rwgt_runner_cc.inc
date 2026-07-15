//==========================================================================
// Copyright (C) 2023-2026 CERN
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Written originally by: Z. Wettersten (Jan 2024) for the MG5aMC CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
//==========================================================================
// This file has been automatically generated for the CUDACPP plugin by
%(info_lines)s
//==========================================================================
//==========================================================================
// A class for reweighting matrix elements for
%(process_lines)s
//--------------------------------------------------------------------------
#ifndef _TREX_
#define _TREX_
#endif
#include "rwgt_instance.h"
#include "fbridge.h"

namespace %(process_namespace)s
{

    std::shared_ptr<REX::eventBelongs> get_comp()
    {
        static std::vector<std::vector<short int>> status = {%(parton_status)s};
        static std::vector<std::vector<long int>> pdg = {%(parton_ids)s};
        if (status.size() != pdg.size())
            throw std::runtime_error("Inconsistent event data in rwgt_runner");
        static std::vector<std::shared_ptr<REX::event>> loc_evs;
        for (size_t i = 0; i < status.size(); ++i)
        {
            auto ev = std::make_shared<REX::event>(pdg[i].size());
            ev->set_status(status[i]);
            ev->set_pdg(pdg[i]);
            loc_evs.push_back(ev);
        }
        return std::make_shared<REX::eventBelongs>(loc_evs, REX::external_legs_comparator);
    }

    std::shared_ptr<std::vector<FORTRANFPTYPE>> amp(int &nEvt, int &nPar, int &nMom, std::vector<FORTRANFPTYPE> &momenta, std::vector<FORTRANFPTYPE> &alphaS, std::vector<FORTRANFPTYPE> &rndHel, std::vector<FORTRANFPTYPE> &rndCol, std::vector<int> &selHel, std::vector<int> &selCol, unsigned int &chanId, bool &goodHel)
    {
        CppObjectInFortran *bridgeInst;
        auto evalScatAmps = std::make_shared<std::vector<FORTRANFPTYPE>>(nEvt);
        fbridgecreate_(&bridgeInst, &nEvt, &nPar, &nMom);
        fbridgesequence_nomultichannel_(&bridgeInst, &momenta.at(0), &alphaS.at(0), &rndHel[0], &rndCol[0], &evalScatAmps->at(0), &selHel[0], &selCol[0], &goodHel);
        fbridgedelete_(&bridgeInst);
        return evalScatAmps;
    }

    rwgt::fBridge bridgeConstr(size_t warpSize)
    {
        rwgt::fBridge bridge;
        bridge.setBridge(amp);
        bridge.nWarp = warpSize;
        return bridge;
    }

    std::shared_ptr<REX::tea::procReweightor> make_reweightor(size_t warpSize)
    {
        static rwgt::fBridge bridge = bridgeConstr(warpSize);
        auto comp = get_comp();
        return std::make_shared<REX::tea::procReweightor>(bridge.getAmp(), comp);
    }
}
