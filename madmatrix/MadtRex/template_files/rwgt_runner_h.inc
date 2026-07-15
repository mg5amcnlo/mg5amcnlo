//==========================================================================
// Copyright (C) 2023-2026 CERN
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Written originally by: Z. Wettersten (June 2024) for the MG5aMC CUDACPP plugin.
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

#ifndef _%(process_namespace)s_RUNNER_H_
#define _%(process_namespace)s_RUNNER_H_

#include "rwgt_instance.h"

namespace %(process_namespace)s {

    REX::eventBelongs get_comp();
    std::shared_ptr<std::vector<FORTRANFPTYPE>> amp( int& nEvt, int& nPar, int& nMom, std::vector<FORTRANFPTYPE>& momenta, std::vector<FORTRANFPTYPE>& alphaS, std::vector<FORTRANFPTYPE>& rndHel, std::vector<FORTRANFPTYPE>& rndCol, std::vector<int>& selHel, std::vector<int>& selCol, int& chanId, bool& goodHel );
    rwgt::fBridge bridgeConstr(size_t warpSize = 32);
    std::shared_ptr<REX::tea::procReweightor> make_reweightor(size_t warpSize = 32);

}



#endif
