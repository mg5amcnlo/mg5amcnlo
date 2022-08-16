//==========================================================================
// This file has been automatically generated for C++
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef Parameters_sm_ckm_H
#define Parameters_sm_ckm_H

#include "Complex.h"

// More includes if plugin is used from within Pythia8.
#ifdef PYTHIA8
#include "Pythia8/ParticleData.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/SusyLesHouches.h"
#endif 

#include "read_slha.h"
using namespace std; 

class Parameters_sm_ckm
{
  public:

    // Define "zero"
    double zero, ZERO; 
    // Model parameters independent of aS
    double mdl_WH, mdl_WW, mdl_WZ, mdl_WT, mdl_ymtau, mdl_ymt, mdl_ymb,
        mdl_etaWS, mdl_rhoWS, mdl_AWS, mdl_lamWS, aS, mdl_Gf, aEWM1, mdl_MH,
        mdl_MZ, mdl_MTA, mdl_MT, mdl_MB, mdl_CKM3x3, mdl_conjg__CKM3x3,
        mdl_lamWS__exp__2, mdl_lamWS__exp__3, mdl_MZ__exp__2, mdl_MZ__exp__4,
        mdl_sqrt__2, mdl_MH__exp__2, mdl_aEW, mdl_MW, mdl_sqrt__aEW, mdl_ee,
        mdl_MW__exp__2, mdl_sw2, mdl_cw, mdl_sqrt__sw2, mdl_sw, mdl_g1, mdl_gw,
        mdl_vev, mdl_vev__exp__2, mdl_lam, mdl_yb, mdl_yt, mdl_ytau, mdl_muH,
        mdl_ee__exp__2, mdl_sw__exp__2, mdl_cw__exp__2;
    Complex<double> mdl_CKM1x1, mdl_CKM1x2, mdl_complexi, mdl_CKM1x3,
        mdl_CKM2x1, mdl_CKM2x2, mdl_CKM2x3, mdl_CKM3x1, mdl_CKM3x2,
        mdl_conjg__CKM1x3, mdl_conjg__CKM2x3, mdl_conjg__CKM2x1,
        mdl_conjg__CKM3x1, mdl_conjg__CKM2x2, mdl_conjg__CKM3x2,
        mdl_conjg__CKM1x1, mdl_conjg__CKM1x2, mdl_I1x31, mdl_I1x32, mdl_I1x33,
        mdl_I2x13, mdl_I2x23, mdl_I2x33, mdl_I3x31, mdl_I3x32, mdl_I3x33,
        mdl_I4x13, mdl_I4x23, mdl_I4x33;
    // Model parameters dependent on aS
    double mdl_sqrt__aS, G, mdl_G__exp__2; 
    // Model couplings independent of aS
    Complex<double> GC_1, GC_2, GC_3, GC_4, GC_5, GC_6, GC_7, GC_8, GC_9,
        GC_13, GC_14, GC_15, GC_17, GC_19, GC_21, GC_25, GC_26, GC_27, GC_28,
        GC_29, GC_30, GC_31, GC_32, GC_33, GC_34, GC_35, GC_36, GC_37, GC_38,
        GC_39, GC_43, GC_44, GC_46, GC_47, GC_48, GC_50, GC_51, GC_52, GC_53,
        GC_54, GC_55, GC_56, GC_57, GC_58, GC_59, GC_60, GC_61, GC_62, GC_63,
        GC_64, GC_65, GC_66, GC_67, GC_68, GC_69, GC_70, GC_71, GC_72, GC_73,
        GC_74, GC_75, GC_76, GC_77, GC_78, GC_79, GC_80, GC_81, GC_82, GC_83,
        GC_94, GC_95, GC_96, GC_97, GC_98, GC_99, GC_100, GC_101, GC_102,
        GC_106, GC_108;
    // Model couplings dependent on aS
    Complex<double> GC_12, GC_11, GC_10; 

    // Further initialization functions if plugin is used from within Pythia8.
    #ifdef PYTHIA8
    // Set parameters that are unchanged during the run
    void setIndependentParameters(Pythia8::ParticleData * & pd,
        Pythia8::Couplings * & csm, Pythia8::SusyLesHouches * & slhaPtr);
    // Set parameters that are changed event by event
    void setDependentParameters(Pythia8::ParticleData * & pd,
        Pythia8::Couplings * & csm, Pythia8::SusyLesHouches * & slhaPtr, double
        alpS);
    #endif

    // Set parameters that are unchanged during the run
    void setIndependentParameters(SLHAReader& slha); 
    // Set couplings that are unchanged during the run
    void setIndependentCouplings(); 
    // Set parameters that are changed event by event
    void setDependentParameters(); 
    // Set couplings that are changed event by event
    void setDependentCouplings(); 

    // Print parameters that are unchanged during the run
    void printIndependentParameters(); 
    // Print couplings that are unchanged during the run
    void printIndependentCouplings(); 
    // Print parameters that are changed event by event
    void printDependentParameters(); 
    // Print couplings that are changed event by event
    void printDependentCouplings(); 

}; 

#endif  // Parameters_sm_ckm_H

