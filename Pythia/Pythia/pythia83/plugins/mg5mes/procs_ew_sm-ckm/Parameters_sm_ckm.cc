//==========================================================================
// This file has been automatically generated for C++ by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include <iostream> 
#include <iomanip> 
#include "Parameters_sm_ckm.h"

void Parameters_sm_ckm::setIndependentParameters(SLHAReader& slha)
{
  // Define "zero"
  zero = 0; 
  ZERO = 0; 
  // Prepare a vector for indices
  vector<int> indices(2, 0); 
  if (slha.is_valid())
    mdl_WH = slha.get_block_entry("decay", 25, 6.382339e-03); 
  if (slha.is_valid())
    mdl_WW = slha.get_block_entry("decay", 24, 2.047600e+00); 
  if (slha.is_valid())
    mdl_WZ = slha.get_block_entry("decay", 23, 2.441404e+00); 
  if (slha.is_valid())
    mdl_WT = slha.get_block_entry("decay", 6, 1.491500e+00); 
  if (slha.is_valid())
    mdl_ymtau = slha.get_block_entry("yukawa", 15, 1.777000e+00); 
  if (slha.is_valid())
    mdl_ymt = slha.get_block_entry("yukawa", 6, 1.730000e+02); 
  if (slha.is_valid())
    mdl_ymb = slha.get_block_entry("yukawa", 5, 4.700000e+00); 
  if (slha.is_valid())
    mdl_etaWS = slha.get_block_entry("wolfenstein", 4, 3.410000e-01); 
  if (slha.is_valid())
    mdl_rhoWS = slha.get_block_entry("wolfenstein", 3, 1.320000e-01); 
  if (slha.is_valid())
    mdl_AWS = slha.get_block_entry("wolfenstein", 2, 8.080000e-01); 
  if (slha.is_valid())
    mdl_lamWS = slha.get_block_entry("wolfenstein", 1, 2.253000e-01); 
  if (slha.is_valid())
    aS = slha.get_block_entry("sminputs", 3, 1.180000e-01); 
  if (slha.is_valid())
    mdl_Gf = slha.get_block_entry("sminputs", 2, 1.166390e-05); 
  if (slha.is_valid())
    aEWM1 = slha.get_block_entry("sminputs", 1, 1.325070e+02); 
  if (slha.is_valid())
    mdl_MH = slha.get_block_entry("mass", 25, 1.250000e+02); 
  if (slha.is_valid())
    mdl_MZ = slha.get_block_entry("mass", 23, 9.118800e+01); 
  if (slha.is_valid())
    mdl_MTA = slha.get_block_entry("mass", 15, 1.777000e+00); 
  if (slha.is_valid())
    mdl_MT = slha.get_block_entry("mass", 6, 1.730000e+02); 
  if (slha.is_valid())
    mdl_MB = slha.get_block_entry("mass", 5, 4.700000e+00); 
  mdl_CKM3x3 = 1.; 
  mdl_conjg__CKM3x3 = 1.; 
  mdl_lamWS__exp__2 = ((mdl_lamWS) * (mdl_lamWS)); 
  mdl_CKM1x1 = 1. - mdl_lamWS__exp__2/2.; 
  mdl_CKM1x2 = mdl_lamWS; 
  mdl_complexi = Complex<double> (0., 1.); 
  mdl_lamWS__exp__3 = ((mdl_lamWS) * (mdl_lamWS) * (mdl_lamWS)); 
  mdl_CKM1x3 = mdl_AWS * mdl_lamWS__exp__3 * (-(mdl_etaWS * mdl_complexi) +
      mdl_rhoWS);
  mdl_CKM2x1 = -mdl_lamWS; 
  mdl_CKM2x2 = 1. - mdl_lamWS__exp__2/2.; 
  mdl_CKM2x3 = mdl_AWS * mdl_lamWS__exp__2; 
  mdl_CKM3x1 = mdl_AWS * mdl_lamWS__exp__3 * (1. - mdl_etaWS * mdl_complexi -
      mdl_rhoWS);
  mdl_CKM3x2 = -(mdl_AWS * mdl_lamWS__exp__2); 
  mdl_MZ__exp__2 = ((mdl_MZ) * (mdl_MZ)); 
  mdl_MZ__exp__4 = ((mdl_MZ) * (mdl_MZ) * (mdl_MZ) * (mdl_MZ)); 
  mdl_sqrt__2 = sqrt(2.); 
  mdl_MH__exp__2 = ((mdl_MH) * (mdl_MH)); 
  mdl_conjg__CKM1x3 = conj(mdl_CKM1x3); 
  mdl_conjg__CKM2x3 = conj(mdl_CKM2x3); 
  mdl_conjg__CKM2x1 = conj(mdl_CKM2x1); 
  mdl_conjg__CKM3x1 = conj(mdl_CKM3x1); 
  mdl_conjg__CKM2x2 = conj(mdl_CKM2x2); 
  mdl_conjg__CKM3x2 = conj(mdl_CKM3x2); 
  mdl_conjg__CKM1x1 = conj(mdl_CKM1x1); 
  mdl_conjg__CKM1x2 = conj(mdl_CKM1x2); 
  mdl_aEW = 1./aEWM1; 
  mdl_MW = sqrt(mdl_MZ__exp__2/2. + sqrt(mdl_MZ__exp__4/4. - (mdl_aEW * M_PI *
      mdl_MZ__exp__2)/(mdl_Gf * mdl_sqrt__2)));
  mdl_sqrt__aEW = sqrt(mdl_aEW); 
  mdl_ee = 2. * mdl_sqrt__aEW * sqrt(M_PI); 
  mdl_MW__exp__2 = ((mdl_MW) * (mdl_MW)); 
  mdl_sw2 = 1. - mdl_MW__exp__2/mdl_MZ__exp__2; 
  mdl_cw = sqrt(1. - mdl_sw2); 
  mdl_sqrt__sw2 = sqrt(mdl_sw2); 
  mdl_sw = mdl_sqrt__sw2; 
  mdl_g1 = mdl_ee/mdl_cw; 
  mdl_gw = mdl_ee/mdl_sw; 
  mdl_vev = (2. * mdl_MW * mdl_sw)/mdl_ee; 
  mdl_vev__exp__2 = ((mdl_vev) * (mdl_vev)); 
  mdl_lam = mdl_MH__exp__2/(2. * mdl_vev__exp__2); 
  mdl_yb = (mdl_ymb * mdl_sqrt__2)/mdl_vev; 
  mdl_yt = (mdl_ymt * mdl_sqrt__2)/mdl_vev; 
  mdl_ytau = (mdl_ymtau * mdl_sqrt__2)/mdl_vev; 
  mdl_muH = sqrt(mdl_lam * mdl_vev__exp__2); 
  mdl_I1x31 = mdl_yb * mdl_conjg__CKM1x3; 
  mdl_I1x32 = mdl_yb * mdl_conjg__CKM2x3; 
  mdl_I1x33 = mdl_yb * mdl_conjg__CKM3x3; 
  mdl_I2x13 = mdl_yt * mdl_conjg__CKM3x1; 
  mdl_I2x23 = mdl_yt * mdl_conjg__CKM3x2; 
  mdl_I2x33 = mdl_yt * mdl_conjg__CKM3x3; 
  mdl_I3x31 = mdl_CKM3x1 * mdl_yt; 
  mdl_I3x32 = mdl_CKM3x2 * mdl_yt; 
  mdl_I3x33 = mdl_CKM3x3 * mdl_yt; 
  mdl_I4x13 = mdl_CKM1x3 * mdl_yb; 
  mdl_I4x23 = mdl_CKM2x3 * mdl_yb; 
  mdl_I4x33 = mdl_CKM3x3 * mdl_yb; 
  mdl_ee__exp__2 = ((mdl_ee) * (mdl_ee)); 
  mdl_sw__exp__2 = ((mdl_sw) * (mdl_sw)); 
  mdl_cw__exp__2 = ((mdl_cw) * (mdl_cw)); 
}
void Parameters_sm_ckm::setIndependentCouplings()
{
  GC_1 = -(mdl_ee * mdl_complexi)/3.; 
  GC_2 = (2. * mdl_ee * mdl_complexi)/3.; 
  GC_3 = -(mdl_ee * mdl_complexi); 
  GC_4 = mdl_ee * mdl_complexi; 
  GC_5 = mdl_ee__exp__2 * mdl_complexi; 
  GC_6 = 2. * mdl_ee__exp__2 * mdl_complexi; 
  GC_7 = -mdl_ee__exp__2/(2. * mdl_cw); 
  GC_8 = (mdl_ee__exp__2 * mdl_complexi)/(2. * mdl_cw); 
  GC_9 = mdl_ee__exp__2/(2. * mdl_cw); 
  GC_13 = mdl_I1x31; 
  GC_14 = mdl_I1x32; 
  GC_15 = mdl_I1x33; 
  GC_17 = -mdl_I2x13; 
  GC_19 = -mdl_I2x23; 
  GC_21 = -mdl_I2x33; 
  GC_25 = mdl_I3x31; 
  GC_26 = mdl_I3x32; 
  GC_27 = mdl_I3x33; 
  GC_28 = -mdl_I4x13; 
  GC_29 = -mdl_I4x23; 
  GC_30 = -mdl_I4x33; 
  GC_31 = -2. * mdl_complexi * mdl_lam; 
  GC_32 = -4. * mdl_complexi * mdl_lam; 
  GC_33 = -6. * mdl_complexi * mdl_lam; 
  GC_34 = (mdl_ee__exp__2 * mdl_complexi)/(2. * mdl_sw__exp__2); 
  GC_35 = -((mdl_ee__exp__2 * mdl_complexi)/mdl_sw__exp__2); 
  GC_36 = (mdl_cw__exp__2 * mdl_ee__exp__2 * mdl_complexi)/mdl_sw__exp__2; 
  GC_37 = -mdl_ee/(2. * mdl_sw); 
  GC_38 = -(mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_39 = (mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_43 = (mdl_CKM1x3 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_44 = (mdl_CKM2x1 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_46 = (mdl_CKM2x3 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_47 = (mdl_CKM3x1 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_48 = (mdl_CKM3x2 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_50 = -(mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_51 = (mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_52 = -((mdl_cw * mdl_ee * mdl_complexi)/mdl_sw); 
  GC_53 = (mdl_cw * mdl_ee * mdl_complexi)/mdl_sw; 
  GC_54 = -mdl_ee__exp__2/(2. * mdl_sw); 
  GC_55 = -(mdl_ee__exp__2 * mdl_complexi)/(2. * mdl_sw); 
  GC_56 = mdl_ee__exp__2/(2. * mdl_sw); 
  GC_57 = (-2. * mdl_cw * mdl_ee__exp__2 * mdl_complexi)/mdl_sw; 
  GC_58 = -(mdl_ee * mdl_complexi * mdl_sw)/(6. * mdl_cw); 
  GC_59 = (mdl_ee * mdl_complexi * mdl_sw)/(2. * mdl_cw); 
  GC_60 = -(mdl_cw * mdl_ee)/(2. * mdl_sw) - (mdl_ee * mdl_sw)/(2. * mdl_cw); 
  GC_61 = -(mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw) + (mdl_ee *
      mdl_complexi * mdl_sw)/(2. * mdl_cw);
  GC_62 = (mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw) + (mdl_ee *
      mdl_complexi * mdl_sw)/(2. * mdl_cw);
  GC_63 = (mdl_cw * mdl_ee__exp__2 * mdl_complexi)/mdl_sw - (mdl_ee__exp__2 *
      mdl_complexi * mdl_sw)/mdl_cw;
  GC_64 = -(mdl_ee__exp__2 * mdl_complexi) + (mdl_cw__exp__2 * mdl_ee__exp__2 *
      mdl_complexi)/(2. * mdl_sw__exp__2) + (mdl_ee__exp__2 * mdl_complexi *
      mdl_sw__exp__2)/(2. * mdl_cw__exp__2);
  GC_65 = mdl_ee__exp__2 * mdl_complexi + (mdl_cw__exp__2 * mdl_ee__exp__2 *
      mdl_complexi)/(2. * mdl_sw__exp__2) + (mdl_ee__exp__2 * mdl_complexi *
      mdl_sw__exp__2)/(2. * mdl_cw__exp__2);
  GC_66 = -(mdl_ee__exp__2 * mdl_vev)/(2. * mdl_cw); 
  GC_67 = (mdl_ee__exp__2 * mdl_vev)/(2. * mdl_cw); 
  GC_68 = -2. * mdl_complexi * mdl_lam * mdl_vev; 
  GC_69 = -6. * mdl_complexi * mdl_lam * mdl_vev; 
  GC_70 = -(mdl_ee__exp__2 * mdl_vev)/(4. * mdl_sw__exp__2); 
  GC_71 = -(mdl_ee__exp__2 * mdl_complexi * mdl_vev)/(4. * mdl_sw__exp__2); 
  GC_72 = (mdl_ee__exp__2 * mdl_complexi * mdl_vev)/(2. * mdl_sw__exp__2); 
  GC_73 = (mdl_ee__exp__2 * mdl_vev)/(4. * mdl_sw__exp__2); 
  GC_74 = -(mdl_ee__exp__2 * mdl_vev)/(2. * mdl_sw); 
  GC_75 = (mdl_ee__exp__2 * mdl_vev)/(2. * mdl_sw); 
  GC_76 = -(mdl_ee__exp__2 * mdl_vev)/(4. * mdl_cw) - (mdl_cw * mdl_ee__exp__2
      * mdl_vev)/(4. * mdl_sw__exp__2);
  GC_77 = (mdl_ee__exp__2 * mdl_vev)/(4. * mdl_cw) - (mdl_cw * mdl_ee__exp__2 *
      mdl_vev)/(4. * mdl_sw__exp__2);
  GC_78 = -(mdl_ee__exp__2 * mdl_vev)/(4. * mdl_cw) + (mdl_cw * mdl_ee__exp__2
      * mdl_vev)/(4. * mdl_sw__exp__2);
  GC_79 = (mdl_ee__exp__2 * mdl_vev)/(4. * mdl_cw) + (mdl_cw * mdl_ee__exp__2 *
      mdl_vev)/(4. * mdl_sw__exp__2);
  GC_80 = -(mdl_ee__exp__2 * mdl_complexi * mdl_vev)/2. - (mdl_cw__exp__2 *
      mdl_ee__exp__2 * mdl_complexi * mdl_vev)/(4. * mdl_sw__exp__2) -
      (mdl_ee__exp__2 * mdl_complexi * mdl_sw__exp__2 * mdl_vev)/(4. *
      mdl_cw__exp__2);
  GC_81 = mdl_ee__exp__2 * mdl_complexi * mdl_vev + (mdl_cw__exp__2 *
      mdl_ee__exp__2 * mdl_complexi * mdl_vev)/(2. * mdl_sw__exp__2) +
      (mdl_ee__exp__2 * mdl_complexi * mdl_sw__exp__2 * mdl_vev)/(2. *
      mdl_cw__exp__2);
  GC_82 = -(mdl_yb/mdl_sqrt__2); 
  GC_83 = -((mdl_complexi * mdl_yb)/mdl_sqrt__2); 
  GC_94 = -((mdl_complexi * mdl_yt)/mdl_sqrt__2); 
  GC_95 = mdl_yt/mdl_sqrt__2; 
  GC_96 = -mdl_ytau; 
  GC_97 = mdl_ytau; 
  GC_98 = -(mdl_ytau/mdl_sqrt__2); 
  GC_99 = -((mdl_complexi * mdl_ytau)/mdl_sqrt__2); 
  GC_100 = (mdl_ee * mdl_complexi * mdl_conjg__CKM1x1)/(mdl_sw * mdl_sqrt__2); 
  GC_101 = (mdl_ee * mdl_complexi * mdl_conjg__CKM1x2)/(mdl_sw * mdl_sqrt__2); 
  GC_102 = (mdl_ee * mdl_complexi * mdl_conjg__CKM1x3)/(mdl_sw * mdl_sqrt__2); 
  GC_106 = (mdl_ee * mdl_complexi * mdl_conjg__CKM3x1)/(mdl_sw * mdl_sqrt__2); 
  GC_108 = (mdl_ee * mdl_complexi * mdl_conjg__CKM3x3)/(mdl_sw * mdl_sqrt__2); 
}
void Parameters_sm_ckm::setDependentParameters()
{
  mdl_sqrt__aS = sqrt(aS); 
  G = 2. * mdl_sqrt__aS * sqrt(M_PI); 
  mdl_G__exp__2 = ((G) * (G)); 
}
void Parameters_sm_ckm::setDependentCouplings()
{
  GC_12 = mdl_complexi * mdl_G__exp__2; 
  GC_11 = mdl_complexi * G; 
  GC_10 = -G; 
}

// Routines for printing out parameters
void Parameters_sm_ckm::printIndependentParameters()
{
  cout <<  "sm_ckm model parameters independent of event kinematics:" << endl; 
  cout << setprecision(20) <<  "mdl_WH " <<  "= " << setprecision(10) << mdl_WH
      << endl;
  cout << setprecision(20) <<  "mdl_WW " <<  "= " << setprecision(10) << mdl_WW
      << endl;
  cout << setprecision(20) <<  "mdl_WZ " <<  "= " << setprecision(10) << mdl_WZ
      << endl;
  cout << setprecision(20) <<  "mdl_WT " <<  "= " << setprecision(10) << mdl_WT
      << endl;
  cout << setprecision(20) <<  "mdl_ymtau " <<  "= " << setprecision(10) <<
      mdl_ymtau << endl;
  cout << setprecision(20) <<  "mdl_ymt " <<  "= " << setprecision(10) <<
      mdl_ymt << endl;
  cout << setprecision(20) <<  "mdl_ymb " <<  "= " << setprecision(10) <<
      mdl_ymb << endl;
  cout << setprecision(20) <<  "mdl_etaWS " <<  "= " << setprecision(10) <<
      mdl_etaWS << endl;
  cout << setprecision(20) <<  "mdl_rhoWS " <<  "= " << setprecision(10) <<
      mdl_rhoWS << endl;
  cout << setprecision(20) <<  "mdl_AWS " <<  "= " << setprecision(10) <<
      mdl_AWS << endl;
  cout << setprecision(20) <<  "mdl_lamWS " <<  "= " << setprecision(10) <<
      mdl_lamWS << endl;
  cout << setprecision(20) <<  "aS " <<  "= " << setprecision(10) << aS <<
      endl;
  cout << setprecision(20) <<  "mdl_Gf " <<  "= " << setprecision(10) << mdl_Gf
      << endl;
  cout << setprecision(20) <<  "aEWM1 " <<  "= " << setprecision(10) << aEWM1
      << endl;
  cout << setprecision(20) <<  "mdl_MH " <<  "= " << setprecision(10) << mdl_MH
      << endl;
  cout << setprecision(20) <<  "mdl_MZ " <<  "= " << setprecision(10) << mdl_MZ
      << endl;
  cout << setprecision(20) <<  "mdl_MTA " <<  "= " << setprecision(10) <<
      mdl_MTA << endl;
  cout << setprecision(20) <<  "mdl_MT " <<  "= " << setprecision(10) << mdl_MT
      << endl;
  cout << setprecision(20) <<  "mdl_MB " <<  "= " << setprecision(10) << mdl_MB
      << endl;
  cout << setprecision(20) <<  "mdl_CKM3x3 " <<  "= " << setprecision(10) <<
      mdl_CKM3x3 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM3x3 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM3x3 << endl;
  cout << setprecision(20) <<  "mdl_lamWS__exp__2 " <<  "= " <<
      setprecision(10) << mdl_lamWS__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_CKM1x1 " <<  "= " << setprecision(10) <<
      mdl_CKM1x1 << endl;
  cout << setprecision(20) <<  "mdl_CKM1x2 " <<  "= " << setprecision(10) <<
      mdl_CKM1x2 << endl;
  cout << setprecision(20) <<  "mdl_complexi " <<  "= " << setprecision(10) <<
      mdl_complexi << endl;
  cout << setprecision(20) <<  "mdl_lamWS__exp__3 " <<  "= " <<
      setprecision(10) << mdl_lamWS__exp__3 << endl;
  cout << setprecision(20) <<  "mdl_CKM1x3 " <<  "= " << setprecision(10) <<
      mdl_CKM1x3 << endl;
  cout << setprecision(20) <<  "mdl_CKM2x1 " <<  "= " << setprecision(10) <<
      mdl_CKM2x1 << endl;
  cout << setprecision(20) <<  "mdl_CKM2x2 " <<  "= " << setprecision(10) <<
      mdl_CKM2x2 << endl;
  cout << setprecision(20) <<  "mdl_CKM2x3 " <<  "= " << setprecision(10) <<
      mdl_CKM2x3 << endl;
  cout << setprecision(20) <<  "mdl_CKM3x1 " <<  "= " << setprecision(10) <<
      mdl_CKM3x1 << endl;
  cout << setprecision(20) <<  "mdl_CKM3x2 " <<  "= " << setprecision(10) <<
      mdl_CKM3x2 << endl;
  cout << setprecision(20) <<  "mdl_MZ__exp__2 " <<  "= " << setprecision(10)
      << mdl_MZ__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_MZ__exp__4 " <<  "= " << setprecision(10)
      << mdl_MZ__exp__4 << endl;
  cout << setprecision(20) <<  "mdl_sqrt__2 " <<  "= " << setprecision(10) <<
      mdl_sqrt__2 << endl;
  cout << setprecision(20) <<  "mdl_MH__exp__2 " <<  "= " << setprecision(10)
      << mdl_MH__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM1x3 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM1x3 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM2x3 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM2x3 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM2x1 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM2x1 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM3x1 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM3x1 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM2x2 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM2x2 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM3x2 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM3x2 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM1x1 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM1x1 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM1x2 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM1x2 << endl;
  cout << setprecision(20) <<  "mdl_aEW " <<  "= " << setprecision(10) <<
      mdl_aEW << endl;
  cout << setprecision(20) <<  "mdl_MW " <<  "= " << setprecision(10) << mdl_MW
      << endl;
  cout << setprecision(20) <<  "mdl_sqrt__aEW " <<  "= " << setprecision(10) <<
      mdl_sqrt__aEW << endl;
  cout << setprecision(20) <<  "mdl_ee " <<  "= " << setprecision(10) << mdl_ee
      << endl;
  cout << setprecision(20) <<  "mdl_MW__exp__2 " <<  "= " << setprecision(10)
      << mdl_MW__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_sw2 " <<  "= " << setprecision(10) <<
      mdl_sw2 << endl;
  cout << setprecision(20) <<  "mdl_cw " <<  "= " << setprecision(10) << mdl_cw
      << endl;
  cout << setprecision(20) <<  "mdl_sqrt__sw2 " <<  "= " << setprecision(10) <<
      mdl_sqrt__sw2 << endl;
  cout << setprecision(20) <<  "mdl_sw " <<  "= " << setprecision(10) << mdl_sw
      << endl;
  cout << setprecision(20) <<  "mdl_g1 " <<  "= " << setprecision(10) << mdl_g1
      << endl;
  cout << setprecision(20) <<  "mdl_gw " <<  "= " << setprecision(10) << mdl_gw
      << endl;
  cout << setprecision(20) <<  "mdl_vev " <<  "= " << setprecision(10) <<
      mdl_vev << endl;
  cout << setprecision(20) <<  "mdl_vev__exp__2 " <<  "= " << setprecision(10)
      << mdl_vev__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_lam " <<  "= " << setprecision(10) <<
      mdl_lam << endl;
  cout << setprecision(20) <<  "mdl_yb " <<  "= " << setprecision(10) << mdl_yb
      << endl;
  cout << setprecision(20) <<  "mdl_yt " <<  "= " << setprecision(10) << mdl_yt
      << endl;
  cout << setprecision(20) <<  "mdl_ytau " <<  "= " << setprecision(10) <<
      mdl_ytau << endl;
  cout << setprecision(20) <<  "mdl_muH " <<  "= " << setprecision(10) <<
      mdl_muH << endl;
  cout << setprecision(20) <<  "mdl_I1x31 " <<  "= " << setprecision(10) <<
      mdl_I1x31 << endl;
  cout << setprecision(20) <<  "mdl_I1x32 " <<  "= " << setprecision(10) <<
      mdl_I1x32 << endl;
  cout << setprecision(20) <<  "mdl_I1x33 " <<  "= " << setprecision(10) <<
      mdl_I1x33 << endl;
  cout << setprecision(20) <<  "mdl_I2x13 " <<  "= " << setprecision(10) <<
      mdl_I2x13 << endl;
  cout << setprecision(20) <<  "mdl_I2x23 " <<  "= " << setprecision(10) <<
      mdl_I2x23 << endl;
  cout << setprecision(20) <<  "mdl_I2x33 " <<  "= " << setprecision(10) <<
      mdl_I2x33 << endl;
  cout << setprecision(20) <<  "mdl_I3x31 " <<  "= " << setprecision(10) <<
      mdl_I3x31 << endl;
  cout << setprecision(20) <<  "mdl_I3x32 " <<  "= " << setprecision(10) <<
      mdl_I3x32 << endl;
  cout << setprecision(20) <<  "mdl_I3x33 " <<  "= " << setprecision(10) <<
      mdl_I3x33 << endl;
  cout << setprecision(20) <<  "mdl_I4x13 " <<  "= " << setprecision(10) <<
      mdl_I4x13 << endl;
  cout << setprecision(20) <<  "mdl_I4x23 " <<  "= " << setprecision(10) <<
      mdl_I4x23 << endl;
  cout << setprecision(20) <<  "mdl_I4x33 " <<  "= " << setprecision(10) <<
      mdl_I4x33 << endl;
  cout << setprecision(20) <<  "mdl_ee__exp__2 " <<  "= " << setprecision(10)
      << mdl_ee__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_sw__exp__2 " <<  "= " << setprecision(10)
      << mdl_sw__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_cw__exp__2 " <<  "= " << setprecision(10)
      << mdl_cw__exp__2 << endl;
}
void Parameters_sm_ckm::printIndependentCouplings()
{
  cout <<  "sm_ckm model couplings independent of event kinematics:" << endl; 
  cout << setprecision(20) <<  "GC_1 " <<  "= " << setprecision(10) << GC_1 <<
      endl;
  cout << setprecision(20) <<  "GC_2 " <<  "= " << setprecision(10) << GC_2 <<
      endl;
  cout << setprecision(20) <<  "GC_3 " <<  "= " << setprecision(10) << GC_3 <<
      endl;
  cout << setprecision(20) <<  "GC_4 " <<  "= " << setprecision(10) << GC_4 <<
      endl;
  cout << setprecision(20) <<  "GC_5 " <<  "= " << setprecision(10) << GC_5 <<
      endl;
  cout << setprecision(20) <<  "GC_6 " <<  "= " << setprecision(10) << GC_6 <<
      endl;
  cout << setprecision(20) <<  "GC_7 " <<  "= " << setprecision(10) << GC_7 <<
      endl;
  cout << setprecision(20) <<  "GC_8 " <<  "= " << setprecision(10) << GC_8 <<
      endl;
  cout << setprecision(20) <<  "GC_9 " <<  "= " << setprecision(10) << GC_9 <<
      endl;
  cout << setprecision(20) <<  "GC_13 " <<  "= " << setprecision(10) << GC_13
      << endl;
  cout << setprecision(20) <<  "GC_14 " <<  "= " << setprecision(10) << GC_14
      << endl;
  cout << setprecision(20) <<  "GC_15 " <<  "= " << setprecision(10) << GC_15
      << endl;
  cout << setprecision(20) <<  "GC_17 " <<  "= " << setprecision(10) << GC_17
      << endl;
  cout << setprecision(20) <<  "GC_19 " <<  "= " << setprecision(10) << GC_19
      << endl;
  cout << setprecision(20) <<  "GC_21 " <<  "= " << setprecision(10) << GC_21
      << endl;
  cout << setprecision(20) <<  "GC_25 " <<  "= " << setprecision(10) << GC_25
      << endl;
  cout << setprecision(20) <<  "GC_26 " <<  "= " << setprecision(10) << GC_26
      << endl;
  cout << setprecision(20) <<  "GC_27 " <<  "= " << setprecision(10) << GC_27
      << endl;
  cout << setprecision(20) <<  "GC_28 " <<  "= " << setprecision(10) << GC_28
      << endl;
  cout << setprecision(20) <<  "GC_29 " <<  "= " << setprecision(10) << GC_29
      << endl;
  cout << setprecision(20) <<  "GC_30 " <<  "= " << setprecision(10) << GC_30
      << endl;
  cout << setprecision(20) <<  "GC_31 " <<  "= " << setprecision(10) << GC_31
      << endl;
  cout << setprecision(20) <<  "GC_32 " <<  "= " << setprecision(10) << GC_32
      << endl;
  cout << setprecision(20) <<  "GC_33 " <<  "= " << setprecision(10) << GC_33
      << endl;
  cout << setprecision(20) <<  "GC_34 " <<  "= " << setprecision(10) << GC_34
      << endl;
  cout << setprecision(20) <<  "GC_35 " <<  "= " << setprecision(10) << GC_35
      << endl;
  cout << setprecision(20) <<  "GC_36 " <<  "= " << setprecision(10) << GC_36
      << endl;
  cout << setprecision(20) <<  "GC_37 " <<  "= " << setprecision(10) << GC_37
      << endl;
  cout << setprecision(20) <<  "GC_38 " <<  "= " << setprecision(10) << GC_38
      << endl;
  cout << setprecision(20) <<  "GC_39 " <<  "= " << setprecision(10) << GC_39
      << endl;
  cout << setprecision(20) <<  "GC_43 " <<  "= " << setprecision(10) << GC_43
      << endl;
  cout << setprecision(20) <<  "GC_44 " <<  "= " << setprecision(10) << GC_44
      << endl;
  cout << setprecision(20) <<  "GC_46 " <<  "= " << setprecision(10) << GC_46
      << endl;
  cout << setprecision(20) <<  "GC_47 " <<  "= " << setprecision(10) << GC_47
      << endl;
  cout << setprecision(20) <<  "GC_48 " <<  "= " << setprecision(10) << GC_48
      << endl;
  cout << setprecision(20) <<  "GC_50 " <<  "= " << setprecision(10) << GC_50
      << endl;
  cout << setprecision(20) <<  "GC_51 " <<  "= " << setprecision(10) << GC_51
      << endl;
  cout << setprecision(20) <<  "GC_52 " <<  "= " << setprecision(10) << GC_52
      << endl;
  cout << setprecision(20) <<  "GC_53 " <<  "= " << setprecision(10) << GC_53
      << endl;
  cout << setprecision(20) <<  "GC_54 " <<  "= " << setprecision(10) << GC_54
      << endl;
  cout << setprecision(20) <<  "GC_55 " <<  "= " << setprecision(10) << GC_55
      << endl;
  cout << setprecision(20) <<  "GC_56 " <<  "= " << setprecision(10) << GC_56
      << endl;
  cout << setprecision(20) <<  "GC_57 " <<  "= " << setprecision(10) << GC_57
      << endl;
  cout << setprecision(20) <<  "GC_58 " <<  "= " << setprecision(10) << GC_58
      << endl;
  cout << setprecision(20) <<  "GC_59 " <<  "= " << setprecision(10) << GC_59
      << endl;
  cout << setprecision(20) <<  "GC_60 " <<  "= " << setprecision(10) << GC_60
      << endl;
  cout << setprecision(20) <<  "GC_61 " <<  "= " << setprecision(10) << GC_61
      << endl;
  cout << setprecision(20) <<  "GC_62 " <<  "= " << setprecision(10) << GC_62
      << endl;
  cout << setprecision(20) <<  "GC_63 " <<  "= " << setprecision(10) << GC_63
      << endl;
  cout << setprecision(20) <<  "GC_64 " <<  "= " << setprecision(10) << GC_64
      << endl;
  cout << setprecision(20) <<  "GC_65 " <<  "= " << setprecision(10) << GC_65
      << endl;
  cout << setprecision(20) <<  "GC_66 " <<  "= " << setprecision(10) << GC_66
      << endl;
  cout << setprecision(20) <<  "GC_67 " <<  "= " << setprecision(10) << GC_67
      << endl;
  cout << setprecision(20) <<  "GC_68 " <<  "= " << setprecision(10) << GC_68
      << endl;
  cout << setprecision(20) <<  "GC_69 " <<  "= " << setprecision(10) << GC_69
      << endl;
  cout << setprecision(20) <<  "GC_70 " <<  "= " << setprecision(10) << GC_70
      << endl;
  cout << setprecision(20) <<  "GC_71 " <<  "= " << setprecision(10) << GC_71
      << endl;
  cout << setprecision(20) <<  "GC_72 " <<  "= " << setprecision(10) << GC_72
      << endl;
  cout << setprecision(20) <<  "GC_73 " <<  "= " << setprecision(10) << GC_73
      << endl;
  cout << setprecision(20) <<  "GC_74 " <<  "= " << setprecision(10) << GC_74
      << endl;
  cout << setprecision(20) <<  "GC_75 " <<  "= " << setprecision(10) << GC_75
      << endl;
  cout << setprecision(20) <<  "GC_76 " <<  "= " << setprecision(10) << GC_76
      << endl;
  cout << setprecision(20) <<  "GC_77 " <<  "= " << setprecision(10) << GC_77
      << endl;
  cout << setprecision(20) <<  "GC_78 " <<  "= " << setprecision(10) << GC_78
      << endl;
  cout << setprecision(20) <<  "GC_79 " <<  "= " << setprecision(10) << GC_79
      << endl;
  cout << setprecision(20) <<  "GC_80 " <<  "= " << setprecision(10) << GC_80
      << endl;
  cout << setprecision(20) <<  "GC_81 " <<  "= " << setprecision(10) << GC_81
      << endl;
  cout << setprecision(20) <<  "GC_82 " <<  "= " << setprecision(10) << GC_82
      << endl;
  cout << setprecision(20) <<  "GC_83 " <<  "= " << setprecision(10) << GC_83
      << endl;
  cout << setprecision(20) <<  "GC_94 " <<  "= " << setprecision(10) << GC_94
      << endl;
  cout << setprecision(20) <<  "GC_95 " <<  "= " << setprecision(10) << GC_95
      << endl;
  cout << setprecision(20) <<  "GC_96 " <<  "= " << setprecision(10) << GC_96
      << endl;
  cout << setprecision(20) <<  "GC_97 " <<  "= " << setprecision(10) << GC_97
      << endl;
  cout << setprecision(20) <<  "GC_98 " <<  "= " << setprecision(10) << GC_98
      << endl;
  cout << setprecision(20) <<  "GC_99 " <<  "= " << setprecision(10) << GC_99
      << endl;
  cout << setprecision(20) <<  "GC_100 " <<  "= " << setprecision(10) << GC_100
      << endl;
  cout << setprecision(20) <<  "GC_101 " <<  "= " << setprecision(10) << GC_101
      << endl;
  cout << setprecision(20) <<  "GC_102 " <<  "= " << setprecision(10) << GC_102
      << endl;
  cout << setprecision(20) <<  "GC_106 " <<  "= " << setprecision(10) << GC_106
      << endl;
  cout << setprecision(20) <<  "GC_108 " <<  "= " << setprecision(10) << GC_108
      << endl;
}
void Parameters_sm_ckm::printDependentParameters()
{
  cout <<  "sm_ckm model parameters dependent on event kinematics:" << endl; 
  cout << setprecision(20) <<  "mdl_sqrt__aS " <<  "= " << setprecision(10) <<
      mdl_sqrt__aS << endl;
  cout << setprecision(20) <<  "G " <<  "= " << setprecision(10) << G << endl; 
  cout << setprecision(20) <<  "mdl_G__exp__2 " <<  "= " << setprecision(10) <<
      mdl_G__exp__2 << endl;
}
void Parameters_sm_ckm::printDependentCouplings()
{
  cout <<  "sm_ckm model couplings dependent on event kinematics:" << endl; 
  cout << setprecision(20) <<  "GC_12 " <<  "= " << setprecision(10) << GC_12
      << endl;
  cout << setprecision(20) <<  "GC_11 " <<  "= " << setprecision(10) << GC_11
      << endl;
  cout << setprecision(20) <<  "GC_10 " <<  "= " << setprecision(10) << GC_10
      << endl;
}

// Usage from inside Pythia8
#ifdef PYTHIA8

void Parameters_sm_ckm::setIndependentParameters(Pythia8::ParticleData * & pd, 
Pythia8::Couplings * & csm, Pythia8::SusyLesHouches * & slhaPtr)
{
  if (false)
    cout << pd << csm << slhaPtr; 
  mdl_WH = pd->mWidth(25); 
  mdl_WW = pd->mWidth(24); 
  mdl_WZ = pd->mWidth(23); 
  mdl_WT = pd->mWidth(6); 
  mdl_ymtau = pd->mRun(15, pd->m0(24)); 
  mdl_ymt = pd->mRun(6, pd->m0(24)); 
  mdl_ymb = pd->mRun(5, pd->m0(24)); 
  if( !slhaPtr->getEntry<double> ("wolfenstein", 4, mdl_etaWS))
  {
    cout <<  "Warning, setting mdl_etaWS to 3.410000e-01" << endl; 
    mdl_etaWS = 3.410000e-01; 
  }
  if( !slhaPtr->getEntry<double> ("wolfenstein", 3, mdl_rhoWS))
  {
    cout <<  "Warning, setting mdl_rhoWS to 1.320000e-01" << endl; 
    mdl_rhoWS = 1.320000e-01; 
  }
  if( !slhaPtr->getEntry<double> ("wolfenstein", 2, mdl_AWS))
  {
    cout <<  "Warning, setting mdl_AWS to 8.080000e-01" << endl; 
    mdl_AWS = 8.080000e-01; 
  }
  if( !slhaPtr->getEntry<double> ("wolfenstein", 1, mdl_lamWS))
  {
    cout <<  "Warning, setting mdl_lamWS to 2.253000e-01" << endl; 
    mdl_lamWS = 2.253000e-01; 
  }
  mdl_Gf = M_PI * csm->alphaEM(((pd->m0(23)) * (pd->m0(23)))) * ((pd->m0(23)) *
      (pd->m0(23)))/(sqrt(2.) * ((pd->m0(24)) * (pd->m0(24))) * (((pd->m0(23))
      * (pd->m0(23))) - ((pd->m0(24)) * (pd->m0(24)))));
  aEWM1 = 1./csm->alphaEM(((pd->m0(23)) * (pd->m0(23)))); 
  mdl_MH = pd->m0(25); 
  mdl_MZ = pd->m0(23); 
  mdl_MTA = pd->m0(15); 
  mdl_MT = pd->m0(6); 
  mdl_MB = pd->m0(5); 
  mdl_CKM3x3 = 1.; 
  mdl_conjg__CKM3x3 = 1.; 
  ZERO = 0.; 
  mdl_lamWS__exp__2 = ((mdl_lamWS) * (mdl_lamWS)); 
  mdl_CKM1x1 = 1. - mdl_lamWS__exp__2/2.; 
  mdl_CKM1x2 = mdl_lamWS; 
  mdl_complexi = Complex<double> (0., 1.); 
  mdl_lamWS__exp__3 = ((mdl_lamWS) * (mdl_lamWS) * (mdl_lamWS)); 
  mdl_CKM1x3 = mdl_AWS * mdl_lamWS__exp__3 * (-(mdl_etaWS * mdl_complexi) +
      mdl_rhoWS);
  mdl_CKM2x1 = -mdl_lamWS; 
  mdl_CKM2x2 = 1. - mdl_lamWS__exp__2/2.; 
  mdl_CKM2x3 = mdl_AWS * mdl_lamWS__exp__2; 
  mdl_CKM3x1 = mdl_AWS * mdl_lamWS__exp__3 * (1. - mdl_etaWS * mdl_complexi -
      mdl_rhoWS);
  mdl_CKM3x2 = -(mdl_AWS * mdl_lamWS__exp__2); 
  mdl_MZ__exp__2 = ((mdl_MZ) * (mdl_MZ)); 
  mdl_MZ__exp__4 = ((mdl_MZ) * (mdl_MZ) * (mdl_MZ) * (mdl_MZ)); 
  mdl_sqrt__2 = sqrt(2.); 
  mdl_MH__exp__2 = ((mdl_MH) * (mdl_MH)); 
  mdl_conjg__CKM1x3 = conj(mdl_CKM1x3); 
  mdl_conjg__CKM2x3 = conj(mdl_CKM2x3); 
  mdl_conjg__CKM2x1 = conj(mdl_CKM2x1); 
  mdl_conjg__CKM3x1 = conj(mdl_CKM3x1); 
  mdl_conjg__CKM2x2 = conj(mdl_CKM2x2); 
  mdl_conjg__CKM3x2 = conj(mdl_CKM3x2); 
  mdl_conjg__CKM1x1 = conj(mdl_CKM1x1); 
  mdl_conjg__CKM1x2 = conj(mdl_CKM1x2); 
  mdl_aEW = 1./aEWM1; 
  mdl_MW = sqrt(mdl_MZ__exp__2/2. + sqrt(mdl_MZ__exp__4/4. - (mdl_aEW * M_PI *
      mdl_MZ__exp__2)/(mdl_Gf * mdl_sqrt__2)));
  mdl_sqrt__aEW = sqrt(mdl_aEW); 
  mdl_ee = 2. * mdl_sqrt__aEW * sqrt(M_PI); 
  mdl_MW__exp__2 = ((mdl_MW) * (mdl_MW)); 
  mdl_sw2 = 1. - mdl_MW__exp__2/mdl_MZ__exp__2; 
  mdl_cw = sqrt(1. - mdl_sw2); 
  mdl_sqrt__sw2 = sqrt(mdl_sw2); 
  mdl_sw = mdl_sqrt__sw2; 
  mdl_g1 = mdl_ee/mdl_cw; 
  mdl_gw = mdl_ee/mdl_sw; 
  mdl_vev = (2. * mdl_MW * mdl_sw)/mdl_ee; 
  mdl_vev__exp__2 = ((mdl_vev) * (mdl_vev)); 
  mdl_lam = mdl_MH__exp__2/(2. * mdl_vev__exp__2); 
  mdl_yb = (mdl_ymb * mdl_sqrt__2)/mdl_vev; 
  mdl_yt = (mdl_ymt * mdl_sqrt__2)/mdl_vev; 
  mdl_ytau = (mdl_ymtau * mdl_sqrt__2)/mdl_vev; 
  mdl_muH = sqrt(mdl_lam * mdl_vev__exp__2); 
  mdl_I1x31 = mdl_yb * mdl_conjg__CKM1x3; 
  mdl_I1x32 = mdl_yb * mdl_conjg__CKM2x3; 
  mdl_I1x33 = mdl_yb * mdl_conjg__CKM3x3; 
  mdl_I2x13 = mdl_yt * mdl_conjg__CKM3x1; 
  mdl_I2x23 = mdl_yt * mdl_conjg__CKM3x2; 
  mdl_I2x33 = mdl_yt * mdl_conjg__CKM3x3; 
  mdl_I3x31 = mdl_CKM3x1 * mdl_yt; 
  mdl_I3x32 = mdl_CKM3x2 * mdl_yt; 
  mdl_I3x33 = mdl_CKM3x3 * mdl_yt; 
  mdl_I4x13 = mdl_CKM1x3 * mdl_yb; 
  mdl_I4x23 = mdl_CKM2x3 * mdl_yb; 
  mdl_I4x33 = mdl_CKM3x3 * mdl_yb; 
  mdl_ee__exp__2 = ((mdl_ee) * (mdl_ee)); 
  mdl_sw__exp__2 = ((mdl_sw) * (mdl_sw)); 
  mdl_cw__exp__2 = ((mdl_cw) * (mdl_cw)); 
}
void Parameters_sm_ckm::setDependentParameters(Pythia8::ParticleData * & pd, 
Pythia8::Couplings * & csm, Pythia8::SusyLesHouches * & slhaPtr, double alpS)
{
  if (false)
    cout << pd << csm << slhaPtr << alpS; 
  aS = alpS; 
  mdl_sqrt__aS = sqrt(aS); 
  G = 2. * mdl_sqrt__aS * sqrt(M_PI); 
  mdl_G__exp__2 = ((G) * (G)); 
}

#endif

