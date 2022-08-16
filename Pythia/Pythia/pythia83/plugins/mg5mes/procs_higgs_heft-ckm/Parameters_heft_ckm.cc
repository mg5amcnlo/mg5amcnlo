//==========================================================================
// This file has been automatically generated for C++ by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include <iostream> 
#include <iomanip> 
#include "Parameters_heft_ckm.h"

void Parameters_heft_ckm::setIndependentParameters(SLHAReader& slha)
{
  // Define "zero"
  zero = 0; 
  ZERO = 0; 
  // Prepare a vector for indices
  vector<int> indices(2, 0); 
  if (slha.is_valid())
    mdl_WH1 = slha.get_block_entry("decay", 9000006, 6.382339e-03); 
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
    mdl_ymt = slha.get_block_entry("yukawa", 6, 1.645000e+02); 
  if (slha.is_valid())
    mdl_ymb = slha.get_block_entry("yukawa", 5, 4.200000e+00); 
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
    mdl_MP = slha.get_block_entry("mass", 9000006, 1.250001e+02); 
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
  mdl_conjg__CKM3x3 = 1.; 
  mdl_CKM3x3 = 1.; 
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
  mdl_MH__exp__4 = ((mdl_MH) * (mdl_MH) * (mdl_MH) * (mdl_MH)); 
  mdl_MT__exp__4 = ((mdl_MT) * (mdl_MT) * (mdl_MT) * (mdl_MT)); 
  mdl_MH__exp__2 = ((mdl_MH) * (mdl_MH)); 
  mdl_MT__exp__2 = ((mdl_MT) * (mdl_MT)); 
  mdl_MH__exp__12 = pow(mdl_MH, 12.); 
  mdl_MH__exp__10 = pow(mdl_MH, 10.); 
  mdl_MH__exp__8 = pow(mdl_MH, 8.); 
  mdl_MH__exp__6 = pow(mdl_MH, 6.); 
  mdl_MT__exp__6 = pow(mdl_MT, 6.); 
  mdl_conjg__CKM1x1 = conj(mdl_CKM1x1); 
  mdl_conjg__CKM1x2 = conj(mdl_CKM1x2); 
  mdl_conjg__CKM1x3 = conj(mdl_CKM1x3); 
  mdl_conjg__CKM2x1 = conj(mdl_CKM2x1); 
  mdl_conjg__CKM2x2 = conj(mdl_CKM2x2); 
  mdl_conjg__CKM2x3 = conj(mdl_CKM2x3); 
  mdl_conjg__CKM3x1 = conj(mdl_CKM3x1); 
  mdl_conjg__CKM3x2 = conj(mdl_CKM3x2); 
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
  mdl_v = (2. * mdl_MW * mdl_sw)/mdl_ee; 
  mdl_ee__exp__2 = ((mdl_ee) * (mdl_ee)); 
  mdl_MW__exp__12 = pow(mdl_MW, 12.); 
  mdl_MW__exp__10 = pow(mdl_MW, 10.); 
  mdl_MW__exp__8 = pow(mdl_MW, 8.); 
  mdl_MW__exp__6 = pow(mdl_MW, 6.); 
  mdl_MW__exp__4 = ((mdl_MW) * (mdl_MW) * (mdl_MW) * (mdl_MW)); 
  mdl_AH = (47. * mdl_ee__exp__2 * (1. - (2. * mdl_MH__exp__4)/(987. *
      mdl_MT__exp__4) - (14. * mdl_MH__exp__2)/(705. * mdl_MT__exp__2) + (213.
      * mdl_MH__exp__12)/(2.634632e7 * mdl_MW__exp__12) + (5. *
      mdl_MH__exp__10)/(119756. * mdl_MW__exp__10) + (41. *
      mdl_MH__exp__8)/(180950. * mdl_MW__exp__8) + (87. *
      mdl_MH__exp__6)/(65800. * mdl_MW__exp__6) + (57. * mdl_MH__exp__4)/(6580.
      * mdl_MW__exp__4) + (33. * mdl_MH__exp__2)/(470. * mdl_MW__exp__2)))/(72.
      * ((M_PI) * (M_PI)) * mdl_v);
  mdl_v__exp__2 = ((mdl_v) * (mdl_v)); 
  mdl_lam = mdl_MH__exp__2/(2. * mdl_v__exp__2); 
  mdl_yb = (mdl_ymb * mdl_sqrt__2)/mdl_v; 
  mdl_yt = (mdl_ymt * mdl_sqrt__2)/mdl_v; 
  mdl_ytau = (mdl_ymtau * mdl_sqrt__2)/mdl_v; 
  mdl_muH = sqrt(mdl_lam * mdl_v__exp__2); 
  mdl_gw__exp__2 = ((mdl_gw) * (mdl_gw)); 
  mdl_cw__exp__2 = ((mdl_cw) * (mdl_cw)); 
  mdl_sw__exp__2 = ((mdl_sw) * (mdl_sw)); 
}
void Parameters_heft_ckm::setIndependentCouplings()
{
  GC_1 = -(mdl_AH * mdl_complexi); 
  GC_2 = -(mdl_ee * mdl_complexi)/3.; 
  GC_3 = (2. * mdl_ee * mdl_complexi)/3.; 
  GC_4 = -(mdl_ee * mdl_complexi); 
  GC_6 = 2. * mdl_ee__exp__2 * mdl_complexi; 
  GC_7 = -mdl_ee__exp__2/(2. * mdl_cw); 
  GC_8 = (mdl_ee__exp__2 * mdl_complexi)/(2. * mdl_cw); 
  GC_9 = mdl_ee__exp__2/(2. * mdl_cw); 
  GC_19 = -(mdl_complexi * mdl_gw__exp__2); 
  GC_20 = mdl_cw__exp__2 * mdl_complexi * mdl_gw__exp__2; 
  GC_21 = -2. * mdl_complexi * mdl_lam; 
  GC_22 = -4. * mdl_complexi * mdl_lam; 
  GC_23 = -6. * mdl_complexi * mdl_lam; 
  GC_24 = -(mdl_ee * mdl_MW); 
  GC_25 = mdl_ee * mdl_MW; 
  GC_26 = (mdl_ee__exp__2 * mdl_complexi)/(2. * mdl_sw__exp__2); 
  GC_27 = -(mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_28 = (mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_29 = mdl_ee/(2. * mdl_sw); 
  GC_31 = (mdl_CKM1x1 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_32 = (mdl_CKM1x2 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_33 = (mdl_CKM1x3 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_34 = (mdl_CKM2x1 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_36 = (mdl_CKM2x3 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_37 = (mdl_CKM3x1 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_38 = (mdl_CKM3x2 * mdl_ee * mdl_complexi)/(mdl_sw * mdl_sqrt__2); 
  GC_40 = -(mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_41 = (mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw); 
  GC_42 = -((mdl_cw * mdl_ee * mdl_complexi)/mdl_sw); 
  GC_43 = (mdl_cw * mdl_ee * mdl_complexi)/mdl_sw; 
  GC_44 = -mdl_ee__exp__2/(2. * mdl_sw); 
  GC_45 = -(mdl_ee__exp__2 * mdl_complexi)/(2. * mdl_sw); 
  GC_46 = mdl_ee__exp__2/(2. * mdl_sw); 
  GC_47 = -(mdl_ee * mdl_MW)/(2. * mdl_sw); 
  GC_48 = -(mdl_ee * mdl_complexi * mdl_MW)/(2. * mdl_sw); 
  GC_49 = (mdl_ee * mdl_MW)/(2. * mdl_sw); 
  GC_50 = -(mdl_ee * mdl_MZ)/(2. * mdl_sw); 
  GC_51 = (mdl_ee * mdl_MZ)/(2. * mdl_sw); 
  GC_52 = -(mdl_ee * mdl_complexi * mdl_MZ)/(2. * mdl_cw * mdl_sw); 
  GC_53 = -(mdl_ee * mdl_complexi * mdl_sw)/(6. * mdl_cw); 
  GC_54 = (mdl_ee * mdl_complexi * mdl_sw)/(2. * mdl_cw); 
  GC_55 = mdl_complexi * mdl_gw * mdl_sw; 
  GC_56 = -2. * mdl_cw * mdl_complexi * mdl_gw__exp__2 * mdl_sw; 
  GC_57 = mdl_complexi * mdl_gw__exp__2 * mdl_sw__exp__2; 
  GC_58 = -(mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw) + (mdl_ee *
      mdl_complexi * mdl_sw)/(2. * mdl_cw);
  GC_59 = (mdl_cw * mdl_ee * mdl_complexi)/(2. * mdl_sw) + (mdl_ee *
      mdl_complexi * mdl_sw)/(2. * mdl_cw);
  GC_60 = (mdl_cw * mdl_ee)/(2. * mdl_sw) + (mdl_ee * mdl_sw)/(2. * mdl_cw); 
  GC_61 = (mdl_cw * mdl_ee__exp__2 * mdl_complexi)/mdl_sw - (mdl_ee__exp__2 *
      mdl_complexi * mdl_sw)/mdl_cw;
  GC_62 = (mdl_cw * mdl_ee * mdl_MW)/(2. * mdl_sw) - (mdl_ee * mdl_MW *
      mdl_sw)/(2. * mdl_cw);
  GC_63 = -(mdl_cw * mdl_ee * mdl_MW)/(2. * mdl_sw) + (mdl_ee * mdl_MW *
      mdl_sw)/(2. * mdl_cw);
  GC_64 = -(mdl_ee__exp__2 * mdl_complexi) + (mdl_cw__exp__2 * mdl_ee__exp__2 *
      mdl_complexi)/(2. * mdl_sw__exp__2) + (mdl_ee__exp__2 * mdl_complexi *
      mdl_sw__exp__2)/(2. * mdl_cw__exp__2);
  GC_65 = mdl_ee__exp__2 * mdl_complexi + (mdl_cw__exp__2 * mdl_ee__exp__2 *
      mdl_complexi)/(2. * mdl_sw__exp__2) + (mdl_ee__exp__2 * mdl_complexi *
      mdl_sw__exp__2)/(2. * mdl_cw__exp__2);
  GC_66 = -(mdl_ee__exp__2 * mdl_v)/(2. * mdl_cw); 
  GC_67 = (mdl_ee__exp__2 * mdl_v)/(2. * mdl_cw); 
  GC_68 = -2. * mdl_complexi * mdl_lam * mdl_v; 
  GC_69 = -6. * mdl_complexi * mdl_lam * mdl_v; 
  GC_70 = (mdl_ee__exp__2 * mdl_complexi * mdl_v)/(2. * mdl_sw__exp__2); 
  GC_73 = mdl_ee__exp__2 * mdl_complexi * mdl_v + (mdl_cw__exp__2 *
      mdl_ee__exp__2 * mdl_complexi * mdl_v)/(2. * mdl_sw__exp__2) +
      (mdl_ee__exp__2 * mdl_complexi * mdl_sw__exp__2 * mdl_v)/(2. *
      mdl_cw__exp__2);
  GC_74 = -((mdl_complexi * mdl_yb)/mdl_sqrt__2); 
  GC_75 = mdl_yb/mdl_sqrt__2; 
  GC_76 = -(mdl_CKM1x3 * mdl_yb); 
  GC_77 = -(mdl_CKM2x3 * mdl_yb); 
  GC_78 = -(mdl_CKM3x3 * mdl_yb); 
  GC_79 = -(mdl_yt/mdl_sqrt__2); 
  GC_80 = -((mdl_complexi * mdl_yt)/mdl_sqrt__2); 
  GC_81 = mdl_CKM3x1 * mdl_yt; 
  GC_82 = mdl_CKM3x2 * mdl_yt; 
  GC_83 = mdl_CKM3x3 * mdl_yt; 
  GC_84 = -mdl_ytau; 
  GC_85 = mdl_ytau; 
  GC_86 = -((mdl_complexi * mdl_ytau)/mdl_sqrt__2); 
  GC_87 = mdl_ytau/mdl_sqrt__2; 
  GC_90 = (mdl_ee * mdl_complexi * mdl_conjg__CKM1x3)/(mdl_sw * mdl_sqrt__2); 
  GC_91 = mdl_yb * mdl_conjg__CKM1x3; 
  GC_95 = mdl_yb * mdl_conjg__CKM2x3; 
  GC_96 = (mdl_ee * mdl_complexi * mdl_conjg__CKM3x1)/(mdl_sw * mdl_sqrt__2); 
  GC_97 = -(mdl_yt * mdl_conjg__CKM3x1); 
  GC_99 = -(mdl_yt * mdl_conjg__CKM3x2); 
  GC_100 = (mdl_ee * mdl_complexi * mdl_conjg__CKM3x3)/(mdl_sw * mdl_sqrt__2); 
  GC_101 = mdl_yb * mdl_conjg__CKM3x3; 
  GC_102 = -(mdl_yt * mdl_conjg__CKM3x3); 
}
void Parameters_heft_ckm::setDependentParameters()
{
  mdl_sqrt__aS = sqrt(aS); 
  G = 2. * mdl_sqrt__aS * sqrt(M_PI); 
  mdl_G__exp__2 = ((G) * (G)); 
  mdl_GH = -(mdl_G__exp__2 * (1. + (13. * mdl_MH__exp__6)/(16800. *
      mdl_MT__exp__6) + mdl_MH__exp__4/(168. * mdl_MT__exp__4) + (7. *
      mdl_MH__exp__2)/(120. * mdl_MT__exp__2)))/(12. * ((M_PI) * (M_PI)) *
      mdl_v);
  mdl_Gphi = -(mdl_G__exp__2 * (1. + mdl_MH__exp__6/(560. * mdl_MT__exp__6) +
      mdl_MH__exp__4/(90. * mdl_MT__exp__4) + mdl_MH__exp__2/(12. *
      mdl_MT__exp__2)))/(8. * ((M_PI) * (M_PI)) * mdl_v);
}
void Parameters_heft_ckm::setDependentCouplings()
{
  GC_17 = -(G * mdl_Gphi); 
  GC_16 = (mdl_complexi * mdl_Gphi)/8.; 
  GC_15 = mdl_complexi * mdl_G__exp__2 * mdl_GH; 
  GC_14 = -(G * mdl_GH); 
  GC_13 = -(mdl_complexi * mdl_GH); 
  GC_12 = mdl_complexi * mdl_G__exp__2; 
  GC_11 = mdl_complexi * G; 
  GC_10 = -G; 
}

// Routines for printing out parameters
void Parameters_heft_ckm::printIndependentParameters()
{
  cout <<  "heft_ckm model parameters independent of event kinematics:" <<
      endl;
  cout << setprecision(20) <<  "mdl_WH1 " <<  "= " << setprecision(10) <<
      mdl_WH1 << endl;
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
  cout << setprecision(20) <<  "mdl_MP " <<  "= " << setprecision(10) << mdl_MP
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
  cout << setprecision(20) <<  "mdl_conjg__CKM3x3 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM3x3 << endl;
  cout << setprecision(20) <<  "mdl_CKM3x3 " <<  "= " << setprecision(10) <<
      mdl_CKM3x3 << endl;
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
  cout << setprecision(20) <<  "mdl_MH__exp__4 " <<  "= " << setprecision(10)
      << mdl_MH__exp__4 << endl;
  cout << setprecision(20) <<  "mdl_MT__exp__4 " <<  "= " << setprecision(10)
      << mdl_MT__exp__4 << endl;
  cout << setprecision(20) <<  "mdl_MH__exp__2 " <<  "= " << setprecision(10)
      << mdl_MH__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_MT__exp__2 " <<  "= " << setprecision(10)
      << mdl_MT__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_MH__exp__12 " <<  "= " << setprecision(10)
      << mdl_MH__exp__12 << endl;
  cout << setprecision(20) <<  "mdl_MH__exp__10 " <<  "= " << setprecision(10)
      << mdl_MH__exp__10 << endl;
  cout << setprecision(20) <<  "mdl_MH__exp__8 " <<  "= " << setprecision(10)
      << mdl_MH__exp__8 << endl;
  cout << setprecision(20) <<  "mdl_MH__exp__6 " <<  "= " << setprecision(10)
      << mdl_MH__exp__6 << endl;
  cout << setprecision(20) <<  "mdl_MT__exp__6 " <<  "= " << setprecision(10)
      << mdl_MT__exp__6 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM1x1 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM1x1 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM1x2 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM1x2 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM1x3 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM1x3 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM2x1 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM2x1 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM2x2 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM2x2 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM2x3 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM2x3 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM3x1 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM3x1 << endl;
  cout << setprecision(20) <<  "mdl_conjg__CKM3x2 " <<  "= " <<
      setprecision(10) << mdl_conjg__CKM3x2 << endl;
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
  cout << setprecision(20) <<  "mdl_v " <<  "= " << setprecision(10) << mdl_v
      << endl;
  cout << setprecision(20) <<  "mdl_ee__exp__2 " <<  "= " << setprecision(10)
      << mdl_ee__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_MW__exp__12 " <<  "= " << setprecision(10)
      << mdl_MW__exp__12 << endl;
  cout << setprecision(20) <<  "mdl_MW__exp__10 " <<  "= " << setprecision(10)
      << mdl_MW__exp__10 << endl;
  cout << setprecision(20) <<  "mdl_MW__exp__8 " <<  "= " << setprecision(10)
      << mdl_MW__exp__8 << endl;
  cout << setprecision(20) <<  "mdl_MW__exp__6 " <<  "= " << setprecision(10)
      << mdl_MW__exp__6 << endl;
  cout << setprecision(20) <<  "mdl_MW__exp__4 " <<  "= " << setprecision(10)
      << mdl_MW__exp__4 << endl;
  cout << setprecision(20) <<  "mdl_AH " <<  "= " << setprecision(10) << mdl_AH
      << endl;
  cout << setprecision(20) <<  "mdl_v__exp__2 " <<  "= " << setprecision(10) <<
      mdl_v__exp__2 << endl;
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
  cout << setprecision(20) <<  "mdl_gw__exp__2 " <<  "= " << setprecision(10)
      << mdl_gw__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_cw__exp__2 " <<  "= " << setprecision(10)
      << mdl_cw__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_sw__exp__2 " <<  "= " << setprecision(10)
      << mdl_sw__exp__2 << endl;
}
void Parameters_heft_ckm::printIndependentCouplings()
{
  cout <<  "heft_ckm model couplings independent of event kinematics:" << endl; 
  cout << setprecision(20) <<  "GC_1 " <<  "= " << setprecision(10) << GC_1 <<
      endl;
  cout << setprecision(20) <<  "GC_2 " <<  "= " << setprecision(10) << GC_2 <<
      endl;
  cout << setprecision(20) <<  "GC_3 " <<  "= " << setprecision(10) << GC_3 <<
      endl;
  cout << setprecision(20) <<  "GC_4 " <<  "= " << setprecision(10) << GC_4 <<
      endl;
  cout << setprecision(20) <<  "GC_6 " <<  "= " << setprecision(10) << GC_6 <<
      endl;
  cout << setprecision(20) <<  "GC_7 " <<  "= " << setprecision(10) << GC_7 <<
      endl;
  cout << setprecision(20) <<  "GC_8 " <<  "= " << setprecision(10) << GC_8 <<
      endl;
  cout << setprecision(20) <<  "GC_9 " <<  "= " << setprecision(10) << GC_9 <<
      endl;
  cout << setprecision(20) <<  "GC_19 " <<  "= " << setprecision(10) << GC_19
      << endl;
  cout << setprecision(20) <<  "GC_20 " <<  "= " << setprecision(10) << GC_20
      << endl;
  cout << setprecision(20) <<  "GC_21 " <<  "= " << setprecision(10) << GC_21
      << endl;
  cout << setprecision(20) <<  "GC_22 " <<  "= " << setprecision(10) << GC_22
      << endl;
  cout << setprecision(20) <<  "GC_23 " <<  "= " << setprecision(10) << GC_23
      << endl;
  cout << setprecision(20) <<  "GC_24 " <<  "= " << setprecision(10) << GC_24
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
  cout << setprecision(20) <<  "GC_31 " <<  "= " << setprecision(10) << GC_31
      << endl;
  cout << setprecision(20) <<  "GC_32 " <<  "= " << setprecision(10) << GC_32
      << endl;
  cout << setprecision(20) <<  "GC_33 " <<  "= " << setprecision(10) << GC_33
      << endl;
  cout << setprecision(20) <<  "GC_34 " <<  "= " << setprecision(10) << GC_34
      << endl;
  cout << setprecision(20) <<  "GC_36 " <<  "= " << setprecision(10) << GC_36
      << endl;
  cout << setprecision(20) <<  "GC_37 " <<  "= " << setprecision(10) << GC_37
      << endl;
  cout << setprecision(20) <<  "GC_38 " <<  "= " << setprecision(10) << GC_38
      << endl;
  cout << setprecision(20) <<  "GC_40 " <<  "= " << setprecision(10) << GC_40
      << endl;
  cout << setprecision(20) <<  "GC_41 " <<  "= " << setprecision(10) << GC_41
      << endl;
  cout << setprecision(20) <<  "GC_42 " <<  "= " << setprecision(10) << GC_42
      << endl;
  cout << setprecision(20) <<  "GC_43 " <<  "= " << setprecision(10) << GC_43
      << endl;
  cout << setprecision(20) <<  "GC_44 " <<  "= " << setprecision(10) << GC_44
      << endl;
  cout << setprecision(20) <<  "GC_45 " <<  "= " << setprecision(10) << GC_45
      << endl;
  cout << setprecision(20) <<  "GC_46 " <<  "= " << setprecision(10) << GC_46
      << endl;
  cout << setprecision(20) <<  "GC_47 " <<  "= " << setprecision(10) << GC_47
      << endl;
  cout << setprecision(20) <<  "GC_48 " <<  "= " << setprecision(10) << GC_48
      << endl;
  cout << setprecision(20) <<  "GC_49 " <<  "= " << setprecision(10) << GC_49
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
  cout << setprecision(20) <<  "GC_84 " <<  "= " << setprecision(10) << GC_84
      << endl;
  cout << setprecision(20) <<  "GC_85 " <<  "= " << setprecision(10) << GC_85
      << endl;
  cout << setprecision(20) <<  "GC_86 " <<  "= " << setprecision(10) << GC_86
      << endl;
  cout << setprecision(20) <<  "GC_87 " <<  "= " << setprecision(10) << GC_87
      << endl;
  cout << setprecision(20) <<  "GC_90 " <<  "= " << setprecision(10) << GC_90
      << endl;
  cout << setprecision(20) <<  "GC_91 " <<  "= " << setprecision(10) << GC_91
      << endl;
  cout << setprecision(20) <<  "GC_95 " <<  "= " << setprecision(10) << GC_95
      << endl;
  cout << setprecision(20) <<  "GC_96 " <<  "= " << setprecision(10) << GC_96
      << endl;
  cout << setprecision(20) <<  "GC_97 " <<  "= " << setprecision(10) << GC_97
      << endl;
  cout << setprecision(20) <<  "GC_99 " <<  "= " << setprecision(10) << GC_99
      << endl;
  cout << setprecision(20) <<  "GC_100 " <<  "= " << setprecision(10) << GC_100
      << endl;
  cout << setprecision(20) <<  "GC_101 " <<  "= " << setprecision(10) << GC_101
      << endl;
  cout << setprecision(20) <<  "GC_102 " <<  "= " << setprecision(10) << GC_102
      << endl;
}
void Parameters_heft_ckm::printDependentParameters()
{
  cout <<  "heft_ckm model parameters dependent on event kinematics:" << endl; 
  cout << setprecision(20) <<  "mdl_sqrt__aS " <<  "= " << setprecision(10) <<
      mdl_sqrt__aS << endl;
  cout << setprecision(20) <<  "G " <<  "= " << setprecision(10) << G << endl; 
  cout << setprecision(20) <<  "mdl_G__exp__2 " <<  "= " << setprecision(10) <<
      mdl_G__exp__2 << endl;
  cout << setprecision(20) <<  "mdl_GH " <<  "= " << setprecision(10) << mdl_GH
      << endl;
  cout << setprecision(20) <<  "mdl_Gphi " <<  "= " << setprecision(10) <<
      mdl_Gphi << endl;
}
void Parameters_heft_ckm::printDependentCouplings()
{
  cout <<  "heft_ckm model couplings dependent on event kinematics:" << endl; 
  cout << setprecision(20) <<  "GC_17 " <<  "= " << setprecision(10) << GC_17
      << endl;
  cout << setprecision(20) <<  "GC_16 " <<  "= " << setprecision(10) << GC_16
      << endl;
  cout << setprecision(20) <<  "GC_15 " <<  "= " << setprecision(10) << GC_15
      << endl;
  cout << setprecision(20) <<  "GC_14 " <<  "= " << setprecision(10) << GC_14
      << endl;
  cout << setprecision(20) <<  "GC_13 " <<  "= " << setprecision(10) << GC_13
      << endl;
  cout << setprecision(20) <<  "GC_12 " <<  "= " << setprecision(10) << GC_12
      << endl;
  cout << setprecision(20) <<  "GC_11 " <<  "= " << setprecision(10) << GC_11
      << endl;
  cout << setprecision(20) <<  "GC_10 " <<  "= " << setprecision(10) << GC_10
      << endl;
}

// Usage from inside Pythia8
#ifdef PYTHIA8

void Parameters_heft_ckm::setIndependentParameters(Pythia8::ParticleData * &
    pd,
Pythia8::Couplings * & csm, Pythia8::SusyLesHouches * & slhaPtr)
{
  if (false)
    cout << pd << csm << slhaPtr; 
  mdl_WH1 = pd->mWidth(9000006); 
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
  mdl_MP = pd->m0(9000006); 
  mdl_MH = pd->m0(25); 
  mdl_MZ = pd->m0(23); 
  mdl_MTA = pd->m0(15); 
  mdl_MT = pd->m0(6); 
  mdl_MB = pd->m0(5); 
  mdl_conjg__CKM3x3 = 1.; 
  mdl_CKM3x3 = 1.; 
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
  mdl_MH__exp__4 = ((mdl_MH) * (mdl_MH) * (mdl_MH) * (mdl_MH)); 
  mdl_MT__exp__4 = ((mdl_MT) * (mdl_MT) * (mdl_MT) * (mdl_MT)); 
  mdl_MH__exp__2 = ((mdl_MH) * (mdl_MH)); 
  mdl_MT__exp__2 = ((mdl_MT) * (mdl_MT)); 
  mdl_MH__exp__12 = pow(mdl_MH, 12.); 
  mdl_MH__exp__10 = pow(mdl_MH, 10.); 
  mdl_MH__exp__8 = pow(mdl_MH, 8.); 
  mdl_MH__exp__6 = pow(mdl_MH, 6.); 
  mdl_MT__exp__6 = pow(mdl_MT, 6.); 
  mdl_conjg__CKM1x1 = conj(mdl_CKM1x1); 
  mdl_conjg__CKM1x2 = conj(mdl_CKM1x2); 
  mdl_conjg__CKM1x3 = conj(mdl_CKM1x3); 
  mdl_conjg__CKM2x1 = conj(mdl_CKM2x1); 
  mdl_conjg__CKM2x2 = conj(mdl_CKM2x2); 
  mdl_conjg__CKM2x3 = conj(mdl_CKM2x3); 
  mdl_conjg__CKM3x1 = conj(mdl_CKM3x1); 
  mdl_conjg__CKM3x2 = conj(mdl_CKM3x2); 
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
  mdl_v = (2. * mdl_MW * mdl_sw)/mdl_ee; 
  mdl_ee__exp__2 = ((mdl_ee) * (mdl_ee)); 
  mdl_MW__exp__12 = pow(mdl_MW, 12.); 
  mdl_MW__exp__10 = pow(mdl_MW, 10.); 
  mdl_MW__exp__8 = pow(mdl_MW, 8.); 
  mdl_MW__exp__6 = pow(mdl_MW, 6.); 
  mdl_MW__exp__4 = ((mdl_MW) * (mdl_MW) * (mdl_MW) * (mdl_MW)); 
  mdl_AH = (47. * mdl_ee__exp__2 * (1. - (2. * mdl_MH__exp__4)/(987. *
      mdl_MT__exp__4) - (14. * mdl_MH__exp__2)/(705. * mdl_MT__exp__2) + (213.
      * mdl_MH__exp__12)/(2.634632e7 * mdl_MW__exp__12) + (5. *
      mdl_MH__exp__10)/(119756. * mdl_MW__exp__10) + (41. *
      mdl_MH__exp__8)/(180950. * mdl_MW__exp__8) + (87. *
      mdl_MH__exp__6)/(65800. * mdl_MW__exp__6) + (57. * mdl_MH__exp__4)/(6580.
      * mdl_MW__exp__4) + (33. * mdl_MH__exp__2)/(470. * mdl_MW__exp__2)))/(72.
      * ((M_PI) * (M_PI)) * mdl_v);
  mdl_v__exp__2 = ((mdl_v) * (mdl_v)); 
  mdl_lam = mdl_MH__exp__2/(2. * mdl_v__exp__2); 
  mdl_yb = (mdl_ymb * mdl_sqrt__2)/mdl_v; 
  mdl_yt = (mdl_ymt * mdl_sqrt__2)/mdl_v; 
  mdl_ytau = (mdl_ymtau * mdl_sqrt__2)/mdl_v; 
  mdl_muH = sqrt(mdl_lam * mdl_v__exp__2); 
  mdl_gw__exp__2 = ((mdl_gw) * (mdl_gw)); 
  mdl_cw__exp__2 = ((mdl_cw) * (mdl_cw)); 
  mdl_sw__exp__2 = ((mdl_sw) * (mdl_sw)); 
}
void Parameters_heft_ckm::setDependentParameters(Pythia8::ParticleData * & pd, 
Pythia8::Couplings * & csm, Pythia8::SusyLesHouches * & slhaPtr, double alpS)
{
  if (false)
    cout << pd << csm << slhaPtr << alpS; 
  aS = alpS; 
  mdl_sqrt__aS = sqrt(aS); 
  G = 2. * mdl_sqrt__aS * sqrt(M_PI); 
  mdl_G__exp__2 = ((G) * (G)); 
  mdl_GH = -(mdl_G__exp__2 * (1. + (13. * mdl_MH__exp__6)/(16800. *
      mdl_MT__exp__6) + mdl_MH__exp__4/(168. * mdl_MT__exp__4) + (7. *
      mdl_MH__exp__2)/(120. * mdl_MT__exp__2)))/(12. * ((M_PI) * (M_PI)) *
      mdl_v);
  mdl_Gphi = -(mdl_G__exp__2 * (1. + mdl_MH__exp__6/(560. * mdl_MT__exp__6) +
      mdl_MH__exp__4/(90. * mdl_MT__exp__4) + mdl_MH__exp__2/(12. *
      mdl_MT__exp__2)))/(8. * ((M_PI) * (M_PI)) * mdl_v);
}

#endif

