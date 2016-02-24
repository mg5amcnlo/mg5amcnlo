//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. %(version)s, %(date)s
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef Pythia8_Sigma_sm_gd_ddxd_H
#define Pythia8_Sigma_sm_gd_ddxd_H

#include <complex> 

#include "Pythia8/SigmaProcess.h"
#include "Parameters_sm.h"

using namespace std; 

namespace Pythia8 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: g d > z d WEIGHTED<=3
// *   Decay: z > d d~ WEIGHTED<=2
// Process: g s > z s WEIGHTED<=3
// *   Decay: z > d d~ WEIGHTED<=2
// Process: g d > z d WEIGHTED<=3
// *   Decay: z > u u~ WEIGHTED<=2
// Process: g s > z s WEIGHTED<=3
// *   Decay: z > u u~ WEIGHTED<=2
// Process: g d > z d WEIGHTED<=3
// *   Decay: z > s s~ WEIGHTED<=2
// Process: g s > z s WEIGHTED<=3
// *   Decay: z > s s~ WEIGHTED<=2
// Process: g u > z u WEIGHTED<=3
// *   Decay: z > d d~ WEIGHTED<=2
// Process: g u > z u WEIGHTED<=3
// *   Decay: z > s s~ WEIGHTED<=2
// Process: g u > z u WEIGHTED<=3
// *   Decay: z > u u~ WEIGHTED<=2
// Process: g d~ > z d~ WEIGHTED<=3
// *   Decay: z > d d~ WEIGHTED<=2
// Process: g s~ > z s~ WEIGHTED<=3
// *   Decay: z > d d~ WEIGHTED<=2
// Process: g d~ > z d~ WEIGHTED<=3
// *   Decay: z > u u~ WEIGHTED<=2
// Process: g s~ > z s~ WEIGHTED<=3
// *   Decay: z > u u~ WEIGHTED<=2
// Process: g d~ > z d~ WEIGHTED<=3
// *   Decay: z > s s~ WEIGHTED<=2
// Process: g s~ > z s~ WEIGHTED<=3
// *   Decay: z > s s~ WEIGHTED<=2
// Process: g u~ > z u~ WEIGHTED<=3
// *   Decay: z > d d~ WEIGHTED<=2
// Process: g u~ > z u~ WEIGHTED<=3
// *   Decay: z > s s~ WEIGHTED<=2
// Process: g u~ > z u~ WEIGHTED<=3
// *   Decay: z > u u~ WEIGHTED<=2
//--------------------------------------------------------------------------

class Sigma_sm_gd_ddxd : public Sigma3Process 
{
  public:

    // Constructor.
    Sigma_sm_gd_ddxd() {}

    // Initialize process.
    virtual void initProc(); 

    // Calculate flavour-independent parts of cross section.
    virtual void sigmaKin(); 

    // Evaluate sigmaHat(sHat).
    virtual double sigmaHat(); 

    // Select flavour, colour and anticolour.
    virtual void setIdColAcol(); 

    // Evaluate weight for decay angles.
    virtual double weightDecay(Event& process, int iResBeg, int iResEnd); 

    // Info on the subprocess.
    virtual string name() const {return "g d > d d~ d (sm)";}

    virtual int code() const {return 10000;}

    virtual string inFlux() const {return "qg";}

    virtual int resonanceA() const {return 23;}
    // Tell Pythia that sigmaHat returns the ME^2
    virtual bool convertM2() const {return true;}

  private:

    // Private functions to calculate the matrix element for all subprocesses
    // Calculate wavefunctions
    void calculate_wavefunctions(const int perm[], const int hel[]); 
    static const int nwavefuncs = 13; 
    std::complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 18; 
    std::complex<double> amp[namplitudes]; 
    double matrix_gd_zd_z_ddx(); 
    double matrix_gd_zd_z_uux(); 
    double matrix_gd_zd_z_ssx(); 
    double matrix_gu_zu_z_ddx(); 
    double matrix_gu_zu_z_uux(); 
    double matrix_gdx_zdx_z_ddx(); 
    double matrix_gdx_zdx_z_uux(); 
    double matrix_gdx_zdx_z_ssx(); 
    double matrix_gux_zux_z_ddx(); 
    double matrix_gux_zux_z_uux(); 

    // Constants for array limits
    static const int nexternal = 5; 
    static const int nprocesses = 20; 

    // Store the matrix element value from sigmaKin
    double matrix_element[nprocesses]; 

    // Color flows, used when selecting color
    double * jamp2[nprocesses]; 

    // Pointer to the model parameters
    Parameters_sm * pars; 

}; 

}  // end namespace Pythia8

#endif  // Pythia8_Sigma_sm_gd_ddxd_H

