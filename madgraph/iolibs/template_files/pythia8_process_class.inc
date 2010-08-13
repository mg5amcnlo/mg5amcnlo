//==========================================================================
// A class for calculating the matrix elements for
%(process_lines)s
//--------------------------------------------------------------------------

class %(process_class_name)s : public Sigma%(nfinal)dProcess 
{
public:

  // Constructor.
  %(process_class_name)s() {}

  // Initialize process. 
  virtual void initProc(); 

  // Calculate flavour-independent parts of cross section.
  virtual void sigmaKin();

  // Evaluate sigmaHat(sHat). 
  virtual double sigmaHat();

  // Select flavour, colour and anticolour.
  virtual void setIdColAcol();

  // Evaluate weight for decay angles.
  virtual double weightDecay( Event& process, int iResBeg, int iResEnd); 

  // Info on the subprocess.
  virtual string name()    const {return "%(process_definition)s";}

  virtual int    code()    const {return %(process_code)s;}

  virtual string inFlux()  const {return "%(inFlux)s";}
  %(id_masses)s
  %(resonances)s
  // Tell Pythia that sigmaHat returns the ME^2
  virtual bool   convertM2() const {return true;}

private:

  // Private functions to calculate the matrix element for all subprocesses
  %(all_sigma_kin_definitions)s
  %(all_matrix_definitions)s

  // Constants for array limits
  static const int nexternal = %(nexternal)d;
  static const int nprocesses = %(nprocesses)d;

  // Store the matrix element value from sigmaKin
  double matrix_element[nprocesses];

  // Color flows, used when selecting color
  double* jamp2[nprocesses];

  // Pointer to the model parameters
  Parameters_%(model_name)s* pars;

};
