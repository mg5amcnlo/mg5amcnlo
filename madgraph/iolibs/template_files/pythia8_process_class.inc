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
  // Tell Pythia that sigmaHat returns the ME^2
  virtual bool   convertME() const {return true;}

private:

  // Private function to calculate the matrix element for given helicities
  double matrix(int helicities[]);

  // Constants for array limits
  const int nexternal = %(nexternal)d;
  const int ncolor = %(ncolor)d;

  // Store the matrix element value from sigmaKin
  double matrix_element;

  // Color flows, used when selecting color
  double jamp2[ncolor];

  // Other process-specific information, e.g. couplings
%(process_variables)s
};
