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

  virtual int    id3Mass() const {return %(id3Mass)d;}

  virtual int    id4Mass() const {return %(id4Mass)d;}

  // Tell Pythia that sigmaHat returns the ME^2
  virtual bool   convertME() const {return true;}

private:
  // Store flavour-specific process information and standard prefactor.
%(process_variables)s
};
