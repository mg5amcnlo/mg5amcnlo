//==========================================================================
// A class for calculating the matrix elements for
%(process_lines)s
//--------------------------------------------------------------------------

class CPPProcess
{
public:

  // Constructor.
  CPPProcess() {}

  // Initialize process. 
  virtual void initProc(string param_card_name); 

  // Calculate flavour-independent parts of cross section.
  virtual void sigmaKin();

  // Evaluate sigmaHat(sHat). 
  virtual double sigmaHat();

  // Info on the subprocess.
  virtual string name()    const {return "%(process_definition)s";}

  virtual int    code()    const {return %(process_code)s;}

  const vector<double>& getMasses() const {return mME;}

  // Get and set momenta for matrix element evaluation
  vector<double*> getMomenta(){return p;}
  void setMomenta(vector<double*>& momenta){p = momenta;}
  void setInitial(int inid1, int inid2){id1 = inid1; id2 = inid2;}

  // Get matrix element vector
  const double* getMatrixElements() const {return matrix_element;}

  // Constants for array limits
  static const int ninitial = %(ninitial)d;
  static const int nexternal = %(nexternal)d;
  static const int nprocesses = %(nprocesses)d;

private:

  // Private functions to calculate the matrix element for all subprocesses
  %(all_sigma_kin_definitions)s
  %(all_matrix_definitions)s

  // Store the matrix element value from sigmaKin
  double matrix_element[nprocesses];

  // Color flows, used when selecting color
  double* jamp2[nprocesses];

  // Pointer to the model parameters
  Parameters_%(model_name)s* pars;

  // vector with external particle masses
  vector<double> mME;

  // vector with momenta (to be changed each event)
  vector<double*> p;
  // Initial particle ids
  int id1, id2;

};
