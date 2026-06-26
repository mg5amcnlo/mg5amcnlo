//==========================================================================
// A class for calculating the matrix elements for
%(process_lines)s
//--------------------------------------------------------------------------

class CPPProcess
{
public:

  // Constants for array limits
  static const int ninitial = %(ninitial)d;
  static const int nexternal = %(nexternal)d;
  static const int nprocesses = %(nprocesses)d;
  static const int ncomb = %(ncomb)d;
  static const int ndiagrams = %(ndiag)d;
  static const int namplitudes = %(namp)d;
  static const int nflavors = %(nflav)d;

  // Constructor.
  CPPProcess(string param_card_name);

  // Calculate flavour-independent parts of cross section.
  double sigmaKin(int flavor_id);

  // Info on the subprocess.
  string name()    const {return "%(process_definition)s";}

  int    code()    const {return %(process_code)s;}

  const vector<double>& getMasses() const {return mME;}

  const double* getAmp2() const { return amp2; }

  Parameters_%(model_name)s& getParameters() { return pars; }

  // Get and set momenta for matrix element evaluation
  vector<double*>& getMomenta(){return p;}
  void setMomenta(vector<double*>& momenta){p = momenta;}
  void setInitial(int inid1, int inid2){id1 = inid1; id2 = inid2;}

private:

  // Private functions to calculate the matrix element for all subprocesses
  %(all_sigma_kin_definitions)s
  %(all_matrix_definitions)s

  std::complex<double> amp[namplitudes];
  double amp2[ndiagrams];

  // Color flows, used when selecting color
  double* jamp2;

  // Pointer to the model parameters
  Parameters_%(model_name)s pars;

  // vector with external particle masses
  vector<double> mME;

  // vector with momenta (to be changed each event)
  vector<double*> p;
  // Initial particle ids
  int id1, id2;

  bool goodhel[nflavors][ncomb];
  int ntry[nflavors], sum_hel[nflavors], ngood[nflavors];
  int igood[nflavors][ncomb];
  int jhel[nflavors];

  // function to compute missing symmetry factors after flavor consolidation
  int broken_sym(const int* flavor);

};
