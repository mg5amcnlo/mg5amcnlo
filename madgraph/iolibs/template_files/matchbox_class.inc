# 1 "/Users/omatt/Downloads/cpp_process_class.inc"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/Users/omatt/Downloads/cpp_process_class.inc"
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

  // Get and set momenta for matrix element evaluation
  void setMomenta(vector<double*>& momenta){p = momenta;}

  // Get amplitude for one colorflow
  std::complex<double> get_jamp(int T);
  
  // Info on number of colorflows
  int NCol();

  // Info on color  
  int colorstring(int i,int j); 

  // Constants for array limits
  static const int ninitial = %(ninitial)d;
  static const int nexternal = %(nexternal)d;
  static const int nprocesses = %(nprocesses)d;
  
  // for Matchbox public functions to calculate the matrix element for all subprocesses
  %(all_sigma_kin_definitions)s

   private:

  // Pointer to the model parameters
  Parameters_%(model_name)s* pars;

  // vector with external particle masses
  vector<double> mME;

  // vector with momenta (to be changed each event)
  vector<double*> p;

  // Color flows, used when selecting color
  std::complex<double> jamp[%(ncolor)d];

};
