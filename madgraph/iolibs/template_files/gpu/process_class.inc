//==========================================================================
// A class for calculating the matrix elements for
%(process_lines)s
//--------------------------------------------------------------------------

class CPPProcess
{
public:

  CPPProcess(int numiterations, int gpublocks, int gputhreads,
             bool verbose = false, bool debug = false);

  ~CPPProcess();

 // Initialize process.
  virtual void initProc(std::string param_card_name);


  virtual int code() const { return 1; }

  const std::vector<fptype> &getMasses() const;

  void setInitial(int inid1, int inid2) {
    id1 = inid1;
    id2 = inid2;
  }

  int getDim() const { return dim; }

  int getNIOParticles() const { return nexternal; }


  // Constants for array limits
  static const int ninitial = mgOnGpu::npari;
  static const int nexternal = mgOnGpu::npar;
  //static const int nprocesses = %(nprocesses)d;

private:
  int m_numiterations;
  // gpu variables
  int gpu_nblocks;
  int gpu_nthreads;
  int dim; // gpu_nblocks * gpu_nthreads;

  // print verbose info
  bool m_verbose;

  // print debug info
  bool m_debug;

  static const int nwavefuncs = %(nwavefuncs)d;
  static const int namplitudes = %(namp)d;
  static const int ncomb = %(ncomb)d;
  static const int wrows = %(sizew)d;
  //static const int nioparticles = %(nexternal)d;

  cxtype** amp;


  // Pointer to the model parameters
  Parameters_%(model)s *pars;

  // vector with external particle masses
  std::vector<fptype> mME;

  // Initial particle ids
  int id1, id2;

};
