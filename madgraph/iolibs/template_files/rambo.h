#include <vector>

using namespace std;

class Random
{
  public:
    double ranmar();
    void rmarin(int ij, int kl);
    
  private:
    double ranu[98];
    double ranc, rancd, rancm;
    int iranmr, jranmr;
};


double rn(int idummy);

vector<double*> get_momenta(int ninitial, double energy, 
				    vector<double> masses, double& wgt);

vector<double*> rambo(double et, vector<double>& xm, double& wt);
