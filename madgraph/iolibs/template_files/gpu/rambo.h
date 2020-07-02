#include <vector>

class Random {
public:
  double ranmar();
  void rmarin(int ij, int kl);

private:
  double ranu[98];
  double ranc, rancd, rancm;
  int iranmr, jranmr;
};

double rn(int idummy);

std::vector<std::vector<double *>> get_momenta(int ninitial, double energy,
                                               std::vector<double> masses,
                                               double &wgt, int dim);

std::vector<double *> rambo(double et, std::vector<double> &xm, double &wt);
