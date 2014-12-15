*     -*-fortran-*-

      integer max_contr
      parameter (max_contr=100)
      integer nFKS(max_contr),itype(max_contr),QCDpower(max_contr)
     &     ,pdg(nexternal,max_contr),icontr
      double precision momenta(0:3,nexternal,max_contr),wgt(3,max_contr)
     &     ,bjx(2,max_contr),scales2(3,max_contr),g_strong(max_contr)
     &     ,weight(max_contr),y_bst(max_contr)
      common /c_weights/momenta,wgt,weight,bjx,scales2,g_strong,y_bst
     &     ,nFKS,itype,QCDpower,pdg,icontr
