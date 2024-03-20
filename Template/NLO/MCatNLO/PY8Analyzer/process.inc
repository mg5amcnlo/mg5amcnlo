      integer n_Born,n_jet_Born,pdg(10)
      character*2 str_pdg(10)
      parameter(n_Born=1,n_jet_Born=0)
c n_Born is the total number of particles, including jets,
c at Born level, implying n_Born >= n_jet_Born.
c For instance in Higgs VBF production n_Born = 3, n_jet_Born = 2 
c For instance in W production         n_Born = 1, n_jet_Born = 0 
c Jets are assigned pdg = 0
c initialise
      pdg=-1
      str_pdg='  '
c
      pdg(1)=24
      str_pdg(1)='W+'
