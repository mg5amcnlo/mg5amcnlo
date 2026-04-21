{
  const char* file="pythia.root";
  TFile *f = new TFile(file,"RECREATE");
  TTree *events = new TTree("events","Events");
  Long64_t nlines1 = events->ReadFile("events.tree","Npart:DJR1:DJR2:DJR3:DJR4:PTISR:PTFSR:PT2MX:NCJET:IFILE");
  cout << "Found "<< nlines1 << " events"<<endl;
  TTree *xsecs = new TTree("xsecs","Xsecs");
  Long64_t nlines2 = xsecs->ReadFile("xsecs.tree","Xsecfact");
  cout << "Found "<< nlines2 << " files"<<endl;
  f->Write();

  gROOT->ProcessLine(".x ../bin/internal/plot_tree.C(\"DJR1\")");
  gROOT->ProcessLine(".x ../bin/internal/plot_tree.C(\"DJR2\")");
  gROOT->ProcessLine(".x ../bin/internal/plot_tree.C(\"DJR3\")");
  gROOT->ProcessLine(".x ../bin/internal/plot_tree.C(\"DJR4\")");
}
