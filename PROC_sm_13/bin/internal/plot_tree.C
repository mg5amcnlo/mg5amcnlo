#include <string>

bool plot_tree(const char* quantity, const char* plotdim = 0, bool log = true)
{
  const char* plotdim_default = "(100,0.,3.)";
  if(plotdim == 0){
    plotdim = plotdim_default;
  }
  char tmp1[250];
  char tmp2[300];
  char tmp3[100];
  sprintf(tmp2,"pythia.root");

  TFile* file;
  file=new TFile(tmp2);
  if(!file->IsOpen()) {
    cout << "No file "<<tmp2<<endl;
    return false;
  }
  file->cd();

  gROOT->SetStyle("Plain");
  const int maxjets=1+4;
  TH1F* hists[maxjets];
  sprintf(tmp3,"%s",quantity);
  TCanvas* c1=new TCanvas(tmp3,tmp3);
  c1->SetLogy();
  TLegend* leg=new TLegend(0.99,0.7,0.7,0.95);
  sprintf(tmp3,"l%s",quantity);
  leg->SetName(tmp3);
  int color[5]={2,4,3,5,6};
  int style[5]={2,3,4,3,4};

  //  bool ptw=false;
  TLeaf *leaf_Xsec = xsecs->FindLeaf("Xsecfact");
  Float_t Xsecfact;
  leaf_Xsec->SetAddress(&Xsecfact);
  xsecs->GetEntry(0);
  if (events->GetEntries()>0) {
  for(int i=0;i<maxjets;i++){
    events->SetLineWidth(2);
    events->SetLineColor(i+2);
    events->SetLineStyle(i+2);
    
    if(log) 
      sprintf(tmp1,"log10(%s)>>%s%i%s",quantity,quantity,i,plotdim);
    else 
      sprintf(tmp1,"%s>>%s%i%s",quantity,quantity,i,plotdim);
    sprintf(tmp2,"%e*(Npart==%i)",Xsecfact,i);

    cout << "events->Draw("<<tmp1<<","<<tmp2<<");"<<endl;
    events->Draw(tmp1,tmp2);

    sprintf(tmp3,"%s%i",quantity,i);
    hists[i]=(TH1F*)gROOT->FindObject(tmp3);
    if(!hists[i]){
      cout << "Failed to get object "<<tmp3<<endl;
      return false;
    }
  }
  }

  TH1F *hsum = (TH1F*)hists[0]->Clone();
  sprintf(tmp3,"%ssum",quantity);
  hsum->SetName(tmp3);
  for(int i=1;i<maxjets;i++)
    hsum->Add(hists[i]);
  cout << "Integral of "<<quantity<<": "<<hsum->Integral()<<endl;
  hsum->SetLineWidth(2);
  hsum->SetLineColor(1);
  hsum->SetLineStyle(1);
  //  hsum->SetMinimum(hsum->GetMaximum()*1e-3);
  hsum->SetStats(kFALSE);
  sprintf(tmp3,"%s",quantity);
  hsum->SetTitle(tmp3);
  sprintf(tmp3,"log10(%s)",quantity);
  hsum->GetXaxis()->SetTitle(tmp3);
  hsum->GetYaxis()->SetTitle("Cross section (pb/bin)");
  hsum->Draw();
  leg->AddEntry(hsum->GetName(),"Sum of contributions");

  for(int i=0;i<maxjets;i++){
    hists[i]->Draw("same");
    sprintf(tmp3,"%i-jet sample",i);
    leg->AddEntry(hists[i]->GetName(),tmp3);
  }

  leg->Draw();

  sprintf(tmp2,"%s.eps",quantity);
  cout << "Saving plot as " << tmp2 << endl;
  c1->SaveAs(tmp2);
  return true;
}

