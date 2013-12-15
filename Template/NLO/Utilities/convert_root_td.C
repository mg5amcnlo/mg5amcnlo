//Converts root histograms to topdrawer format. The output should
//by readable by the read*.for utility files
//
//This code can be executed directly from a shell with the command:
//   root -l -b convert_root_td.C
//
//It has beed tested on SLC6 machines with root v5.34.11. 
//It may not work with older versions

#include <TCanvas.h>
#include <TFile.h>
#include <TList.h>
#include <TKey.h>
#include <TString.h>
#include <TH1.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <THStack.h>
#include <TLegend.h>
#include <TLatex.h>
#include <string>
#include <sstream>
#include <fstream>
#include <ostream>
#include <TMath.h>
#include <TObject.h>
#include <TH1D.h>
#include <TH2D.h>

using namespace std;

void convert_root_td(){

  //Set myverbose=true for printing debugging messages
  bool myverbose=false;

  //Target file
  ofstream targetfile("MADatNLO.root.top");

  //root file to be converted
  TFile SourceFile;
  TString tempstring;
  tempstring="MADatNLO.root";

  TFile SourceFile(tempstring);
  if(!SourceFile.IsOpen()){
  	std::cout << "Error in combine_root.C" << std::endl;
  	std::cout << "File is not open" << i << " " << filenames[i] << std::endl;
  	return ;
  }
      
  //Keep only 1-dimensional histograms in double format
  vector<TH1D*>  HistoContents;
  TList *KeyList;

  KeyList=SourceFile.GetListOfKeys();
  for(int j = 0; j <  KeyList->GetEntries(); j++){
    //Check class name. If not TH1D, skip and continue
    TKey *currentKey  = (TKey *)KeyList->At(j);
    TString className = currentKey->GetClassName();
    if (! className.CompareTo("TH1D")==0) continue;
    TH1D *thishist = (TH1D*)currentKey->ReadObj();
    HistoContents.push_back(thishist);
  }
      
  int llength = HistoContents.size();
  if(myverbose)cout << "Total number of histograms: " << llength << endl;

  for(vector<TH1D*>::iterator it = HistoContents.begin();it != HistoContents.end();++it){
    int nbins = (*it)-> GetNbinsX();
    double xmin = (*it)-> GetBinCenter(1);
    double xmax = (*it)-> GetBinCenter(nbins);
    double bwidth = (*it)-> GetBinWidth(1);
    double drange = xmin-bwidth/2;
    double urange = xmax+bwidth/2;
    int iwheremin = (*it)-> GetMinimumBin();
    int iwheremax = (*it)-> GetMaximumBin();
    double ymin = (*it)-> GetBinContent(iwheremin);
    double ymax = (*it)-> GetBinContent(iwheremax);
    //That is smart: root computes the integral, and then normalises it to one...
    double xintegral = (*it)->ComputeIntegral();
    double entries  = (*it)->GetEntries();
    TString histtitle = (*it)->GetTitle();
    if(myverbose){
      cout << "Now doing: " << (*it)->GetName() << endl;
      cout << "Title: " << histtitle << endl;
      cout << "bins: " << nbins << " " << xmin << " " << xmax << " " << bwidth << endl;
      cout << "min, max: " << ymin << " " << ymax  << endl;
    }
    targetfile << " (" << endl;
    targetfile << " new plot" << endl;
    targetfile << " (" << endl;
    targetfile << "  TITLE    TOP   " << '"' << " " << histtitle;
    targetfile << " "  << '"' << endl;
    targetfile << "  SET SCALE Y LIN" << endl;
    targetfile << "  SET LIMITS X " << drange << "  " << urange << endl;
    targetfile << "( SET LIMITS Y " << ymin*0.95 << "  " << ymax*1.05 << endl;
    targetfile << "  SET ORDER X Y DY" << endl;
    targetfile << " (  " << histtitle << endl;
    targetfile << " ( INT=" << xintegral << "    ENTRIES=" << entries << endl;
    for(int j = 1; j <= nbins; j++){
      targetfile << "   " << (*it)->GetBinCenter(j);
      targetfile << "   " << (*it)->GetBinContent(j);
      targetfile << "   " << (*it)->GetBinError(j) << endl;
    }
    targetfile << "  HIST SOLID" << endl;
    targetfile << "   PLOT" << endl;
  }

  SourceFile.Close();
  targetfile.close();

}
