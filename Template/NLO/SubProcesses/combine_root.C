//Sums root histograms produced by MadGraph5_aMC@NLO and saves 
//the results in a single root file. 
//
//This code is steered by combine_root.sh, but can be executed directly
//from a shell with the command:
//   root -l -b combine_root.C
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
#include <TMath.h>
#include <TObject.h>
#include <TH1D.h>
#include <TH2D.h>

using namespace std;

void combine_root(){

  //Set myverbose=true for printing debugging messages
  bool myverbose=false;

  //The list of root files to be combined must be in temp_root_files.txt.
  //See combine_root.sh to see how it is created
  ifstream fileoffiles("temp_root_files.txt");

  //Maximum number of files: may increase this number if memory permits.
  //Note that only one file at the time is open, so I reckon this limitation
  //can be easily lifted
  int maxfiles=1000;

  if(fileoffiles.is_open())
    {
      string filenames[maxfiles];

      int numoffiles;
      fileoffiles >> numoffiles;

      if(numoffiles>maxfiles){
	std::cout << "Error: increase maxfiles in combine_root.C" << std::endl;
	return;
      }

      string currentdir;
      fileoffiles >> currentdir;

      for(int i = 0; i < numoffiles; ++i)
        {
	  fileoffiles >> filenames[i];
        }

      std::cout << "  " << std::endl;
      std::cout << "Done reading files: # = " << numoffiles << std::endl;
      if(myverbose){
	std::cout << "  " << std::endl;
	std::cout << "Files are: " << std::endl;
	for(int i = 0; i < numoffiles; ++i)
	  {
	    std::cout << i << " " << filenames[i] << std::endl;
	  }
      }

    }
  else {
    std::cout << "Error in combine_root.C" << std::endl;
    std::cout << "Probably cant find temp_root_files.txt" << std::endl;
    return ;
  }


  //These are the source root files, whose names have been read before
  TFile HistoFiles;
  TString tempstring;
  vector<TH1D*>  HistoContents;
  TList *KeyList;

  //Target histograms
  vector<TH1D*>  CombinedHistos;
  CombinedHistos.clear();

  for(int i = 0; i < numoffiles; ++i)
    {
      if(myverbose){
	std::cout << "  " << std::endl;
	std::cout << "Dealing with #" << i << " " << filenames[i] << std::endl;
      }
      tempstring = filenames[i];
      TFile HistoFiles(tempstring);
      if(!HistoFiles.IsOpen()){
	std::cout << "Error in combine_root.C" << std::endl;
	std::cout << "File is not open" << i << " " << filenames[i] << std::endl;
	return ;
      }
      
      HistoContents.clear();
      KeyList=HistoFiles.GetListOfKeys();
      
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

      if(i==0){
	int llength0 = llength;
	for(int j = 0; j <  llength; j++){
	  TString histname = HistoContents[j]->GetName();
	  if(myverbose){
	    cout << "Now filling: " << histname << " " << endl;
	    cout << "First bin: " << HistoContents[j]->GetBinContent(1) << endl;
	  }
          TH1D* temphist = HistoContents[j];
	  //WARNING: root deletes from memory all variables associated with a given file
	  //when the file is closed. The following instruction prevents this from happening.
	  //DO NOT REMOVE
          temphist->SetDirectory(0);
	  //If the name of the combined histogram needs be changed, use what follows
	  //However, by doing so the sum done below will not work in the present form
	  //TString newname = "S"+histname;
	  //temphist->SetName(newname);
	  CombinedHistos.push_back(temphist);
	}
      }
      else {
	if(llength != llength0)
	  {
	    std::cout << "Error in combine_root.C" << std::endl;
	    std::cout << "Probably different contents in 0 and " << i << std::endl;
	    std::cout << llength << " " << llength0  << std::endl;
	    return ;
	  };
	for(vector<TH1D*>::iterator it = CombinedHistos.begin();it != CombinedHistos.end();++it){
	  TString histname = (*it)->GetName();
          TH1D* temphist2 = (TH1D*)HistoFiles.Get(histname);
          temphist2->SetDirectory(0);
          (*it)->Add(temphist2);
          if(myverbose){
	    cout << "Now summing " << histname << " with " << temphist2->GetName() << endl;
	    cout << "Result in first bin: " << (*it)->GetBinContent(1) << endl;
	  }
        }
      }
      
       HistoFiles.Close();
    }

  //All done: write final results
  tempstring=currentdir+"/MADatNLO.root";
  TFile FinalFile(tempstring,"RECREATE");

  if(!FinalFile.IsOpen()){
    std::cout << "Error in combine_root.C" << std::endl;
    std::cout << "Target file is not open" << std::endl;
    return ;
  }
  
  if(myverbose)cout << " "  << endl;
  for(vector<TH1D*>::iterator it = CombinedHistos.begin();it != CombinedHistos.end();++it){
    if(myverbose){
      cout << "Final histo " << (*it)->GetName() << endl;
      cout << "Result in first bin: " << (*it)->GetBinContent(1) << endl;
    }
    (*it)->Write();
  }  
  if(myverbose){
    cout << " " << endl;
    cout << "List of final histograms " << endl;
    cout << FinalFile.GetListOfKeys()->Print() << endl;
  }
  FinalFile.Close();
  
  std::cout << "  " << std::endl;
  std::cout << "All done. Final result in:" << std::endl;
  std::cout << tempstring << std::endl;
}
