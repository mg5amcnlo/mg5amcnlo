#include "LHEF.h"
#include <iomanip>
#include <string>

class MyReader {

  private:

  public:

  MyReader() : filename(""), reader("") {}
  MyReader(std::string file) : filename(file), reader(file) {}
  ~MyReader(){}

  std::string filename;
  LHEF::Reader reader;


  void lhef_read_wgtsinfo_(int &cwgtinfo_nn, char (cwgtinfo_weights_info[250][15])) {

    // Read header of event file
    std::stringstream hss;
    std::string hs,hsvec[5];
    cwgtinfo_nn=0;

    while (true){
      hss << reader.headerBlock;
      std::getline(hss,hs,'\n');
      if (hs.find("</header>") != std::string::npos) break;

      // Read the scale block
      if (hs.find("<weightgroup type='scale_variation'") != std::string::npos) {
	while (true) {
	  std::getline(hss,hs,'\n');
	  if (hs.find("</weightgroup>") != std::string::npos) break;
          //find the values of muR and muF
	  std::string xmuR = hs.substr(hs.find("muR")+4,hs.length());
	  xmuR = xmuR.substr(0,xmuR.find("muF")-1);
	  std::string xmuF = hs.substr(hs.find("muF")+4,hs.length());
	  xmuF = xmuF.substr(0,xmuF.find("</w")-1);
	  double muR = atof(xmuR.c_str());
	  double muF = atof(xmuF.c_str());
          //store the plot label
          sprintf(cwgtinfo_weights_info[cwgtinfo_nn], "muR=%2.1f muF=%2.1f", muR, muF);
	  ++cwgtinfo_nn;
	}
      }

      // Read the PDF block
      if (hs.find("<weightgroup type='PDF_variation'") != std::string::npos) {
	while (true) {
	  std::getline(hss,hs,'\n');
	  if (hs.find("</weightgroup>") != std::string::npos) break;
          //find the PDF set used
	  std::string PDF = hs.substr(hs.find("pdfset")+8,hs.length());
	  PDF = PDF.substr(0,PDF.find("</w")-1);
          int iPDF = atoi(PDF.c_str());
          //store the plot label
          sprintf(cwgtinfo_weights_info[cwgtinfo_nn], "PDF=%8d   ", iPDF);
	  ++cwgtinfo_nn;
	}
      }
    }
  }


  void lhef_read_wgts_(double (cwgt_ww[250])) {
    
    // Read events
    if (reader.readEvent()) {
      double wgtxsecmu[4][4],wgtxsecPDF[200];
      std::string svec[16],isvec[4],refstr;
      std::stringstream ss;
      int i,j;
  
      // Read aMCatNLO extra informations
      ss << reader.eventComments;
      for (i=0; i<=15; i++) ss >> svec[i];
      std::string ch1 = svec[0];
      int iSorH_lhe = atoi(svec[1].c_str());
      int ifks_lhe = atoi(svec[2].c_str());
      int jfks_lhe = atoi(svec[3].c_str());
      int fksfather_lhe = atoi(svec[4].c_str());
      int ipartner_lhe = atoi(svec[5].c_str());
      double scale1_lhe = atof(svec[6].c_str());
      double scale2_lhe = atof(svec[7].c_str());
      int jwgtinfo = atoi(svec[8].c_str());
      int mexternal = atoi(svec[9].c_str());
      int iwgtnumpartn = atoi(svec[10].c_str());
      double wgtcentral = atof(svec[11].c_str());
      double wgtmumin = atof(svec[12].c_str());
      double wgtmumax = atof(svec[13].c_str());
      double wgtpdfmin = atof(svec[14].c_str());
      double wgtpdfmax = atof(svec[15].c_str());

      // Reweighting
      std::string s;
      int iww = 1;
      if (jwgtinfo != 9) {
	std::exit;
      }
      else {
	ss << reader.headerBlock;
	while (true) {
	  ss << reader.headerBlock;
	  std::getline(ss,s,'\n');
	  if (s.find("</rwgt>") != std::string::npos) break;
	  if (s.find("id=") != std::string::npos) {
	    std::string sww = s.substr(s.find("id=")+11,s.length());
	    sww = sww.substr(0,sww.find("</w")-1);
	    cwgt_ww[iww] = atof(sww.c_str());
	    ++iww;
	  }
	}
	iww-=1;
      }
    }
  }
};
