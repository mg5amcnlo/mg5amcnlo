#include "LHEF.h"
#include <iomanip>
#include <string>


extern "C" {
    int get_wgts_info_len(void);
}

const int wgts_info_len_used = 80;

class MyReader {

  private:

  public:

  MyReader() : filename(""), reader("") {}
  MyReader(std::string file) : filename(file), reader(file) {}
  ~MyReader(){}

  std::string filename;
  LHEF::Reader reader;


  void lhef_read_wgtsinfo_(int &cwgtinfo_nn, char (cwgtinfo_weights_info[1024][wgts_info_len_used])) {

    // check that wgts_info_len_used > wgts_info_len from the fortran module
    if (wgts_info_len_used != get_wgts_info_len()) {
      std::cout << " wgts_info_len_used is different from the corresponding quantity in HwU_wgts_info_len\n"
                << " Please set wgts_info_len_used in LHEFRead.h to "<< get_wgts_info_len() <<"\n"
                << " Program stopped.\n";
      exit(1);
    }

    // Read header of event file
    std::stringstream hss;
    std::string hs;
    std::stringstream format_str;
    format_str <<"%"<<wgts_info_len_used<<"s";
    sprintf(cwgtinfo_weights_info[0],format_str.str().c_str(),"central value");
    cwgtinfo_nn=1;
    while (true){
      hss << reader.headerBlock;
      std::getline(hss,hs,'\n');
      if (hs.find("</header>") != std::string::npos) break;
      // Read the wgt information
      if (hs.find("<initrwgt>") != std::string::npos) {
	while (true) {
	  std::getline(hss,hs,'\n');
	  if (hs.find("</initrwgt>") != std::string::npos) break;
	  if (hs.find("<weightgroup") != std::string::npos) continue;
	  if (hs.find("</weightgroup>") != std::string::npos) continue;
	  if (hs.find("<weight id") != std::string::npos) {
	    std::string sRWGT = hs.substr(hs.find("'>")+2,hs.find("</w")-3);
	    sRWGT = sRWGT.substr(0,sRWGT.find("<"));
	    sprintf(cwgtinfo_weights_info[cwgtinfo_nn],format_str.str().c_str(),sRWGT.c_str());
	    ++cwgtinfo_nn;
	  }
	}
      }
      
    }
  }


  void lhef_read_wgts_(double (cwgt_ww[1024])) {
    
    // Read events
    if (reader.readEvent()) {
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
	    int ioffs=s.find("'>")+2;
	    std::string sww = s.substr(ioffs,s.length());
	    sww = sww.substr(0,sww.find("</w")-1);
	    cwgt_ww[iww] = atof(sww.c_str());
	    ++iww;
	  }
	}
      }
    }
  }
};
