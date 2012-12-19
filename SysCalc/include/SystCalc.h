#ifndef SystCalc_h
#define SystCalc_h

#include <iostream>
#include <vector>

#include "tinyxml2.h"

using namespace std;
using namespace tinyxml2;

class SystCalc
{
 public:
  SystCalc() {_parsed_events = 0; };
  SystCalc(istream& conffile, string sysfilename = "",
	   string orgPDF = "cteq6ll.LHpdf",
	   int orgMember = 0);
  
  bool parseEvent(string event = "");
  bool convertEvent();
  bool writeHeader(ostream& outfile);
  bool writeEvent(ostream& outfile);

  bool noFile() { return _filestatus != 0; }
  XMLError fileStatus() { return _filestatus; }
  int parsedEvents() { return _parsed_events; }

 protected:
  // Helper functions
  void tokenize(const string& str,
		vector<string>& tokens,
		const string& delimiters = "\t ");
  void clean_tokens(vector<string>& tokens);
  void insert_tokens_double(vector<string>& tokens, vector<double>& var);
  void insert_tokens_int(vector<string>& tokens, vector<int>& var);
  void fillPDFData(XMLElement* element, vector<int>& pdg, 
		   vector<double>& x, vector<double>& q);
  double calculatePDFWeight(int pdfnum, double fact, 
			    vector<int>& pdg, 
			    vector<double>& x, 
			    vector<double>& q);

 private:
  /*** Original PDF info, needed for alpha_s reweighting ***/
  string _orgPDF;
  int _org_member;

  /*** Conversion variables ***/
  // Central scale factors (typically, 0.5,1,2)
  vector<double> _scalefacts;
  // alpha_s emission scale factors (typically, 0.5,1,2)
  vector<double> _alpsfacts;
  // matching scales (in GeV)
  vector<double> _matchscales;
  // LHAPDF PDF sets
  vector<string> _PDFsets;
  // Number of members in each of the sets
  vector<int> _members;
  
  /*** Parser variables ***/
  XMLDocument _sysfile;
  XMLError _filestatus;
  XMLElement* _element;
  int _parsed_events;

  /*** Event variables ***/
  int _event_number;
  int _n_qcd;
  double _ren_scale;
  vector<double> _alpsem_scales;
  vector<int> _pdf_pdg1;
  vector<double> _pdf_x1;
  vector<double> _pdf_q1;
  vector<int> _pdf_pdg2;
  vector<double> _pdf_x2;
  vector<double> _pdf_q2;
  double _smin, _scomp, _smax;
  double _total_reweight_factor;

  /*** Final systematics weights ***/
  // Central scale weights
  vector<double> _scaleweights;
  // alpha_s emission scale weights
  vector<double> _alpsweights;
  // PDF weights
  vector< vector<double> > _PDFweights;
  // matching scale weights
  vector<double> _matchweights;
};

#endif /* SystCalc_h */
