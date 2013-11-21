#ifndef SysCalc_h
#define SysCalc_h

#include <iostream>
#include <vector>

#include "tinyxml2.h"

using namespace std;
using namespace tinyxml2;

class SysCalc
{
 public:
  SysCalc() {_parsed_events = 0; };
  SysCalc(istream& conffile, string sysfilename = "",
	   string orgPDF = "cteq6ll.LHpdf",
	   int orgMember = 0, int beam1 = 1, int beam2 = 1);
  
  bool parseEvent(string event = "");
  bool convertEvent();
  bool writeHeader(ostream& outfile);
  bool writeEvent(ostream& outfile);

  void lheOutput(bool lheoutput) { _lhe_output = lheoutput; }
  bool lheOutput() { return _lhe_output; }

  bool fileStatus() { return _filestatus; }
  int parsedEvents() { return _parsed_events; }
  void write_xsec();

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
  double calculatePDFWeight(int pdfnum, double fact, int beam,
			    vector<int>& pdg, 
			    vector<double>& x, 
			    vector<double>& q);

 private:
  /*** Original PDF info, needed for alpha_s reweighting ***/
  string _orgPDF;
  int _org_member;
  // Info about beams (+-1) needed for PDF calculation for Tevatron
  int _beam[2];

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
  // Combination method for members in each of the sets
  vector<string> _combinations;
  // Whether or not to write full event information
  bool _lhe_output;

  
  /*** Parser variables ***/
  ifstream* _sysfile;
  XMLDocument _xml_document;
  bool _filestatus; // true if open, otherwise false
  string _header_text;
  XMLElement* _event;
  XMLElement* _element;
  int _parsed_events;

  /*** Event variables ***/
  string _eventtext;
  int _event_number;
  double _event_weight;
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

  /*** Final systematics cross-section ***/
  // Central scale cross-section (ratio)
  vector<double> _scalexsec;
  // alpha_s emission scale cross-section (ratio)
  vector<double> _alpsxsec;
  // PDF cross-section (ratio)
  vector<double> _pdfxsec;

};

#endif /* SysCalc_h */
