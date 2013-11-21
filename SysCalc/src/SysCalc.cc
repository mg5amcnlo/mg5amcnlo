#include <math.h>
#include <fstream>

#include "SysCalc.h"
#include "tinyxml2.h"
#include "LHAPDF/LHAPDF.h"

using namespace std;
using namespace tinyxml2;
using namespace LHAPDF;

bool DEBUG = false;

std::vector<std::string> &split(const std::string &s, char delim,
		std::vector<std::string> &elems, int maxsplit) {
    std::stringstream ss(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, delim) and elems.size() < maxsplit) {
    	if (item != "") elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim, int maxsplit=20) {
    std::vector<std::string> elems;
    split(s, delim, elems, maxsplit);
    return elems;
}


void SysCalc::clean_tokens(vector<string>& tokens){
// Remove everything after "#"
  for(vector<string>::iterator i = tokens.begin();
      i != tokens.end();++i){
    if((*i)[0] == '#'){
      tokens.assign(tokens.begin(),i);
      break;
    }
  }
}

void SysCalc::write_xsec(){
	double min,max;
	//min = 0.;
	//max = 0.;
	  for (int i=0; i < _scalexsec.size(); i++) {
	  	  	  	  	  	  cout <<"scale fact cross-section  :"<< i <<" "<< _scalexsec[i] <<endl;
	//  	  	  	  	  	  if (min == 0. or _scalexsec[i] < min) min =  _scalexsec[i];
	//  	  	  	  	  	  if (max == 0. or _scalexsec[i] > max) max =  _scalexsec[i];
  	  	  	  	  	  }
	//cout << "min/max:"<< min<< " " << max << endl;
	//min=0.;
	//max =0.;
  	  	  	  	  for (int i=0; i < _alpsxsec.size(); i++) {
  	  	  	  	  	  	  cout <<"alpha_s emission scale fact cross-section  :"<< i <<" "<< _alpsxsec[i] <<endl;
	//  	  	  	  	  	  if (min == 0. or _alpsxsec[i] < min) min =  _alpsxsec[i];
	//  	  	  	  	  	  if (max == 0. or _alpsxsec[i] > max) max =  _alpsxsec[i];
  	  	  	  	  }
  	  	  	cout << "min/max:"<< min<< " " << max << endl;
  	  	min=0.;
  	  	max =0.;
  	  	  	  	  for (int i=0; i < _pdfxsec.size(); i++) {
  	  	  	  	  	  	  cout <<"pdf reweighted cross-section  :"<< i <<" "<< _pdfxsec[i] <<endl;
	  	  	  	  	  	  if (min == 0. or _pdfxsec[i] < min) min =  _pdfxsec[i];
	  	  	  	  	  	  if (max == 0. or _pdfxsec[i] > max) max =  _pdfxsec[i];
    	  	  	  	  	  }
  	  	  	  	cout << "min/max:"<< min<< " " << max << endl;


}



void SysCalc::tokenize(const string& str,
			vector<string>& tokens,
			const string& delimiters)
{
  /**
     Split a string into tokens with given delimiter (default tab and space)
   **/

  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos     = str.find_first_of(delimiters, lastPos);

  if(tokens.size() > 0) tokens.clear();

  while (string::npos != pos || string::npos != lastPos)
  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
  clean_tokens(tokens);
}

void SysCalc::insert_tokens_double(vector<string>& tokens, vector<double>& var)
{
  /**
     Insert tokens into vector var
   **/
  if(tokens.size() > 0) {
    for(vector<string>::iterator str=tokens.begin();
	str != tokens.end();str++)
      var.push_back(atof(str->c_str()));
  }
}

void SysCalc::insert_tokens_int(vector<string>& tokens, vector<int>& var)
{
  /**
     Insert tokens into vector var
   **/
  if(tokens.size() > 0) {
    for(vector<string>::iterator str=tokens.begin();
	str != tokens.end();str++)
      var.push_back(atoi(str->c_str()));
  }
}

SysCalc::SysCalc(istream& conffile,
		   string sysfilename,
		   string orgPDF,
		   int org_member,
		   int beam1, int beam2)
{
  /** Supply config file as an istream. 
      If sysfilename is set, read and parse full systematics file.
      If not, need to supply orgPDF and orgMember (for alpha_s)

      Example of config file:
      # Central scale factors
      scalefact:
      0.5 1 2
      # \alpha_s emission scale factors
      alpsfact:
      0.5 1 2
      # matching scales
      matchscale:
      20 30 40
      # PDF sets, number of members (default/0 means use all members), 
      # combination method (default hessian, note that NNPDF uses gaussian)
      PDF:
      CT10.LHgrid 52 hessian
      MSTW2008nlo68cl.LHgrid
   **/

  _parsed_events = 0;
  _beam[0] = 1;
  _beam[1] = 1;
  _lhe_output = false;
  _event_weight = 1;
  _event_number = 0;
  
  string line;
  vector<string> tokens;
  string typenow = "";
  while(!conffile.eof()){
    getline(conffile, line);
    tokenize(line, tokens);
    if(tokens.size() == 0) continue;
    string first = tokens[0];
    if(first.find("scalefact:") == 0) {
      typenow = "scalefact";
      // Remove first elemens, to assign additional elements to scalefacts
      tokens.erase(tokens.begin());
      insert_tokens_double(tokens, _scalefacts);
    }
    else if(first.find("alpsfact:") == 0) {
      typenow = "alpsfact";
      // Remove first elemens, to assign additional elements to scalefacts
      tokens.erase(tokens.begin());
      insert_tokens_double(tokens, _alpsfacts);
    }
    else if(first.find("matchscale:") == 0) {
      typenow = "matchscale";
      // Remove first elemens, to assign additional elements to scalefacts
      tokens.erase(tokens.begin());
      insert_tokens_double(tokens, _matchscales);
    }
    else if(first.find("PDF:") == 0) {
      typenow = "PDF";
    }
    else if(typenow == "scalefact") {
      insert_tokens_double(tokens, _scalefacts);
    }
    else if(typenow == "alpsfact") {
      insert_tokens_double(tokens, _alpsfacts);
    }
    else if(typenow == "matchscale") {
      insert_tokens_double(tokens, _matchscales);
    }
    else if(typenow == "PDF") {
      _PDFsets.push_back(tokens[0]);
      if(tokens.size() > 1)
	_members.push_back(atoi(tokens[1].c_str()));
      else
	_members.push_back(0);
      if(tokens.size() > 2)
	_combinations.push_back(tokens[2]);
      else
	_combinations.push_back("hessian");
    }
  }

  // Initialize LHAPDF
  initLHAPDF();
  // If too many PDF sets, remove the last ones
  if(_PDFsets.size() > getMaxNumSets()-1) {
    cout << "Warning! LHA only allows " << getMaxNumSets()-1
	 << " PDF sets simultaneously (besides the original set)" << endl;
    cout << "Ignoring additional sets" << endl;
    _PDFsets.erase(_PDFsets.begin()+getMaxNumSets()-1,_PDFsets.end());
  }
  // Initialize all PDF sets
  for(int i=0; i < _PDFsets.size(); i++) {
    cout << "Init PDF set " << _PDFsets[i] << endl;
    int pdfnum = i+1; // Has to start with 1, otherwise get segfault in LHAPDF
    initPDFSet(pdfnum, _PDFsets[i]);
    if(_members[i] == 0) {
      _members[i] = numberPDF(pdfnum);
      if(_members[i] > 1) _members[i]++; // LHAPDF reports wrong for error PDFs
    }
    cout << "Using " << _members[i] << " members for this set" << endl;
  }

  // SET default original PDF and member and beam info
  _orgPDF = orgPDF;
  _org_member = org_member;
  _beam[0] = beam1;
  _beam[1] = beam2;

  // Load systematics file and read orgpdf and beams info
  if(sysfilename != "") {
    // Open file
    _sysfile = new ifstream(sysfilename.c_str(), ifstream::in);
    _filestatus = _sysfile->is_open();
    if (!_filestatus) {
      return;
    }
    // Extract header from file
    char line[1000];
    _sysfile->getline(line, 1000);
    string linestr = line;
    bool done = (linestr.find("<event") != string::npos) || _sysfile->eof();
    bool hold = false;
    while (!done){
      if (!hold) hold = (linestr.find("<initrwgt") != string::npos);
      if (!hold){
	_header_text.append(linestr);
	_header_text.push_back('\n');
      }
      if (hold) hold = (linestr.find("</initrwgt") == string::npos);
      _sysfile->getline(line, 1000);
      linestr = line;
      done = (linestr.find("<event") != string::npos) || _sysfile->eof();
    }
    
    // Make sure that there there is still something in the file
    if (_sysfile->eof()){
      _filestatus = false;
      return;
    }
    // Push back last line read
    _sysfile->putback('\n');
    for(int i = linestr.length() - 1; i > -1; i--)
      _sysfile->putback(linestr[i]);      
    
    // Read orgpdf info
    int start = _header_text.find("<header");
    _xml_document.Parse(_header_text.substr(start).c_str());
    XMLElement* header = _xml_document.FirstChildElement("header");
    if (! header) {
      cout << "Critical Error: header not well formed" << endl;
      cout << "  Please check for unmatched XML <tags>" << endl;
      exit(1);
    }
    XMLElement* orgpdf = header->FirstChildElement("orgpdf");
    if(orgpdf){
      tokenize(orgpdf->GetText(), tokens);
      if(tokens.size() > 0)
	_orgPDF = tokens[0];
      if(tokens.size() > 1)
	_org_member = atoi(tokens[1].c_str());
    }
    // Read beams info
    XMLElement* beams = header->FirstChildElement("beams");
    if(beams){
      tokenize(beams->GetText(), tokens);
      if(tokens.size() < 2) cout << "Warning: <beams> info not correct" << endl;
      _beam[0] = atoi(tokens[0].c_str());
      _beam[1] = atoi(tokens[1].c_str());
      cout << "Set beam info: " << _beam[0] << " " << _beam[1] << endl;
    }
    if(!orgpdf || !beams){
      // Try reading the information from the <init> block
      XMLElement* init = _xml_document.FirstChildElement("init");      
      tokenize(init->GetText(), tokens);
      if(tokens.size() < 8) cout << "Warning: <init> info not correct" << endl;      
      _beam[0] = copysign(1,atoi(tokens[0].c_str()));
      _beam[1] = copysign(1,atoi(tokens[1].c_str()));
      cout << "Set beam info: " << _beam[0] << " " << _beam[1] << endl;
      _orgPDF = tokens[7];
    }
  }
  else
    _filestatus = false;

  cout << "Set original PDF = " << _orgPDF << " with member " << _org_member << endl;

  // Initialize original PDF using _PDFsets.size() as number flag
  cout << "Init original PDF " << _orgPDF << endl;
  int orgnum = _PDFsets.size()+1;
  if(atoi(_orgPDF.c_str()) > 0)
    initPDFSet(orgnum, atoi(_orgPDF.c_str()), _org_member);
  else
    initPDFSet(orgnum, _orgPDF, _org_member);
  cout << "Initialization done" << endl;
}

void SysCalc::fillPDFData(XMLElement* element, vector<int>& pdg, 
			   vector<double>& x, vector<double>& q)
{
  /**
     Fill event PDF data for one beam
   **/
  vector<string> tokens;
  tokenize(element->GetText(), tokens);
  int n_pdf = atoi(tokens[0].c_str());
  for(int i=0; i < n_pdf; i++)
    pdg.push_back(atoi(tokens[i+1].c_str()));
  for(int i=0; i < n_pdf; i++)
    x.push_back(atof(tokens[i+1+n_pdf].c_str()));
  for(int i=0; i < n_pdf; i++)
    q.push_back(atof(tokens[i+1+2*n_pdf].c_str()));
  if(DEBUG){
    cout << "PDG codes: ";
    for(vector<int>::iterator value = pdg.begin(); value != pdg.end();value++)
      cout << *value << " ";
    cout << endl;
    cout << "x values: ";
    for(vector<double>::iterator value = x.begin(); value != x.end();value++)
      cout << *value << " ";
    cout << endl;
    cout << "q values: ";
    for(vector<double>::iterator value = q.begin(); value != q.end();value++)
      cout << *value << " ";
    cout << endl;
  }
}

bool SysCalc::parseEvent(string event)
{
  /**
     Parse one event from the XML file sysfile.
   **/

  // Set up event for parsing

  if(DEBUG) cout << "Start parsing event" << endl;

  if(event != ""){
    _eventtext = event;
  }
  else if (!_filestatus){
    cout << "No more events in file" << endl;
    return false;
  }
  else {
    // Extract one event from file
    _eventtext = "";
    char line[1000];
    string linestr = line;
    bool start = false;
    bool done = false;
    while (!done){
      _sysfile->getline(line, 1000);
      linestr = line;
      if (!start){
	start = linestr.find("<event") != string::npos;
	if(_sysfile->eof()) done = true;
	if (!start) continue;
      }
      done = (linestr.find("</event>") != string::npos) 
	|| (linestr.find("<rwgt>") != string::npos)
	|| _sysfile->eof();
      if (done) linestr = "</event>";
      _eventtext.append(linestr);
      _eventtext.push_back('\n');
    }
    // Check if there is still something in the file
    if (_sysfile->eof()){
      _filestatus = false;
    }
  }
  if(DEBUG) cout << "Read event from file:" << endl;
  if(DEBUG) cout << _eventtext << endl;

  _xml_document.Parse(_eventtext.c_str());
  _event = _xml_document.FirstChildElement("event");

  if(!_event){
    return false;
  }

  if(DEBUG) cout << "Valid event tag found" << endl;

  // Try extracting the event weight if this is an lhe file
  vector<string> tokens;
  if (_event->GetText()) {
    tokenize(_event->GetText(), tokens);
    if(tokens.size() > 6) _event_weight = atof(tokens[3].c_str());
    if(DEBUG) cout << "Set event weight: " << _event_weight << endl;
  }

  _element = _event->FirstChildElement("mgrwt");
  if(!_element){
    cout << "Warning: No element <mgrwt> in event" << endl;
    return false;
  }
  
  if(DEBUG) cout << "Valid mgrwt tag found" << endl;

  // Prepare for filling variables

  _n_qcd = 0;
  _ren_scale = 0;
  _alpsem_scales.clear();
  _pdf_pdg1.clear();
  _pdf_x1.clear();
  _pdf_q1.clear();
  _pdf_pdg2.clear();
  _pdf_x2.clear();
  _pdf_q2.clear();
  _smin = 0;
  _scomp = 0;
  _smax = 0;
  _total_reweight_factor = 1;

  // Start filling variables

  if(DEBUG) cout << "Start reading variables from event" << endl;

  XMLElement* subelement = 0;

  // Event number
  if(_event->IntAttribute("id") != 0)
    _event_number = _event->IntAttribute("id");
  else
    _event_number++;

  if(DEBUG) cout << "Event number: " << _event_number << endl;
    
  // nQCD and Ren scale
  subelement = _element->FirstChildElement("rscale");
  if(subelement){
    tokenize(subelement->GetText(), tokens);
    _n_qcd = atoi(tokens[0].c_str());
    _ren_scale = atof(tokens[1].c_str());
    if(DEBUG) cout << "nQCD = " << _n_qcd << " ren scale = " << _ren_scale << endl;
  }

  // asrwt
  subelement = _element->FirstChildElement("asrwt");
  if(subelement){
    tokenize(subelement->GetText(), tokens);
    int n_alpsem = atoi(tokens[0].c_str());
    tokens.erase(tokens.begin());
    insert_tokens_double(tokens, _alpsem_scales);
    if(DEBUG) {
      cout << "alpsem_scales: ";
      for(vector<double>::iterator value = _alpsem_scales.begin();value != _alpsem_scales.end();value++)
	cout << *value << " ";
      cout << endl;
    }
    if(_alpsem_scales.size() != n_alpsem) 
      cout << "Warning: Wrong number of alpsem scales in event " << _event_number << endl;
  }    

  // matchscales
  subelement = _element->FirstChildElement("matchscale");
  if(subelement){
    tokenize(subelement->GetText(), tokens);
    _smin = atof(tokens[0].c_str());
    _scomp = atof(tokens[1].c_str());
    _smax = atof(tokens[2].c_str());
    if(DEBUG) cout << "smin = " << _smin 
		   << " scomp = " << _scomp
		   << " smax = " << _smax << endl;
  }
  else {
    // No matching scale information, so remove from list
    _matchscales.clear();
  }

  // pdfrwt
  for(int i=0;i<2;i++){
    if(i == 0) subelement = _element->FirstChildElement("pdfrwt");
    else subelement = subelement->NextSiblingElement("pdfrwt");
    if(subelement){
      int beam = atoi(subelement->Attribute("beam"));
      if(DEBUG) cout << "pdf data for beam: " << beam << endl;
      switch(beam) {
      case 1:
	fillPDFData(subelement, _pdf_pdg1, _pdf_x1, _pdf_q1);
	break;
      case 2:
	fillPDFData(subelement, _pdf_pdg2, _pdf_x2, _pdf_q2);
	break;
      }
    }
  }

  // totfact
  subelement = _element->FirstChildElement("totfact");
  if(subelement){
    tokenize(subelement->GetText(), tokens);
    _total_reweight_factor = atof(tokens[0].c_str());
    if(DEBUG) cout << "Reweight factor = " << _total_reweight_factor << endl;
  }
  
  _parsed_events++;

  return true;
}

double SysCalc::calculatePDFWeight(int pdfnum, double fact, int beam,
				    vector<int>& pdg, 
				    vector<double>& x, 
				    vector<double>& q)
{
  double weight = 1;
  int max = pdg.size()-1;
  double maxq = fact*q[max];
  int maxpdg = beam*pdg[max];
  if(abs(maxpdg) == 21) maxpdg = 0;
  maxpdg += 6;
  vector<double> pdfs = xfx(pdfnum, x[max], maxq);
  weight *= pdfs[maxpdg]/x[max];
  int pdgnow;
  for(int i=0; i < max; i++){
    pdfs = xfx(pdfnum, x[i], min(q[i], maxq));
    pdgnow = beam*pdg[i];
    if(abs(pdgnow) == 21) pdgnow = 0;
    pdgnow += 6;
    weight *= pdfs[pdgnow]/x[i];
    pdfs = xfx(pdfnum, x[i+1], min(q[i], maxq));
    pdgnow = beam*pdg[i+1];
    if(abs(pdgnow) == 21) pdgnow = 0;
    pdgnow += 6;
    weight /= (pdfs[pdgnow]/x[i+1]);
  }
  if(DEBUG) cout << "PDF weight one side: " << weight << endl;
  return weight;
}


bool SysCalc::convertEvent()
{
  // Set which member to use for alpha_s reweighting
  int orgnum = _PDFsets.size() + 1;
  if (DEBUG) cout << "Original PDF number: " << orgnum << endl;
 
  // Calculate original weight for the different factors
  double org_ren_alps = 1;
  if(_n_qcd > 0) org_ren_alps = pow(alphasPDF(orgnum, _ren_scale), _n_qcd);
  if (DEBUG) cout << "Org central alps value: " << alphasPDF(orgnum, _ren_scale) << endl;
  if (DEBUG) cout << "Org central alps factor: " << org_ren_alps << endl;

  double org_em_alps = 1;
  for(int i=0; i < _alpsem_scales.size(); i++)
    org_em_alps *= alphasPDF(orgnum, _alpsem_scales[i]);
  if (DEBUG) cout << "Org emission alps factor: " << org_em_alps << endl;

  double org_pdf_fact = 1;
  org_pdf_fact *= calculatePDFWeight(orgnum, 1., _beam[0], _pdf_pdg1, _pdf_x1, _pdf_q1);
  org_pdf_fact *= calculatePDFWeight(orgnum, 1., _beam[1], _pdf_pdg2, _pdf_x2, _pdf_q2);
  if (DEBUG) cout << "Org PDF factor: " << org_pdf_fact << endl;

  double org_weight = org_ren_alps*org_em_alps*org_pdf_fact;
  if (DEBUG) cout << "Calculated reweight factor: " << org_weight << endl;

//  if (fabs(org_weight - _total_reweight_factor)/
//      (org_weight+_total_reweight_factor) > 5e-3) {
//    cout << "Warning: Reweight factor differs in event "
//	 << _event_number << ": ";
//    cout << org_ren_alps*org_em_alps*org_pdf_fact << " (cf. "
//	 << _total_reweight_factor << ")" << endl;
//  }

  /*** Perform reweighting ***/

  _scaleweights.clear();
  _alpsweights.clear();
  _matchweights.clear();
  for (int i=0;i < _PDFweights.size(); i++)
    _PDFweights[i].clear();
  _PDFweights.clear();

  // Reweight central scale (scalefact)
  for (int i=0; i < _scalefacts.size(); i++) {

    double sf = _scalefacts[i];
    if (DEBUG) cout << "Reweight with scalefact " << sf << endl;

    double ren_alps = 1;
    if(_n_qcd > 0) ren_alps = pow(alphasPDF(orgnum, sf*_ren_scale), _n_qcd);
    if (DEBUG) cout << "New central alps factor: " << ren_alps << endl;
    
    double pdf_fact = 1;
    pdf_fact *= calculatePDFWeight(orgnum, sf, _beam[0], _pdf_pdg1, _pdf_x1, _pdf_q1);
    pdf_fact *= calculatePDFWeight(orgnum, sf, _beam[1], _pdf_pdg2, _pdf_x2, _pdf_q2);
    if (DEBUG) cout << "New PDF factor: " << pdf_fact << endl;

    _scaleweights.push_back(ren_alps*pdf_fact/(org_ren_alps*org_pdf_fact) * _event_weight);
    if (_scaleweights.size() -1 == _scalexsec.size()){
       	_scalexsec.push_back(_scaleweights[_scaleweights.size()-1]);
       }else{
       	_scalexsec[_scaleweights.size()-1] += _scaleweights[_scaleweights.size()-1];
       }

    if (DEBUG) cout << "scalefact weight: " << _scaleweights[_scaleweights.size()-1]
		    << endl;
  }

  // alpsfact
  for(int j=0; j < _alpsfacts.size(); j++){
    double as = _alpsfacts[j];
    if (DEBUG) cout << "Reweight with alpsfact " << as << endl;
    double em_alps = 1;
    for(int i=0; i < _alpsem_scales.size(); i++)
      em_alps *= alphasPDF(orgnum, as*_alpsem_scales[i]);
    if (DEBUG) cout << "New emission alps factor: " << org_em_alps << endl;
    _alpsweights.push_back(em_alps/org_em_alps * _event_weight);
    if (DEBUG) cout << "alpsfact weight: " << _alpsweights[_alpsweights.size()-1]
		    << endl;

    if (_alpsweights.size() -1 == _alpsxsec.size()){
       	_alpsxsec.push_back(_alpsweights[_alpsweights.size()-1]);
    }else{
       	_alpsxsec[_alpsweights.size()-1] += _alpsweights[_alpsweights.size()-1];
    }
  }

  // Different PDF sets
  for(int i=0; i < _PDFsets.size(); i++){
    int pdfnum = i+1;
    // Initialize a new vector for the values of the members of this PDF
    if (DEBUG) cout << "Reweighting with PDF set " << pdfnum << ": "
		    << _PDFsets[i] << endl;
    vector<double>* pdffacts = new vector<double>;
    for(int j=0; j < _members[i]; j++) {
      if (DEBUG) cout << "PDF set member " << j << endl;      
      // Set PDF set member
      usePDFMember(pdfnum, j);
      // First recalculate alpha_s weights, since alpha_s differs between PDFs
      // Recalculate PDF weights
      double pdf_fact = 1;
      if(_n_qcd > 0) {
	pdf_fact *= pow(alphasPDF(pdfnum,_ren_scale), _n_qcd);
	if (DEBUG) cout << "New central alps value due to PDF: " << alphasPDF(pdfnum, _ren_scale) << endl;
	if (DEBUG) cout << "New alps fact due to PDF: " << pdf_fact << endl;
      }
      if (DEBUG) cout << "After PDF central alps factor: " << pdf_fact << endl;
      for(int k=0; k < _alpsem_scales.size(); k++) {
	if (DEBUG) cout << "Emission alpha_s due to PDF: " 
			<< alphasPDF(pdfnum, _alpsem_scales[k]) 
			<< " for scale " << _alpsem_scales[k] << endl;
	pdf_fact *= alphasPDF(pdfnum, _alpsem_scales[k]);
      }
      if (DEBUG) cout << "After PDF emission alps factor: " << pdf_fact << endl;
      pdf_fact *= calculatePDFWeight(pdfnum, 1., _beam[0], _pdf_pdg1, _pdf_x1, _pdf_q1);
      pdf_fact *= calculatePDFWeight(pdfnum, 1., _beam[1], _pdf_pdg2, _pdf_x2, _pdf_q2);
      if (DEBUG) cout << "Total PDF factor: " << pdf_fact << endl;
      pdffacts->push_back(pdf_fact/org_weight * _event_weight);
      if (DEBUG) cout << "PDF weight: " << (*pdffacts)[pdffacts->size()-1] << endl;
      if (pdffacts->size() -1 == _pdfxsec.size()){
         	_pdfxsec.push_back(pdf_fact/org_weight * _event_weight);
      }else{
         	_pdfxsec[pdffacts->size() -1] += pdf_fact/org_weight * _event_weight;
      }


    }
    _PDFweights.push_back(*pdffacts);
  }

  // matching scale weights
  for(int i=0; i < _matchscales.size(); i++) {
    double ms = _matchscales[i];
    if (DEBUG) cout << "Reweight with matchscale " << ms << endl;
    if(ms > _smin || _smax > max(ms, _scomp))
      _matchweights.push_back(0);
    else
      _matchweights.push_back(1);
    if (DEBUG) cout << "Matching weight: " << _matchweights[_matchweights.size()-1]
		    << endl;
  }

  return true;
}

bool SysCalc::writeHeader(ostream& outfile)
{
  if (!_lhe_output) {
    // Write a clean header
    outfile << "<header>\n";
  }
  else {
    // First copy the full header text, then add rwgt info at the end
    int end = _header_text.find("</header");
    outfile << _header_text.substr(0, end);
  }
  outfile << "<initrwgt>\n";
  int weight_id = 0;
  if(_scalefacts.size() > 0) {
    outfile << "  <weightgroup type=\"Central scale variation\" combine=\"envelope\">" << endl;
    for (int i=0; i < _scalefacts.size(); i++)
      outfile << "    <weight id=\"" << ++weight_id << "\"> scalefact="
	      << _scalefacts[i] << "</weight>" << endl;
    outfile << "  </weightgroup>" << endl;
  }
  if(_alpsfacts.size() > 0) {
    outfile << "  <weightgroup type=\"Emission scale variation\" combine=\"envelope\">" << endl;
    for (int i=0; i < _alpsfacts.size(); i++)
      outfile << "    <weight id=\"" << ++weight_id << "\"> alpsfact="
	      << _alpsfacts[i] << "</weight>" << endl;
    outfile << "  </weightgroup>" << endl;
  }
  if(_PDFsets.size() > 0){
    for (int i=0; i < _PDFsets.size(); i++){
      outfile << "  <weightgroup type=\"" << _PDFsets[i] 
	      << "\" combine=\"" << _combinations[i] << "\">" << endl;
      for (int j=0; j < _members[i]; j++)
	outfile << "    <weight id=\"" << ++weight_id << "\">Member "
		<< j << "</weight>" << endl;
      outfile << "  </weightgroup>" << endl;
    }
  }
  if(_matchscales.size() > 0) {
    outfile << "  <weightgroup type=\"Matching scale variation\" combine=\"envelope\">" << endl;
    for (int i=0; i < _matchscales.size(); i++)
      outfile << "    <weight id=\"" << ++weight_id << "\"> Qmatch="
	      << _matchscales[i] << "</weight>" << endl;
    outfile << "  </weightgroup>" << endl;
  }
  outfile << "</initrwgt>\n";
  if (!_lhe_output)
    outfile << "</header>\n";
  else {
    // Add rest of header text and init info
    int start = _header_text.find("</header");
    outfile << _header_text.substr(start);    
  }
  return true;
}

bool SysCalc::writeEvent(ostream& outfile)
{
  if (!_lhe_output) {
    // Write a clean event
    outfile << "<event";
    if (_event_number > 0) outfile << " id=\"" << _event_number << "\"";
    outfile << ">\n";
  }
  else {
    // First copy the full event text, then add rwgt info at the end
    int end = _eventtext.find("</event");
    outfile << _eventtext.substr(0, end);
  }
  outfile << "<rwgt>\n";
  int weight_id = 0;
  if(_scaleweights.size() > 0) {
    for (int i=0; i < _scaleweights.size(); i++) {
      outfile << "  <wgt id=\"" << ++weight_id << "\">";
      outfile << _scaleweights[i]*_event_weight; 
      outfile << "</wgt>\n";
    }
  }
  if(_alpsweights.size() > 0) {
    for (int i=0; i < _alpsweights.size(); i++) {
      outfile << "  <wgt id=\"" << ++weight_id << "\">";
      outfile << _alpsweights[i]*_event_weight; 
      outfile << "</wgt>\n";
    }
  }
  if(_PDFweights.size() > 0){
    for (int i=0; i < _PDFweights.size(); i++){
      for (int j=0; j < _PDFweights[i].size(); j++) {
	outfile << "  <wgt id=\"" << ++weight_id << "\">";
	outfile << _PDFweights[i][j]*_event_weight; 
	outfile << "</wgt>\n";
      }
    }
  }
  if(_matchweights.size() > 0) {
    for (int i=0; i < _matchweights.size(); i++) {
      outfile << "  <wgt id=\"" << ++weight_id << "\">";
      outfile << _matchweights[i]*_event_weight; 
      outfile << "</wgt>\n";
    }
  }
  outfile << "</rwgt>\n";
  outfile << "</event>\n";
  return true;
}
