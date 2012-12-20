#include <math.h>

#include "SystCalc.h"
#include "tinyxml2.h"
#include "LHAPDF/LHAPDF.h"

using namespace std;
using namespace tinyxml2;
using namespace LHAPDF;

bool DEBUG = false;

void SystCalc::clean_tokens(vector<string>& tokens){
// Remove everything after "#"
  for(vector<string>::iterator i = tokens.begin();
      i != tokens.end();++i){
    if((*i)[0] == '#'){
      tokens.assign(tokens.begin(),i);
      break;
    }
  }
}

void SystCalc::tokenize(const string& str,
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

void SystCalc::insert_tokens_double(vector<string>& tokens, vector<double>& var)
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

void SystCalc::insert_tokens_int(vector<string>& tokens, vector<int>& var)
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

SystCalc::SystCalc(istream& conffile,
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
  _beam[0]=1;
  _beam[1]=1;
  
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
  for(int i=0; i < max(int(_PDFsets.size()), getMaxNumSets()-1); i++) {
    cout << "Init PDF set " << _PDFsets[i] << endl;
    int pdfnum = i+1; // Has to start with 1, otherwise get segfault in LHAPDF
    if(atoi(_PDFsets[i].c_str()) > 0)
      initPDFSet(pdfnum, atoi(_PDFsets[i].c_str()));
    else
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
    _filestatus = _sysfile.LoadFile(sysfilename.c_str());
    if (_filestatus != 0) {
      return;
    }
    // Read orgpdf info
    XMLElement* element = _sysfile.FirstChildElement("orgpdf");
    if(element){
      tokenize(element->GetText(), tokens);
      if(tokens.size() > 0)
	_orgPDF = tokens[0];
      if(tokens.size() > 1)
	_org_member = atoi(tokens[1].c_str());
    }
    else
      cout << "Warning: Failed to read orgpdf from systematics file" << endl;
    // Read beams info
    element = _sysfile.FirstChildElement("beams");
    if(element){
      tokenize(element->GetText(), tokens);
      if(tokens.size() < 2) cout << "Warning: <beams> info not correct" << endl;
      _beam[0] = atoi(tokens[0].c_str());
      _beam[1] = atoi(tokens[1].c_str());
      cout << "Set beam info: " << _beam[0] << " " << _beam[1] << endl;
    }
    else
      cout << "Warning: Failed to read beams from systematics file" << endl;
  }
  else
    _filestatus = XML_ERROR_FILE_NOT_FOUND;

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

void SystCalc::fillPDFData(XMLElement* element, vector<int>& pdg, 
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

bool SystCalc::parseEvent(string event)
{
  /**
     Parse one event from the XML file sysfile.
   **/

  // Set up event for parsing

  if(event != ""){
    _sysfile.Parse(event.c_str());
    _element = _sysfile.FirstChildElement("mgrwt");
  }
  else if (_filestatus != 0) 
    return false;
  else if (_parsed_events == 0)
    _element = _sysfile.FirstChildElement("mgrwt");
  else
    _element = _element->NextSiblingElement("mgrwt");
  
  // Prepare for filling variables

  _event_number = 0;
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

  if(!_element){
    return false;
  }

  // Start filling variables
  vector<string> tokens;

  XMLElement* subelement = 0;

  // Event number
  if(_element->Attribute("event")){
    _event_number = atoi(_element->Attribute("event"));
    if(DEBUG) cout << "Event number: " << _event_number << endl;
  }

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

double SystCalc::calculatePDFWeight(int pdfnum, double fact, int beam,
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


bool SystCalc::convertEvent()
{
  // Set which member to use for alpha_s reweighting
  int orgnum = _PDFsets.size() + 1;
  
  // Calculate original weight for the different factors
  double org_ren_alps = 1;
  if(_n_qcd > 0) org_ren_alps = pow(alphasPDF(orgnum, _ren_scale), _n_qcd);
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

  if (fabs(org_weight - _total_reweight_factor)/
      (org_weight+_total_reweight_factor) > 5e-3) {
    cout << "Warning: Reweight factor not correctly calculated in event "
	 << _event_number << ": ";
    cout << org_ren_alps*org_em_alps*org_pdf_fact << " (cf. "
	 << _total_reweight_factor << ")" << endl;
  }

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
    if (DEBUG) cout << "New central alps factor: " << org_ren_alps << endl;
    
    double pdf_fact = 1;
    pdf_fact *= calculatePDFWeight(orgnum, sf, _beam[0], _pdf_pdg1, _pdf_x1, _pdf_q1);
    pdf_fact *= calculatePDFWeight(orgnum, sf, _beam[1], _pdf_pdg2, _pdf_x2, _pdf_q2);
    if (DEBUG) cout << "New PDF factor: " << pdf_fact << endl;

    _scaleweights.push_back(ren_alps*pdf_fact/(org_ren_alps*org_pdf_fact));
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
    _alpsweights.push_back(em_alps/org_em_alps);
    if (DEBUG) cout << "alpsfact weight: " << _alpsweights[_alpsweights.size()-1]
		    << endl;
  }

  // Different PDF sets
  for(int i=0; i < _PDFsets.size(); i++){
    int pdfnum = i+1;
    // Initialize a new vector for the values of the members of this PDF
    if (DEBUG) cout << "Reweighting with PDF set " << _PDFsets[i] << endl;
    vector<double>* pdffacts = new vector<double>;
    for(int j=0; j < _members[i]; j++) {
      if (DEBUG) cout << "PDF set member " << j << endl;      
      // Set PDF set member
      usePDFMember(pdfnum, j);
      // First recalculate alpha_s weights, since alpha_s differs between PDFs
      // Recalculate PDF weights
      double pdf_fact = 1;
      if(_n_qcd > 0) pdf_fact *= pow(alphasPDF(pdfnum, _ren_scale), _n_qcd);
      if (DEBUG) cout << "After PDF central alps factor: " << org_ren_alps << endl;
      for(int k=0; k < _alpsem_scales.size(); k++)
	pdf_fact *= alphasPDF(pdfnum, _alpsem_scales[k]);
      if (DEBUG) cout << "After PDF emission alps factor: " << org_em_alps << endl;
      pdf_fact *= calculatePDFWeight(pdfnum, 1., _beam[0], _pdf_pdg1, _pdf_x1, _pdf_q1);
      pdf_fact *= calculatePDFWeight(pdfnum, 1., _beam[1], _pdf_pdg2, _pdf_x2, _pdf_q2);
      if (DEBUG) cout << "Total PDF factor: " << pdf_fact << endl;
      pdffacts->push_back(pdf_fact/org_weight);
      if (DEBUG) cout << "PDF weight: " << (*pdffacts)[pdffacts->size()-1] << endl;
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

bool SystCalc::writeHeader(ostream& outfile)
{
  outfile << "<header>\n";
  outfile << "  <initrwgt>\n";
  if(_scalefacts.size() > 0) {
    outfile << "    <scale type=\"central\" entries=\"" << _scalefacts.size() 
	    << "\">";
    for (int i=0; i < _scalefacts.size(); i++) {
      if(i > 0) outfile << " ";
      outfile << _scalefacts[i]; }
    outfile << "</scale>\n";
  }
  if(_alpsfacts.size() > 0) {
    outfile << "    <scale type=\"emission\" entries=\"" << _alpsfacts.size() 
	    << "\">";
    for (int i=0; i < _alpsfacts.size(); i++) {
      if(i > 0) outfile << " ";
      outfile << _alpsfacts[i]; }
    outfile << "</scale>\n";
  }
  if(_matchscales.size() > 0) {
    outfile << "    <qmatch entries=\"" << _matchscales.size() << "\">";
    for (int i=0; i < _matchscales.size(); i++) {
      if(i > 0) outfile << " ";
      outfile << _matchscales[i]; }
    outfile << "</qmatch>\n";
  }
  if(_PDFsets.size() > 0){
    for (int i=0; i < _PDFsets.size(); i++)
      outfile << "    <pdf type=\"" << _PDFsets[i] << "\" entries=\"" << _members[i]
	      << "\" combine=\"" << _combinations[i] << "\"></pdf>\n";
  }
  outfile << "  </initrwgt>\n";
  outfile << "</header>\n";
  return true;
}

bool SystCalc::writeEvent(ostream& outfile)
{
  outfile << "<rwgt";
  if (_event_number > 0) outfile << " event=\"" << _event_number << "\"";
  outfile << ">\n";
  if(_scaleweights.size() > 0) {
    outfile << "  <scale type=\"central\">";
    for (int i=0; i < _scaleweights.size(); i++) {
      if(i > 0) outfile << " ";
      outfile << _scaleweights[i]; }
    outfile << "</scale>\n";
  }
  if(_alpsweights.size() > 0) {
    outfile << "  <scale type=\"emission\">";
    for (int i=0; i < _alpsweights.size(); i++) {
      if(i > 0) outfile << " ";
      outfile << _alpsweights[i]; }
    outfile << "</scale>\n";
  }
  if(_matchweights.size() > 0) {
    outfile << "  <qmatch>";
    for (int i=0; i < _matchweights.size(); i++) {
      if(i > 0) outfile << " ";
      outfile << _matchweights[i]; }
    outfile << "</qmatch>\n";
  }
  if(_PDFweights.size() > 0){
    for (int i=0; i < _PDFweights.size(); i++){
      outfile << "  <pdf type=\"" << _PDFsets[i] << "\">";
      for (int j=0; j < _PDFweights[i].size(); j++) {
	if(j > 0) outfile << " ";
	outfile << _PDFweights[i][j]; }
      outfile << "</pdf>\n";
    }
  }
  outfile << "</rwgt>\n";
  return true;
}
