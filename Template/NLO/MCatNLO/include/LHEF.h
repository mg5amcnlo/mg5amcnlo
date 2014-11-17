// -*- C++ -*-
#ifndef THEPEG_LHEF_H
#define THEPEG_LHEF_H
//
// This is the declaration of the Les Houches Event File classes.
//


#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <limits>

namespace LHEF {

/**
 * The XMLTag struct is used to represent all information within an
 * XML tag. It contains the attributes as a map, any sub-tags as a
 * vector of pointers to other XMLTag objects, and any other
 * information as a single string.
 */
struct XMLTag {

  /**
   * Convenient typdef.
   */
  typedef std::string::size_type pos_t;

  /**
   * Convenient alias for npos.
   */
  static const pos_t end = std::string::npos;
  

  /**
   * The destructor also destroys any sub-tags.
   */
  ~XMLTag() {
    for ( int i = 0, N = tags.size(); i < N; ++i ) delete tags[i];
  }

  /**
   * The name of this tag.
   */
  std::string name;

  /**
   * The attributes of this tag.
   */
  std::map<std::string,std::string> attr;

  /**
   * A vector of sub-tags.
   */
  std::vector<XMLTag*> tags;

  /**
   * The contents of this tag.
   */
  std::string contents;

  /**
   * Find an attribute named \a n and set the double variable \a v to
   * the corresponding value. @return false if no attribute was found.
   */
  bool getattr(std::string n, double & v) const {
    std::map<std::string,std::string>::const_iterator it = attr.find(n);
    if ( it == attr.end() ) return false;
    v = std::atof(it->second.c_str());
    return true;
  }

  /**
   * Find an attribute named \a n and set the bool variable \a v to
   * true if the corresponding value is "yes". @return false if no
   * attribute was found.
   */
  bool getattr(std::string n, bool & v) const {
    std::map<std::string,std::string>::const_iterator it = attr.find(n);
    if ( it == attr.end() ) return false;
    if ( it->second == "yes" ) v = true;
    return true;
  }

  /**
   * Find an attribute named \a n and set the long variable \a v to
   * the corresponding value. @return false if no attribute was found.
   */
  bool getattr(std::string n, long & v) const {
    std::map<std::string,std::string>::const_iterator it = attr.find(n);
    if ( it == attr.end() ) return false;
    v = std::atoi(it->second.c_str());
    return true;
  }

  /**
   * Find an attribute named \a n and set the long variable \a v to
   * the corresponding value. @return false if no attribute was found.
   */
  bool getattr(std::string n, int & v) const {
    std::map<std::string,std::string>::const_iterator it = attr.find(n);
    if ( it == attr.end() ) return false;
    v = int(std::atoi(it->second.c_str()));
    return true;
  }

  /**
   * Find an attribute named \a n and set the string variable \a v to
   * the corresponding value. @return false if no attribute was found.
   */
  bool getattr(std::string n, std::string & v) const {
    std::map<std::string,std::string>::const_iterator it = attr.find(n);
    if ( it == attr.end() ) return false;
    v = it->second;
    return true;
  }

  /**
   * Scan the given string and return all XML tags found as a vector
   * of pointers to XMLTag objects.
   */
  static std::vector<XMLTag*> findXMLTags(std::string str,
					  std::string * leftover = 0) {
    std::vector<XMLTag*> tags;
    pos_t curr = 0;

    while ( curr != end ) {

      // Find the first tag
      pos_t begin = str.find("<", curr);

      // Skip comments
      if ( str.find("<!--", curr) == begin ) {
	pos_t endcom = str.find("-->", begin);
	if ( endcom == end ) {
	  if ( leftover ) *leftover += str.substr(curr);
	  return tags;
	}
	if ( leftover ) *leftover += str.substr(curr, endcom - curr);
	curr = endcom;
	continue;
      }

      if ( leftover ) *leftover += str.substr(curr, begin - curr);
      if ( begin == end || begin > str.length() - 3 || str[begin + 1] == '/' )
	return tags; 

      pos_t close = str.find(">", curr);
      if ( close == end ) return tags;

      // find the tag name.
      curr = str.find_first_of(" \t\n/>", begin);
      tags.push_back(new XMLTag());
      tags.back()->name = str.substr(begin + 1, curr - begin - 1);

      while ( true ) {

	// Now skip some white space to see if we can find an attribute.
	curr = str.find_first_not_of(" \t\n", curr);
	if ( curr == end || curr >= close ) break;

	pos_t tend = str.find_first_of("= \t\n", curr);
	if ( tend == end || tend >= close ) break;

	std::string name = str.substr(curr, tend - curr);
	curr = str.find("=", curr) + 1;

	// OK now find the beginning and end of the atribute.
	curr = str.find("\"", curr);
	if ( curr == end || curr >= close ) break;
	pos_t bega = ++curr;
	curr = str.find("\"", curr);
	while ( curr != end && str[curr - 1] == '\\' )
	  curr = str.find("\"", curr + 1);

	std::string value = str.substr(bega, curr == end? end: curr - bega);

	tags.back()->attr[name] = value;

	++curr;

      }

      curr = close + 1;
      if ( str[close - 1] == '/' ) continue;

      pos_t endtag = str.find("</" + tags.back()->name + ">", curr);
      if ( endtag == end ) {
	tags.back()->contents = str.substr(curr);
	curr = endtag;
      } else {
	tags.back()->contents = str.substr(curr, endtag - curr);
	curr = endtag + tags.back()->name.length() + 3;
      }

      std::string leftovers;
      tags.back()->tags = findXMLTags(tags.back()->contents, &leftovers);
      if ( leftovers.find_first_not_of(" \t\n") == end ) leftovers="";
      tags.back()->contents = leftovers;

    }
    return tags;

  }

  /**
   * Print out this tag to a stream.
   */
  void print(std::ostream & os) const {
    os << "<" << name;
    for ( std::map<std::string,std::string>::const_iterator it = attr.begin();
	  it != attr.end(); ++it )
      os << " " << it->first << "=\"" << it->second << "\"";
    if ( contents.empty() && tags.empty() ) {
      os << "/>" << std::endl;
      return;
    }
    os << ">" << std::endl;
    for ( int i = 0, N = tags.size(); i < N; ++i )
      tags[i]->print(os);

    os << "````" << contents << "''''</" << name << ">" << std::endl;
  }

};


/**
 * The XSecInfo class contains information given in the xsecinfo tag.
 */
struct XSecInfo {

  /**
   * Intitialize default values.
   */
  XSecInfo(): neve(-1), maxweight(1.0), meanweight(1.0), negweights(false), 
	      varweights(false) {}

  /**
   * Create from XML tag
   */
  XSecInfo(const XMLTag & tag) : neve(-1), maxweight(1.0), meanweight(1.0),
				 negweights(false), varweights(false) {
    if ( !tag.getattr("neve", neve) ) 
      throw std::runtime_error("Found xsecinfo tag without neve attribute "
			       "in Les Houches Event File.");
    if ( !tag.getattr("totxsec", totxsec) ) 
      throw std::runtime_error("Found xsecinfo tag without totxsec "
			       "attribute in Les Houches Event File.");
    tag.getattr("maxweight", maxweight);
    tag.getattr("meanweight", meanweight);
    tag.getattr("negweights", negweights);
    tag.getattr("varweights", varweights);

  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {
    file << "<xsecinfo neve=\"" << neve << "\""
	 << " totxsec=\"" << totxsec << "\""
	 << " maxweight=\"" << maxweight << "\""
	 << " meanweight=\"" << meanweight << "\"";
    if ( negweights ) file << " negweights=\"yes\"";
    if ( varweights ) file << " varweights=\"yes\"";
    file << "/>" << std::endl;
  }

  /**
   * The number of events.
   */
  long neve;

  /**
   * The total cross section in pb.
   */
  double totxsec;

  /**
   * The maximum weight.
   */
  double maxweight;

  /**
   * The average weight.
   */
  double meanweight;

  /**
   * Does the file contain negative weights?
   */
  bool negweights;

  /**
   * Does the file contain varying weights?
   */
  bool varweights;

};

/**
 * The Cut class represents a cut used by the Matrix Element generator.
 */
struct Cut {

  /**
   * Intitialize default values.
   */
  Cut(): min(-0.99*std::numeric_limits<double>::max()),
	 max(0.99*std::numeric_limits<double>::max()) {}

  /**
   * Create from XML tag.
   */
  Cut(const XMLTag & tag,
      const std::map<std::string,std::set<long> >& ptypes):
    min(-0.99*std::numeric_limits<double>::max()),
    max(0.99*std::numeric_limits<double>::max()) {
    if ( !tag.getattr("type", type) )
      throw std::runtime_error("Found cut tag without type attribute "
			       "in Les Houches file");
    long tmp;
    if ( tag.getattr("p1", np1) ) {
      if ( ptypes.find(np1) != ptypes.end() )
	p1 =  ptypes.find(np1)->second;
      else {
	tag.getattr("p1", tmp);
	p1.insert(tmp);
	np1 = "";
      }
    }
    if ( tag.getattr("p2", np2) ) {
      if ( ptypes.find(np2) != ptypes.end() )
	p2 =  ptypes.find(np2)->second;
      else {
	tag.getattr("p2", tmp);
	p2.insert(tmp);
	np2 = "";
      }
    }

    std::istringstream iss(tag.contents);
    iss >> min;
    if ( iss >> max ) {
      if ( min >= max )
	min = -0.99*std::numeric_limits<double>::max();
    } else
      max = 0.99*std::numeric_limits<double>::max();
  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {
    file << "<cut type=\"" << type << "\"";
    if ( !np1.empty() ) file << " p1=\"" << np1 << "\"";
    else if ( p1.size() == 1 ) file << " p1=\"" << *p1.begin() << "\"";
    if ( !np2.empty() )  file << " p2=\"" << np2 << "\"";
    else if ( p2.size() == 1 ) file << " p2=\"" << *p2.begin() << "\"";
    file << ">";
    if ( min > -0.9*std::numeric_limits<double>::max() )
      file << min;
    else
      file << max;
    if ( max < 0.9*std::numeric_limits<double>::max() )
      file << " " << max;
    file << "</cut>" << std::endl;
  }

  /**
   * Check if a \a id1 matches p1 and \a id2 matches p2. Only non-zero
   * values are considered.
   */
  bool match(long id1, long id2 = 0) const {
    std::pair<bool,bool> ret(false, false);
    if ( !id2 ) ret.second = true;
    if ( !id1 ) ret.first = true;
    if ( p1.find(0) != p1.end() ) ret.first = true;
    if ( p1.find(id1) != p1.end() ) ret.first = true;
    if ( p2.find(0) != p2.end() ) ret.second = true;
    if ( p2.find(id2) != p2.end() ) ret.second = true;
    return ret.first && ret.second;
  }

  /**
   * Check if the particles given as a vector of PDG \a id numbers,
   * and a vector of vectors of momentum components, \a p, will pass
   * the cut defined in this event.
   */
  bool passCuts(const std::vector<long> & id,
		const std::vector< std::vector<double> >& p ) const {
    if ( ( type == "m" && !p2.size() ) || type == "kt" || type == "eta" ||
	 type == "y" || type == "E" ) {
      for ( int i = 0, N = id.size(); i < N; ++i )
	if ( match(id[i]) ) {
	  if ( type == "m" ) {
	    double v = p[i][4]*p[i][4] - p[i][3]*p[i][3] - p[i][2]*p[i][2]
	      - p[i][1]*p[i][1];
	    v = v >= 0.0? std::sqrt(v): -std::sqrt(-v);
	    if ( outside(v) ) return false;
	  }
	  else if ( type == "kt" ) {
	    if ( outside(std::sqrt(p[i][2]*p[i][2] + p[i][1]*p[i][1])) )
	      return false;
	  }
	  else if ( type == "E" ) {
	    if ( outside(p[i][4]) ) return false;
	  }
	  else if ( type == "eta" ) {
	    if ( outside(eta(p[i])) ) return false;
	  }
	  else if ( type == "y" ) {
	    if ( outside(rap(p[i])) ) return false;
	  }
	}
    }
    else if ( type == "m"  || type == "deltaR" ) {
      for ( int i = 1, N = id.size(); i < N; ++i )
	for ( int j = 0; j < i; ++j )
	  if ( match(id[i], id[j]) || match(id[j], id[i]) ) {
	    if ( type == "m" ) {
	      double v = (p[i][4] + p[j][4])*(p[i][4] + p[j][4])
		- (p[i][3] + p[j][3])*(p[i][3] + p[j][3])
		- (p[i][2] + p[j][2])*(p[i][2] + p[j][2])
		- (p[i][1] + p[j][1])*(p[i][1] + p[j][1]);
	      v = v >= 0.0? std::sqrt(v): -std::sqrt(-v);
	      if ( outside(v) ) return false;
	    }
	    else if ( type == "deltaR" ) {
	      if ( outside(deltaR(p[i], p[j])) ) return false;
	    }
	  }
    }
    else if ( type == "ETmiss" ) {
      double x = 0.0;
      double y = 0.0;
      for ( int i = 0, N = id.size(); i < N; ++i )
	if ( match(id[i]) && !match(0, id[i]) ) {
	  x += p[i][1];
	  y += p[i][2];
	}
      if ( outside(std::sqrt(x*x + y*y)) ) return false;
    }
    else if ( type == "HT" ) {
      double pt = 0.0;
      for ( int i = 0, N = id.size(); i < N; ++i )
	if ( match(id[i]) && !match(0, id[i]) )
	  pt += std::sqrt(p[i][1]*p[i][1] + p[i][2]*p[i][2]);
      if ( outside(pt) ) return false;
    }
    return true;
  }

  /**
   * Return the pseudorapidity of a particle with momentum \a p.
   */
  static double eta(const std::vector<double> & p) {
    double pt2 = p[2]*p[2] + p[1]*p[1];
    if ( pt2 != 0.0 ) {
      double dum = std::sqrt(pt2 + p[3]*p[3]) + p[3];
      if ( dum != 0.0 )
	return std::log(dum/std::sqrt(pt2));
    }
    return p[3] < 0.0? -std::numeric_limits<double>::max():
      std::numeric_limits<double>::max();
  }
    
  /**
   * Return the true rapidity of a particle with momentum \a p.
   */
  static double rap(const std::vector<double> & p) {
    double pt2 = p[5]*p[5] + p[2]*p[2] + p[1]*p[1];
    if ( pt2 != 0.0 ) {
      double dum = std::sqrt(pt2 + p[3]*p[3]) + p[3];
      if ( dum != 0.0 )
	return std::log(dum/std::sqrt(pt2));
    }
    return p[3] < 0.0? -std::numeric_limits<double>::max():
      std::numeric_limits<double>::max();
  }
    
  /**
   * Return the delta-R of a particle pair with momenta \a p1 and \a p2.
   */
  static double deltaR(const std::vector<double> & p1,
		       const std::vector<double> & p2) {
    double deta = eta(p1) - eta(p2);
    double dphi = std::atan2(p1[1], p1[2]) - std::atan2(p2[1], p2[2]);
    if ( dphi > M_PI ) dphi -= 2.0*M_PI;
    if ( dphi < -M_PI ) dphi += 2.0*M_PI;
    return std::sqrt(dphi*dphi + deta*deta);
  }

  /**
   * Return true if the given \a value is outside limits.
   */
  bool outside(double value) const {
    return value < min || value >= max;
  }

  /**
   * The variable in which to cut.
   */
  std::string type;

  /**
   * The first types particle types for which this cut applies.
   */
  std::set<long> p1;

  /**
   * Symbolic name for p1.
   */
  std::string np1;

  /**
   * The second type of particles for which this cut applies.
   */
  std::set<long> p2;

  /**
   * Symbolic name for p1.
   */
  std::string np2;

  /**
   * The minimum value of the variable
   */
  double min;
  /**
   * The maximum value of the variable
   */
  double max;

};

/**
 * The ProcInfo class represents the information in a procinfo tag.
 */
struct ProcInfo {

  /**
   * Intitialize default values.
   */
  ProcInfo(): iproc(0), loops(0), qcdorder(-1), eworder(-1) {}

  /**
   * Create from XML tag.
   */
  ProcInfo(const XMLTag & tag): iproc(0), loops(0), qcdorder(-1), eworder(-1) {
    tag.getattr("iproc", iproc);
    tag.getattr("loops", loops);
    tag.getattr("qcdorder", qcdorder);
    tag.getattr("eworder", eworder);
    tag.getattr("rscheme", rscheme);
    tag.getattr("fscheme", fscheme);
    tag.getattr("scheme", scheme);
    description = tag.contents;
  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {
    file << "<procinfo iproc=\"" << iproc << "\"";
    if ( loops >= 0 ) file << " loops=\"" << loops  << "\"";
    if ( qcdorder >= 0 ) file  << " qcdorder=\"" << qcdorder  << "\"";
    if ( eworder >= 0 )	file << " eworder=\"" << eworder  << "\"";
    if ( !rscheme.empty() ) file << " rscheme=\"" << rscheme  << "\"";
    if ( !fscheme.empty() ) file << " fscheme=\"" << fscheme  << "\"";
    if ( !scheme.empty() ) file << " scheme=\"" << scheme  << "\"";
    if ( description.empty() )
      file << "/>" << std::endl;
    else
      file << ">" << description << "<procinfo>" << std::endl;
  }

  /**
   * The id number for the process.
   */
  int iproc;

  /**
   * The number of loops
   */
  int loops;

  /**
   * The number of QCD vertices.
   */
  int qcdorder;

  /**
   * The number of electro-weak vertices.
   */
  int eworder;

  /**
   * The factorization scheme used.
   */
  std::string fscheme;

  /**
   * The renormalization scheme used.
   */
  std::string rscheme;

  /**
   * The NLO scheme used.
   */
  std::string scheme;

  /**
   * Description of the process.
   */
  std::string description;

};

/**
 * The MergeInfo class represents the information in a mergeingo tag.
 */
struct MergeInfo {

  /**
   * Intitialize default values.
   */
  MergeInfo(): iproc(0), mergingscale(0.0), maxmult(false) {}

  /**
   * Creat from XML tag.
   */
  MergeInfo(const XMLTag & tag): iproc(0), mergingscale(0.0), maxmult(false) {
    tag.getattr("iproc", iproc);
    tag.getattr("mergingscale", mergingscale);
    tag.getattr("maxmult", maxmult);
    scheme = tag.contents;
  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {
    file << "<mergeinfo iproc=\"" << iproc << "\"";
    if ( mergingscale > 0.0 )
      file << " mergingscale=\"" << mergingscale  << "\"";
    if ( maxmult ) file << " maxmult=\"yes\"";
    file << ">" << scheme << "</mergeinfo>" << std::endl;
  }

  /**
   * The id number for the process.
   */
  int iproc;

  /**
   * The sceme used to reweight events.
   */
  std::string scheme;

  /**
   * The merging scale used if different from the cut definitions.
   */
  double mergingscale;

  /**
   * Is this event reweighted as if it was the maximum multiplicity.
   */
  bool maxmult;

};

/**
 * The WeightInfo class encodes the description of a given weight
 * present for all events.
 */
struct WeightInfo {

  /**
   * Constructors
   */
  WeightInfo(): inGroup(-1), muf(1.0), mur(1.0), pdf(0), pdf2(0) {}

  /**
   * Construct from the XML tag
   */
  WeightInfo(const XMLTag & tag): inGroup(-1), muf(1.0), mur(1.0), pdf(0) {
    tag.getattr("name", name);
    tag.getattr("mur", mur);
    tag.getattr("muf", muf);
    tag.getattr("pdf", pdf);
    tag.getattr("pdf2", pdf2);
  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {

    file << "<weightinfo name=\"" << name << "\"";
    if ( mur != 1.0 ) file << " mur=\"" << mur << "\"";
    if ( muf != 1.0 ) file << " muf=\"" << muf << "\"";
    if ( pdf != 1.0 ) file << " pdf=\"" << pdf << "\"";
    if ( pdf2 != 1.0 ) file << " pdf2=\"" << pdf2 << "\"";
    file << " />" << std::endl;
  }

  /**
   * If inside a group, this is the index of that group.
   */
  int inGroup;

  /**
   * The name.
   */
  std::string name;

  /**
   * Factor multiplying the nominal factorization scale for this weight.
   */
  double muf;

  /**
   * Factor multiplying the nominal renormalization scale for this weight.
   */
  double mur;

  /**
   * The LHAPDF set relevant for this weight
   */
  long pdf;

  /**
   * The LHAPDF set for the second beam relevant for this weight if different from pdf.
   */
  long pdf2;

};

/**
 * The WeightGroup assigns a group-name to a set of WeightInfo objects.
 */
struct WeightGroup {

  /**
   * Default constructor;
   */
  WeightGroup() {}

  /**
   * Construct a group of WeightInfo objects from an XML tag and
   * insert them in the given vector.
   */
  WeightGroup(const XMLTag & tag, int groupIndex, std::vector<WeightInfo> & wiv) {
    tag.getattr("name", name);
    for ( int i = 0, N = tag.tags.size(); i < N; ++i ) {
      WeightInfo wi(*tag.tags[i]);
      wi.inGroup = groupIndex;
      wiv.push_back(wi);
    }
  }

  /**
   * The name.
   */
  std::string name;

};


/**
 * The Weight class represents the information in a weight tag.
 */
struct Weight {

  /**
   * Initialize default values.
   */
  Weight(): born(0.0), sudakov(0.0) {}

  /**
   * Create from XML tag
   */
  Weight(const XMLTag & tag): born(0.0), sudakov(0.0) {
    tag.getattr("name", name);
    tag.getattr("born", born);
    tag.getattr("sudakov", sudakov);
    std::istringstream iss(tag.contents);
    double w;
    while ( iss >> w ) weights.push_back(w);
  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {
    file << "<weight";
    if ( !name.empty() ) file << " name=\"" << name << "\"";
    if ( born != 0.0 ) file << " born=\"" << born << "\"";
    if ( sudakov != 0.0 ) file << " sudakov=\"" << sudakov << "\"";
    file << ">";
    for ( int j = 0, M = weights.size(); j < M; ++j ) file << " " << weights[j];
    file << "</weight>" << std::endl;
  }

  /**
   * The identifyer for this set of weights.
   */
  std::string name;

  /**
   * The relative size of the born cross section of this event.
   */
  double born;

  /**
   * The relative size of the sudakov applied to this event.
   */
  double sudakov;

  /**
   * The weights of this event.
   */
  std::vector<double> weights;

};

/**
 * The Clus class represents a clustering of two particle entries into
 * one as defined in a clustering tag.
 */
struct Clus {

  /**
   * Initialize default values.
   */
  Clus(): scale(-1.0), alphas(-1.0) {}

  /**
   * Initialize default values.
   */
  Clus(const XMLTag & tag): scale(-1.0), alphas(-1.0) {
    tag.getattr("scale", scale);
    tag.getattr("alphas", alphas);
    std::istringstream iss(tag.contents);
    iss >> p1 >> p2;
    if ( !( iss >> p0 ) ) p0 = p1;
  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {
    file << "<clus";
    if ( scale > 0.0 ) file << " scale=\"" << scale << "\"";
    if ( alphas > 0.0 ) file << " alphas\"" << alphas << "\"";
    file << ">" << p1 << " " << p2;
    if ( p1 != p0 ) file << " " << p0;
    file << "</clus>" << std::endl;
  }

  /**
   * The first particle entry that has been clustered.
   */
  int p1;

  /**
   * The second particle entry that has been clustered.
   */
  int p2;

  /**
   * The particle entry corresponding to the clustered particles.
   */
  int p0;

  /**
   * The scale in GeV associated with the clustering.
   */
  double scale;

  /**
   * The alpha_s used in the corresponding vertex, if this was used in
   * the cross section.
   */
  double alphas;

};

/**
 * Collect different scales relevant for an event.
 */
struct Scales {

  /**
   * Empty constructor.
   */
  Scales(double defscale = -1.0)
  : muf(defscale), mur(defscale), mups(defscale), SCALUP(defscale) {}

  /**
   * Construct from an XML-tag
   */
  Scales(const XMLTag & tag, double defscale = -1.0)
    : muf(defscale), mur(defscale), mups(defscale), SCALUP(defscale) {
    for ( std::map<std::string,std::string>::const_iterator it = tag.attr.begin();
	  it != tag.attr.end(); ++it ) {
      double v = std::atof(it->second.c_str());
      if ( it->first == "muf" ) muf = v;
      else if ( it->first == "mur" ) mur = v;
      else if ( it->first == "mups" ) mups = v;
      else mu[it->first] = v;
    }
  }

  /**
   * Print out the corresponding XML-tag.
   */
  void print(std::ostream & file) const {
    if ( muf == SCALUP && mur == SCALUP && mups == SCALUP && mu.size() == 0 ) return;
    file << "<scales";
    if ( muf != SCALUP ) file << " muf=\"" << muf << "\"";
    if ( mur != SCALUP ) file << " mur=\"" << mur << "\"";
    if ( mups != SCALUP ) file << " mups=\"" << mups << "\"";
    for ( std::map<std::string,double>::const_iterator it = mu.begin();
	  it != mu.end(); ++it )
      file << " " << it->first << "=\"" << it->second << "\"";
    file << " />" << std::endl;
  }

  /**
   * The factorization scale used for this event.
   */
  double muf;

  /**
   * The renormalization scale used for this event.
   */
  double mur;

  /**
   * The starting scale for the parton shower as suggested by the
   * matrix element generator.
   */
  double mups;

  /**
   * Any other scales reported by the matrix element generator.
   */
  std::map<std::string,double> mu;

  /**
   * The default scale in this event.
   */
  double SCALUP;

};

/**
 * The PDFInfo class represents the information in a pdfinto tag.
 */
struct PDFInfo {

  /**
   * Initialize default values.
   */
  PDFInfo(double defscale = -1.0): p1(0), p2(0), x1(-1.0), x2(-1.0),
	     xf1(-1.0),  xf2(-1.0), scale(defscale), SCALUP(defscale) {}

  /**
   * Create from XML tag.
   */
  PDFInfo(const XMLTag & tag, double defscale = -1.0)
    : p1(0), p2(0), x1(-1.0), x2(-1.0), xf1(-1.0),  xf2(-1.0),
      scale(defscale), SCALUP(defscale) {
    tag.getattr("scale", scale);
    tag.getattr("p1", p1);
    tag.getattr("p2", p2);
    tag.getattr("x1", x1);
    tag.getattr("x2", x2);
    std::istringstream iss(tag.contents);
    iss >> xf1 >> xf2;

  }

  /**
   * Print out an XML tag.
   */
  void print(std::ostream & file) const {
    if ( xf1 <= 0 ) return;
    file << "<pdfinfo";
    if ( p1 != 0 ) file << " p1=\"" << p1 << "\"";
    if ( p2 != 0 ) file << " p2=\"" << p2 << "\"";
	if ( x1 > 0 ) file << " x1=\"" << x1 << "\"";
	if ( x2 > 0 ) file << " x2=\"" << x2 << "\"";
	if ( scale != SCALUP ) file << " scale=\"" << scale << "\"";
	file << ">" << xf1 << " " << xf2 << "</pdfinfo>" << std::endl;
  }

  /**
   * The type of the incoming particle 1.
   */
  long p1;

  /**
   * The type of the incoming particle 2.
   */
  long p2;

  /**
   * The x-value used for the incoming particle 1.
   */
  double x1;

  /**
   * The x-value used for the incoming particle 2.
   */
  double x2;

  /**
   * The value of the pdf for the incoming particle 1.
   */
  double xf1;

  /**
   * The value of the pdf for the incoming particle 2.
   */
  double xf2;

  /**
   * The scale used in the PDF:s
   */
  double scale;

  /**
   * THe default scale in the event.
   */
  double SCALUP;

};

/**
 * The HEPRUP class is a simple container corresponding to the Les Houches
 * accord (<A HREF="http://arxiv.org/abs/hep-ph/0109068">hep-ph/0109068</A>)
 * common block with the same name. The members are named in the same
 * way as in the common block. However, fortran arrays are represented
 * by vectors, except for the arrays of length two which are
 * represented by pair objects.
 */
class HEPRUP {

public:

  /** @name Standard constructors and destructors. */
  //@{
  /**
   * Default constructor.
   */
  HEPRUP()
    : IDWTUP(0), NPRUP(0) {}

  /**
   * Assignment operator.
   */
  HEPRUP & operator=(const HEPRUP & x) {
    IDBMUP = x.IDBMUP;
    EBMUP = x.EBMUP;
    PDFGUP = x.PDFGUP;
    PDFSUP = x.PDFSUP;
    IDWTUP = x.IDWTUP;
    NPRUP = x.NPRUP;
    XSECUP = x.XSECUP;
    XERRUP = x.XERRUP;
    XMAXUP = x.XMAXUP;
    LPRUP = x.LPRUP;
    xsecinfo = x.xsecinfo;
    cuts = x.cuts;
    ptypes = x.ptypes;
    procinfo = x.procinfo;
    mergeinfo = x.mergeinfo;
    generators = x.generators;

    return *this;
  }

  /**
   * Destructor.
   */
  ~HEPRUP() {}
  //@}

public:

  /**
   * Set the NPRUP variable, corresponding to the number of
   * sub-processes, to \a nrup, and resize all relevant vectors
   * accordingly.
   */
  void resize(int nrup) {
    NPRUP = nrup;
    resize();
  }

  /**
   * Assuming the NPRUP variable, corresponding to the number of
   * sub-processes, is correctly set, resize the relevant vectors
   * accordingly.
   */
  void resize() {
    XSECUP.resize(NPRUP);
    XERRUP.resize(NPRUP);
    XMAXUP.resize(NPRUP);
    LPRUP.resize(NPRUP);
  }

public:

  /**
   * PDG id's of beam particles. (first/second is in +/-z direction).
   */
  std::pair<long,long> IDBMUP;

  /**
   * Energy of beam particles given in GeV.
   */
  std::pair<double,double> EBMUP;

  /**
   * The author group for the PDF used for the beams according to the
   * PDFLib specification.
   */
  std::pair<int,int> PDFGUP;

  /**
   * The id number the PDF used for the beams according to the
   * PDFLib specification.
   */
  std::pair<int,int> PDFSUP;

  /**
   * Master switch indicating how the ME generator envisages the
   * events weights should be interpreted according to the Les Houches
   * accord.
   */
  int IDWTUP;

  /**
   * The number of different subprocesses in this file.
   */
  int NPRUP;

  /**
   * The cross sections for the different subprocesses in pb.
   */
  std::vector<double> XSECUP;

  /**
   * The statistical error in the cross sections for the different
   * subprocesses in pb.
   */
  std::vector<double> XERRUP;

  /**
   * The maximum event weights (in HEPEUP::XWGTUP) for different
   * subprocesses.
   */
  std::vector<double> XMAXUP;

  /**
   * The subprocess code for the different subprocesses.
   */
  std::vector<int> LPRUP;

  /**
   * Contents of the xsecinfo tag
   */
  XSecInfo xsecinfo;

  /**
   * Contents of the cuts tag.
   */
  std::vector<Cut> cuts;

  /**
   * A map of codes for different particle types.
   */
  std::map<std::string, std::set<long> > ptypes;

  /**
   * Contents of the procinfo tags
   */
  std::map<long,ProcInfo> procinfo;

  /**
   * Contents of the mergeinfo tags
   */
  std::map<long,MergeInfo> mergeinfo;

  /**
   * The names of the programs and their version information used to
   * create this file.
   */
  std::vector< std::pair<std::string,std::string> > generators;

  /**
   * The vector of WeightInfo objects for this file.
   */
  std::vector<WeightInfo> weightinfo;

  /**
   * A map relating names of weights to indices of the weightinfo vector.
   */
  std::map<std::string,int> weightmap;

  /**
   * The vector of WeightGroup objects in this file.
   */
  std::vector<WeightGroup> weightgroup;

};

class HEPEUP;

/**
 * The EventGroup represents a set of events which are to be
 * considered together.
 */
struct EventGroup: public std::vector<HEPEUP*> {

  /**
   * Initialize default values.
   */
  EventGroup(): nreal(-1), ncounter(-1) {}

  /**
   * The copy constructor also copies the included HEPEUP object.
   */
  EventGroup(const EventGroup &);

  /**
   * The assignment also copies the included HEPEUP object.
   */
  EventGroup & operator=(const EventGroup &);

  /**
   * Remove all subevents.
   */
  void clear();

  /**
   * The destructor deletes the included HEPEUP objects.
   */
  ~EventGroup();

  /**
   * The number of real events in this event group.
   */
  int nreal;

  /**
   * The number of counter events in this event group.
   */
  int ncounter;

};


/**
 * The HEPEUP class is a simple container corresponding to the Les Houches accord
 * (<A HREF="http://arxiv.org/abs/hep-ph/0109068">hep-ph/0109068</A>)
 * common block with the same name. The members are named in the same
 * way as in the common block. However, fortran arrays are represented
 * by vectors, except for the arrays of length two which are
 * represented by pair objects.
 */
class HEPEUP {

public:

  /** @name Standard constructors and destructors. */
  //@{
  /**
   * Default constructor.
   */
  HEPEUP()
    : NUP(0), IDPRUP(0), XWGTUP(0.0), XPDWUP(0.0, 0.0),
      SCALUP(0.0), AQEDUP(0.0), AQCDUP(0.0), heprup(0), currentWeight(0) {}

  /**
   * Copy constructor
   */
  HEPEUP(const HEPEUP & x) {
    operator=(x);
  }

  /**
   * Copy information from the given HEPEUP. Sub event information is
   * left untouched.
   */
  HEPEUP & setEvent(const HEPEUP & x) {
    NUP = x.NUP;
    IDPRUP = x.IDPRUP;
    XWGTUP = x.XWGTUP;
    XPDWUP = x.XPDWUP;
    SCALUP = x.SCALUP;
    AQEDUP = x.AQEDUP;
    AQCDUP = x.AQCDUP;
    IDUP = x.IDUP;
    ISTUP = x.ISTUP;
    MOTHUP = x.MOTHUP;
    ICOLUP = x.ICOLUP;
    PUP = x.PUP;
    VTIMUP = x.VTIMUP;
    SPINUP = x.SPINUP;
    heprup = x.heprup;
    oldweights = x.oldweights;
    weights = x.weights;
    pdfinfo = x.pdfinfo;
    PDFGUPsave = x.PDFGUPsave;
    PDFSUPsave = x.PDFSUPsave;
    clustering = x.clustering;
    scales = x.scales;
    currentWeight = x.currentWeight;
    return *this;
  }

  /**
   * Assignment operator.
   */
  HEPEUP & operator=(const HEPEUP & x) {
    clear();
    setEvent(x);
    subevents = x.subevents;
    return *this;
  }


  /**
   * Destructor.
   */
  ~HEPEUP() {
    clear();
  };
  //@}

public:

  /**
   * Reset the HEPEUP object (does not touch the sub events).
   */
  void reset() {
    setWeightInfo(0);
    NUP = 0;
    clustering.clear();
    weights.clear();
  }

  /**
   * Clear the HEPEUP object.
   */
  void clear() {
    reset();
    subevents.clear();
  }

  /**
   * Set the NUP variable, corresponding to the number of particles in
   * the current event, to \a nup, and resize all relevant vectors
   * accordingly.
   */
  void resize(int nup) {
    NUP = nup;
    resize();
  }

  /**
   * Return the main weight for this event.
   */
  double weight() const {
    if ( !subevents.empty() ) {
      double w = 0.0;
      for ( int i = 0, N = subevents.size(); i < N; ++i )
	w += subevents[i]->weight();
      return w;
    }
    if ( oldweights.empty() || oldweights[0].weights.empty() )
      return XWGTUP;
    else
      return oldweights[0].weights[0];
  }

  /**
   * Assuming the NUP variable, corresponding to the number of
   * particles in the current event, is correctly set, resize the
   * relevant vectors accordingly.
   */
  void resize() {
    IDUP.resize(NUP);
    ISTUP.resize(NUP);
    MOTHUP.resize(NUP);
    ICOLUP.resize(NUP);
    PUP.resize(NUP, std::vector<double>(5));
    VTIMUP.resize(NUP);
    SPINUP.resize(NUP);
  }

  /**
   * Setup the current event to use weight i. If zero, the default
   * weight will be used.
   */
  bool setWeightInfo(unsigned int i) {
    if ( i >= weights.size() ) return false;
    if ( currentWeight ) {
      scales.mur /= currentWeight->mur;
      scales.muf /= currentWeight->muf;
      heprup->PDFGUP = PDFGUPsave;
      heprup->PDFSUP = PDFSUPsave;
    }
    XWGTUP = weights[i].first;
    currentWeight = weights[i].second;
    if ( currentWeight ) {
      scales.mur *= currentWeight->mur;
      scales.muf *= currentWeight->muf;
      PDFGUPsave = heprup->PDFGUP;
      PDFSUPsave = heprup->PDFSUP;
      if ( currentWeight->pdf ) {
	heprup->PDFGUP.first =  heprup->PDFGUP.second = 0;
	heprup->PDFSUP.first =  heprup->PDFSUP.second = currentWeight->pdf;
      }
      if ( currentWeight->pdf2 ) {
	heprup->PDFSUP.second = currentWeight->pdf2;
      }

    }
    return true;
  }

  /**
   * Setup the current event to use sub event i. If zero, no sub event
   * will be chsen.
   */
  bool setSubEvent(unsigned int i) {
    if ( i > subevents.size() || subevents.empty() ) return false;
    if ( i == 0 ) {
      reset();
      weights = subevents[0]->weights;
      for ( int i = 1, N = subevents.size(); i < N; ++i )
	for ( int j = 0, M = weights.size(); j < M; ++j )
	  weights[j].first += subevents[i]->weights[j].first;
      currentWeight = 0;
    } else {
      setEvent(*subevents[i - 1]);
    }
    return true;
  }

public:

  /**
   * The number of particle entries in the current event.
   */
  int NUP;

  /**
   * The subprocess code for this event (as given in LPRUP).
   */
  int IDPRUP;

  /**
   * The weight for this event.
   */
  double XWGTUP;

  /**
   * The PDF weights for the two incoming partons. Note that this
   * variable is not present in the current LesHouches accord
   * (<A HREF="http://arxiv.org/abs/hep-ph/0109068">hep-ph/0109068</A>),
   * hopefully it will be present in a future accord.
   */
  std::pair<double,double> XPDWUP;

  /**
   * The scale in GeV used in the calculation of the PDF's in this
   * event.
   */
  double SCALUP;

  /**
   * The value of the QED coupling used in this event.
   */
  double AQEDUP;

  /**
   * The value of the QCD coupling used in this event.
   */
  double AQCDUP;

  /**
   * The PDG id's for the particle entries in this event.
   */
  std::vector<long> IDUP;

  /**
   * The status codes for the particle entries in this event.
   */
  std::vector<int> ISTUP;

  /**
   * Indices for the first and last mother for the particle entries in
   * this event.
   */
  std::vector< std::pair<int,int> > MOTHUP;

  /**
   * The colour-line indices (first(second) is (anti)colour) for the
   * particle entries in this event.
   */
  std::vector< std::pair<int,int> > ICOLUP;

  /**
   * Lab frame momentum (Px, Py, Pz, E and M in GeV) for the particle
   * entries in this event.
   */
  std::vector< std::vector<double> > PUP;

  /**
   * Invariant lifetime (c*tau, distance from production to decay in
   * mm) for the particle entries in this event.
   */
  std::vector<double> VTIMUP;

  /**
   * Spin info for the particle entries in this event given as the
   * cosine of the angle between the spin vector of a particle and the
   * 3-momentum of the decaying particle, specified in the lab frame.
   */
  std::vector<double> SPINUP;

  /**
   * A pointer to the current HEPRUP object.
   */
  HEPRUP * heprup;

  /**
   * The current weight info object.
   */
  const WeightInfo * currentWeight;

  /**
   * The weights associated with this event
   */
  std::vector<Weight> oldweights;

  /**
   * The weights for this event and their corresponding WeightInfo object.
   */
  std::vector< std::pair<double, const WeightInfo *> > weights;

  /**
   * Contents of the clustering tag.
   */
  std::vector<Clus> clustering;

  /**
   * Contents of the pdfinfo tag.
   */
  PDFInfo pdfinfo;

  /**
   * Saved information about pdfs if different in a selected weight.
   */
  std::pair<int,int> PDFGUPsave;

  /**
   * Saved information about pdfs if different in a selected weight.
   */
  std::pair<int,int> PDFSUPsave;
  

  /**
   * Contents of the scales tag
   */
  Scales scales;

  /**
   * If this is not a single event, but an event group, the events
   * included in the group are in this vector;
   */
  EventGroup subevents;

};


// Destructor implemented here.

void EventGroup::clear() {
  while ( size() > 0 ) {
    delete back();
    pop_back();
  }
}

EventGroup::~EventGroup() {
  clear();
}

EventGroup::EventGroup(const EventGroup &) {
  for ( int i = 0, N = size(); i < N; ++i ) at(i) = new HEPEUP(*at(i));
}

EventGroup & EventGroup::operator=(const EventGroup & x) {
  if ( &x == this ) return *this;
  clear();
  nreal = x.nreal;
  ncounter = x.ncounter;
  for ( int i = 0, N = x.size(); i < N; ++i ) push_back(new HEPEUP(*x.at(i)));
  return *this;
}


/**
 * The Reader class is initialized with a stream from which to read a
 * version 1/2 Les Houches Accord event file. In the constructor of
 * the Reader object the optional header information is read and then
 * the mandatory init is read. After this the whole header block
 * including the enclosing lines with tags are available in the public
 * headerBlock member variable. Also the information from the init
 * block is available in the heprup member variable and any additional
 * comment lines are available in initComments. After each successful
 * call to the readEvent() function the standard Les Houches Accord
 * information about the event is available in the hepeup member
 * variable and any additional comments in the eventComments
 * variable. A typical reading sequence would look as follows:
 *
 *
 */
class Reader {

public:

  /**
   * Initialize the Reader with a stream from which to read an event
   * file. After the constructor is called the whole header block
   * including the enclosing lines with tags are available in the
   * public headerBlock member variable. Also the information from the
   * init block is available in the heprup member variable and any
   * additional comment lines are available in initComments.
   *
   * @param is the stream to read from.
   */
  Reader(std::istream & is)
    : file(is) {
    init();
  }

  /**
   * Initialize the Reader with a filename from which to read an event
   * file. After the constructor is called the whole header block
   * including the enclosing lines with tags are available in the
   * public headerBlock member variable. Also the information from the
   * init block is available in the heprup member variable and any
   * additional comment lines are available in initComments.
   *
   * @param filename the name of the file to read from.
   */
  Reader(std::string filename)
    : intstream(filename.c_str()), file(intstream) {
    init();
  }

private:

  /**
   * Used internally in the constructors to read header and init
   * blocks.
   */
  void init() {

    bool readingHeader = false;
    bool readingInit = false;

    // Make sure we are reading a LHEF file:
    getline();
    if ( currentLine.find("<LesHouchesEvents" ) == std::string::npos )
      throw std::runtime_error
	("Tried to read a file which does not start with the "
	 "LesHouchesEvents tag.");
    version = 1;
    if ( currentLine.find("version=\"2" ) != std::string::npos )
      version = 2;
    //else if ( currentLine.find("version=\"1" ) == std::string::npos )
    //  throw std::runtime_error
    //("Tried to read a LesHouchesEvents file which is not version 1.0.");

    // Loop over all lines until we hit the </init> tag.
    while ( getline() && currentLine.find("</init>") == std::string::npos ) {
      if ( currentLine.find("<header") != std::string::npos ) {
	// We have hit the header block, so we should dump this and
	// all following lines to headerBlock until we hit the end of
	// it.
	readingHeader = true;
	headerBlock = currentLine + "\n";
      }
      else if ( currentLine.find("<init>") != std::string::npos ) {
	// We have hit the init block, so we should expect to find the
	// standard information in the following.
	readingInit = true;

	// The first line tells us how many lines to read next.
	getline();
	std::istringstream iss(currentLine);
	if ( !( iss >> heprup.IDBMUP.first >> heprup.IDBMUP.second
		    >> heprup.EBMUP.first >> heprup.EBMUP.second
	            >> heprup.PDFGUP.first >> heprup.PDFGUP.second
	            >> heprup.PDFSUP.first >> heprup.PDFSUP.second
		    >> heprup.IDWTUP >> heprup.NPRUP ) ) {
	  heprup.NPRUP = -42;
	  return;
	}
	heprup.resize();

	for ( int i = 0; i < heprup.NPRUP; ++i ) {
	  getline();
	  std::istringstream iss(currentLine);
	  if ( !( iss >> heprup.XSECUP[i] >> heprup.XERRUP[i]
	              >> heprup.XMAXUP[i] >> heprup.LPRUP[i] ) ) {
	    heprup.NPRUP = -42;
	    return;
	  }
	}
      }
      else if ( currentLine.find("</header>") != std::string::npos ) {
	// The end of the header block. Dump this line as well to the
	// headerBlock and we're done.
	readingHeader = false;
	headerBlock += currentLine + "\n";
      }
      else if ( readingHeader ) {
	// We are in the process of reading the header block. Dump the
	// line to haderBlock.
	headerBlock += currentLine + "\n";
      }
      else if ( readingInit ) {
	// Here we found a comment line. Dump it to initComments.
	initComments += currentLine + "\n";
      }
      else {
	// We found some other stuff outside the standard tags.
	outsideBlock += currentLine + "\n";
      }
    }
    if ( !file ) heprup.NPRUP = -42;
    if ( version < 2 ) return;

    heprup.procinfo.clear();
    heprup.mergeinfo.clear();
    // Scan the init block for XML tags
    std::string leftovers;
    std::vector<XMLTag*> tags = XMLTag::findXMLTags(initComments, &leftovers);
    if ( leftovers.find_first_not_of(" \t\n") == std::string::npos )
      leftovers="";
    initComments = leftovers;

    for ( int i = 0, N = tags.size(); i < N; ++i ) {
      const XMLTag & tag = *tags[i];

      if ( tag.name == "weightinfo" ) {
	heprup.weightinfo.push_back(WeightInfo(tag));
      }
      if ( tag.name == "weightgroup" ) {
	heprup.weightgroup.push_back(WeightGroup(tag, heprup.weightgroup.size(),
						 heprup.weightinfo));
      }
      if ( tag.name == "xsecinfo" ) {
	heprup.xsecinfo = XSecInfo(tag);
      }
      if ( tag.name == "generator" ) {
	std::string vers;
	tag.getattr("version", vers);
	heprup.generators.push_back(make_pair(tag.contents, vers));
      }
      else if ( tag.name == "cutsinfo" ) {
	heprup.cuts.clear();
	heprup.ptypes.clear();
	for ( int j = 0, M = tag.tags.size(); j < M; ++j ) {
	  XMLTag & ctag = *tag.tags[j];

	  if ( ctag.name == "ptype" ) {
	    std::string tname = ctag.attr["name"];
	    long id;
	    std::istringstream iss(ctag.contents);
	    while ( iss >> id ) heprup.ptypes[tname].insert(id);
	  }
	  else if ( ctag.name == "cut" )
	    heprup.cuts.push_back(Cut(ctag, heprup.ptypes));
	}
      }
      else if ( tag.name == "procinfo" ) {
	ProcInfo proc(tag);
	heprup.procinfo[proc.iproc] = proc;
      }
      else if ( tag.name == "mergeinfo" ) {
	MergeInfo merge(tag);
	heprup.mergeinfo[merge.iproc] = merge;
      }

    }

    for ( int i = 0, N = heprup.weightinfo.size(); i < N; ++i )
      heprup.weightmap[heprup.weightinfo[i].name] = i;

    if ( heprup.xsecinfo.neve < 0 )
      throw
	std::runtime_error("Found Les Houches event file without xsecinfo tag.");

  }

public:

  /**
   * Read an event from the file and store it in the hepeup
   * object. Optional comment lines are stored i the eventComments
   * member variable.
   * @return true if the read sas successful.
   */
  bool readEvent(HEPEUP * peup = 0) {

    HEPEUP & eup = (peup? *peup: hepeup);
    eup.clear();
    eup.heprup = &heprup;

    // Check if the initialization was successful. Otherwise we will
    // not read any events.
    if ( heprup.NPRUP < 0 ) return false;
    eventComments = "";
    outsideBlock = "";
    eup.NUP = 0;

    // Keep reading lines until we hit the next event or the end of
    // the event block. Save any inbetween lines. Exit if we didn't
    // find an event.
    while ( getline() && currentLine.find("<event") == std::string::npos &&
      currentLine.find("</eventgroup>") == std::string::npos )
      outsideBlock += currentLine + "\n";

    if ( currentLine.find("<eventgroup") != std::string::npos ) {
      std::vector<XMLTag*> tags = XMLTag::findXMLTags(currentLine + 
						       "</eventgroup>");
      if ( tags.empty() )
	throw std::runtime_error("Found incomplete eventgroup tag in "
				 "Les Houches file.");
      tags[0]->getattr("nreal", eup.subevents.nreal);
      tags[0]->getattr("ncounter", eup.subevents.ncounter);

      while ( !tags.empty() ) {
	delete tags.back();
	tags.pop_back();
      }

      while ( true ) {
	HEPEUP * subeup = new HEPEUP();
	if ( readEvent(subeup) )
	  eup.subevents.push_back(subeup);
	else {
	  delete subeup;
	  break;
	}
      }
      if ( eup.subevents.empty() ) return false;

      eup.setSubEvent(0);

      return true;

    }

    if ( currentLine.find("</eventgroup>") != std::string::npos ) return false;


    if ( !getline()  ) return false;
    
    // We found an event. The first line determines how many
    // subsequent particle lines we have.
    std::istringstream iss(currentLine);
    if ( !( iss >> eup.NUP >> eup.IDPRUP >> eup.XWGTUP
	        >> eup.SCALUP >> eup.AQEDUP >> eup.AQCDUP ) )
      return false;
    eup.resize();

    // Read all particle lines.
    for ( int i = 0; i < eup.NUP; ++i ) {
      if ( !getline() ) return false;
      std::istringstream iss(currentLine);
      if ( !( iss >> eup.IDUP[i] >> eup.ISTUP[i]
	          >> eup.MOTHUP[i].first >> eup.MOTHUP[i].second
         	  >> eup.ICOLUP[i].first >> eup.ICOLUP[i].second
	          >> eup.PUP[i][0] >> eup.PUP[i][1] >> eup.PUP[i][2]
	          >> eup.PUP[i][3] >> eup.PUP[i][4]
        	  >> eup.VTIMUP[i] >> eup.SPINUP[i] ) )
	return false;
    }

    // Now read any additional comments.
    while ( getline() && currentLine.find("</event>") == std::string::npos )
      eventComments += currentLine + "\n";

    if ( !file ) return false;
    if ( version < 2 ) return true;

    eup.scales = Scales(eup.SCALUP);
    eup.pdfinfo = PDFInfo(eup.SCALUP);

    std::map<std::string,std::string>::const_iterator attrit;
    // Scan the init block for XML tags
    std::string leftovers;
    std::vector<XMLTag*> tags = XMLTag::findXMLTags(eventComments, &leftovers);
    if ( leftovers.find_first_not_of(" \t\n") == std::string::npos )
      leftovers="";
    eventComments = leftovers;

    for ( int i = 0, N = tags.size(); i < N; ++i ) {
      XMLTag & tag = *tags[i];

      if ( tag.name == "weights" ) {
	eup.weights.clear();
	eup.weights.resize(heprup.weightinfo.size() + 1, std::make_pair(eup.XWGTUP, (WeightInfo*)(0)));
	for ( int i = 1, N = eup.weights.size(); i < N; ++i ) 
	  eup.weights[i].second =  &heprup.weightinfo[i - 1];
	eup.weights.front().first = eup.XWGTUP;
	double w = 0.0;
	int i = 0;
	std::istringstream iss(tag.contents);
	while ( iss >> w && ++i < int(eup.weights.size()) ) eup.weights[i].first = w;
      }
      if ( tag.name == "weight" ) {
	eup.oldweights.push_back(Weight(tag));
      }
      else if ( tag.name == "clustering" ) {
	for ( int j = 0, M= tag.tags.size(); j < M; ++j ) {
	  if ( tag.tags[j]->name == "clus" )
	    eup.clustering.push_back(Clus(*tag.tags[j]));
	}
      }
      else if ( tag.name == "pdfinfo" ) {
	eup.pdfinfo = PDFInfo(tag, eup.SCALUP);
      }
      else if ( tag.name == "scales" ) {
	eup.scales = Scales(tag, eup.SCALUP);
      }

      delete &tag;

    }
    return true;
    
  }

protected:

  /**
   * Used internally to read a single line from the stream.
   */
  bool getline() {
    return ( std::getline(file, currentLine) );
  }

protected:

  /**
   * A local stream which is unused if a stream is supplied from the
   * outside.
   */
  std::ifstream intstream;

  /**
   * The stream we are reading from. This may be a reference to an
   * external stream or the internal intstream.
   */
  std::istream & file;

  /**
   * The last line read in from the stream in getline().
   */
  std::string currentLine;

public:

  /**
   * XML file version
   */
  int version;

  /**
   * All lines (since the last readEvent()) outside the header, init
   * and event tags.
   */
  std::string outsideBlock;

  /**
   * All lines from the header block.
   */
  std::string headerBlock;

  /**
   * The standard init information.
   */
  HEPRUP heprup;

  /**
   * Additional comments found in the init block.
   */
  std::string initComments;

  /**
   * The standard information about the last read event.
   */
  HEPEUP hepeup;

  /**
   * Additional comments found with the last read event.
   */
  std::string eventComments;

private:

  /**
   * The default constructor should never be used.
   */
  Reader();

  /**
   * The copy constructor should never be used.
   */
  Reader(const Reader &);

  /**
   * The Reader cannot be assigned to.
   */
  Reader & operator=(const Reader &);

};

/**
 * The Writer class is initialized with a stream to which to write a
 * version 1.0 Les Houches Accord event file. In the constructor of
 * the Writer object the main XML tag is written out, with the
 * corresponding end tag is written in the destructor. After a Writer
 * object has been created, it is possible to assign standard init
 * information in the heprup member variable. In addition any XML
 * formatted information can be added to the headerBlock member
 * variable (directly or via the addHeader() function). Further
 * comment line (beginning with a <code>#</code> character) can be
 * added to the initComments variable (directly or with the
 * addInitComment() function). After this information is set, it
 * should be written out to the file with the init() function.
 *
 * Before each event is written out with the writeEvent() function,
 * the standard event information can then be assigned to the hepeup
 * variable and optional comment lines (beginning with a
 * <code>#</code> character) may be given to the eventComments
 * variable (directly or with the addEventComment() function).
 *
 */
class Writer {

public:

  /**
   * Create a Writer object giving a stream to write to.
   * @param os the stream where the event file is written.
   */
  Writer(std::ostream & os, bool commentEventCommentsIn = false)
    : file(os) { commentEventComments=commentEventCommentsIn; }

  /**
   * Create a Writer object giving a filename to write to.
   * @param filename the name of the event file to be written.
   */
  Writer(std::string filename, bool commentEventCommentsIn = false)
    : intstream(filename.c_str()), file(intstream)
    { commentEventComments=commentEventCommentsIn; }

  /**
   * The destructor writes out the final XML end-tag.
   */
  ~Writer() {
    file << "</LesHouchesEvents>" << std::endl;
  }

  /**
   * Add header lines consisting of XML code with this stream.
   */
  std::ostream & headerBlock() {
    return headerStream;
  }

  /**
   * Add comment lines to the init block with this stream.
   */
  std::ostream & initComments() {
    return initStream;
  }

  /**
   * Add comment lines to the next event to be written out with this stream.
   */
  std::ostream & eventComments() {
    return eventStream;
  }

  /**
   * Write out an optional header block followed by the standard init
   * block information together with any comment lines.
   */
  void init() {

    // Write out the standard XML tag for the event file.
    if ( heprup.xsecinfo.neve > 0 )
      file << "<LesHouchesEvents version=\"2.1\">\n";
    else
      file << "<LesHouchesEvents version=\"1.0\">\n";


    file << std::setprecision(8);

    using std::setw;

    std::string headerBlock = headerStream.str();
    if ( headerBlock.length() ) {
      if ( headerBlock.find("<header>") == std::string::npos )
	file << "<header>\n";
      if ( headerBlock[headerBlock.length() - 1] != '\n' )
	headerBlock += '\n';
      file << headerBlock;
      if ( headerBlock.find("</header>") == std::string::npos )
	file << "</header>\n";
    }
    file << "<init>\n"
	 << " " << setw(8) << heprup.IDBMUP.first
	 << " " << setw(8) << heprup.IDBMUP.second
	 << " " << setw(14) << heprup.EBMUP.first
	 << " " << setw(14) << heprup.EBMUP.second
	 << " " << setw(4) << heprup.PDFGUP.first
	 << " " << setw(4) << heprup.PDFGUP.second
	 << " " << setw(4) << heprup.PDFSUP.first
	 << " " << setw(4) << heprup.PDFSUP.second
	 << " " << setw(4) << heprup.IDWTUP
	 << " " << setw(4) << heprup.NPRUP << std::endl;
    heprup.resize();
    for ( int i = 0; i < heprup.NPRUP; ++i )
      file << " " << setw(14) << heprup.XSECUP[i]
	   << " " << setw(14) << heprup.XERRUP[i]
	   << " " << setw(14) << heprup.XMAXUP[i]
	   << " " << setw(6) << heprup.LPRUP[i] << std::endl;

    if ( heprup.xsecinfo.neve <= 0 ) {
      file << hashline(initStream.str()) << "</init>" << std::endl;
      eventStream.str("");
      return;
    }

    for ( int i = 0, N = heprup.generators.size(); i < N; ++i ) {
      file << "<generator";
      if ( heprup.generators[i].second.size() )
	file << " version=\"" << heprup.generators[i].second << "\"";
      file << ">" << heprup.generators[i].first << "</generator>" << std::endl;
    }

    heprup.xsecinfo.print(file);

    if ( heprup.cuts.size() > 0 ) {
      file << "<cutsinfo>" << std::endl;
    
      for ( std::map<std::string, std::set<long> >::iterator ptit =
	      heprup.ptypes.begin(); ptit !=  heprup.ptypes.end(); ++ptit ) {
	file << "<ptype name=\"" << ptit->first << "\">";
	for ( std::set<long>::const_iterator it = ptit->second.begin();
	      it != ptit->second.end(); ++it )
	  file << " " << *it;
	file << "</ptype>" << std::endl;
      }

      for ( int i = 0, N = heprup.cuts.size(); i < N; ++i )
	heprup.cuts[i].print(file);
      file << "</cutsinfo>" << std::endl;
    }

    for ( std::map<long,ProcInfo>::const_iterator it = heprup.procinfo.begin();
	  it != heprup.procinfo.end(); ++it )
      it->second.print(file);

    for ( std::map<long,MergeInfo>::const_iterator it = heprup.mergeinfo.begin();
	  it != heprup.mergeinfo.end(); ++it )
      it->second.print(file);

    int ingroup = -1;
    for ( int i = 0, N = heprup.weightinfo.size(); i < N; ++i ) {
      int group = heprup.weightinfo[i].inGroup;
      if ( group != ingroup ) {
	if ( ingroup != -1 ) file << "</weightgroup>\n";
	if ( group != -1 )
	  file << "<weightgroup name=\"" << heprup.weightgroup[group].name << "\" >\n";
	ingroup = group;
      }
      heprup.weightinfo[i].print(file);
    }
    if ( ingroup != -1 ) file << "</weightgroup>\n";

    file << hashline(initStream.str()) << "</init>" << std::endl;
    eventStream.str("");
  }

  /**
   * Write out the event stored in hepeup, followed by optional
   * comment lines.
   */
  bool writeEvent(HEPEUP * peup = 0) {

    HEPEUP & eup = (peup? *peup: hepeup);

    if ( !eup.subevents.empty() ) {
      file << "<eventgroup";
      if ( eup.subevents.nreal > 0 )
	file << " nreal=\"" << eup.subevents.nreal << "\"";
      if ( eup.subevents.ncounter > 0 )
	file << " ncounter=\"" << eup.subevents.ncounter << "\"";
      file << ">" << std::endl;

      for ( int i = 0, N = eup.subevents.size(); i < N; ++i )
	if ( !writeEvent(eup.subevents[i]) ) return false;

      file << "</eventgroup>" << std::endl;

      return true;

    }

    using std::setw;

    file << "<event>\n";
    file << " " << setw(4) << eup.NUP
	 << " " << setw(6) << eup.IDPRUP
	 << " " << setw(14) << eup.XWGTUP
	 << " " << setw(14) << eup.SCALUP
	 << " " << setw(14) << eup.AQEDUP
	 << " " << setw(14) << eup.AQCDUP << "\n";
    eup.resize();

    for ( int i = 0; i < eup.NUP; ++i )
      file << " " << setw(8) << eup.IDUP[i]
	   << " " << setw(2) << eup.ISTUP[i]
	   << " " << setw(4) << eup.MOTHUP[i].first
	   << " " << setw(4) << eup.MOTHUP[i].second
	   << " " << setw(4) << eup.ICOLUP[i].first
	   << " " << setw(4) << eup.ICOLUP[i].second
	   << " " << setw(14) << eup.PUP[i][0]
	   << " " << setw(14) << eup.PUP[i][1]
	   << " " << setw(14) << eup.PUP[i][2]
	   << " " << setw(14) << eup.PUP[i][3]
	   << " " << setw(14) << eup.PUP[i][4]
	   << " " << setw(1) << eup.VTIMUP[i]
	   << " " << setw(1) << eup.SPINUP[i] << std::endl;

    if ( heprup.xsecinfo.neve > 0 ) {

      if ( eup.weights.size() > 0 ) {
	file << "<weights>";
	for ( int i = 1, N = eup.weights.size(); i < N; ++i )
	  file << " " << eup.weights[i].first;
	file << "</weights>\n";
      }

      for ( int i = 0, N = eup.oldweights.size(); i < N; ++i )
	eup.oldweights[i].print(file);

      if ( !eup.clustering.empty() ) {
	file << "<clustering>" << std::endl;
	for ( int i = 0, N = eup.clustering.size(); i < N; ++i )
	  eup.clustering[i].print(file);
	file << "</clustering>" << std::endl;	
      }

      eup.pdfinfo.print(file);
      eup.scales.print(file);

    }

    file << hashline(eventStream.str()) << "</event>\n";

    eventStream.str("");

    if ( !file ) return false;

    return true;

  }

protected:

  /**
   * Make sure that each line in the string \a s starts with a
   * #-character and that the string ends with a new-line.
   */
  std::string hashline(std::string s) {
    std::string ret;
    std::istringstream is(s);
    std::string ss;
    while ( getline(is, ss) ) {
      if ( commentEventComments
          && ( ss.find('#') == std::string::npos ||
               ss.find('#') != ss.find_first_not_of(" \t") ) )
          ss = "# " + ss;
      ret += ss + '\n';
    }
    return ret;
  }

protected:

  /**
   * A local stream which is unused if a stream is supplied from the
   * outside.
   */
  std::ofstream intstream;

  /**
   * The stream we are writing to. This may be a reference to an
   * external stream or the internal intstream.
   */
  std::ostream & file;

  /**
   * Boolean specifying if the writer should always hash-tag eventComment lines.
   */
  bool commentEventComments;

public:

  /**
   * Stream to add all lines in the header block.
   */
  std::ostringstream headerStream;

  /**
   * The standard init information.
   */
  HEPRUP heprup;

  /**
   * Stream to add additional comments to be put in the init block.
   */
  std::ostringstream initStream;

  /**
   * The standard information about the event we will write next.
   */
  HEPEUP hepeup;

  /**
   * Stream to add additional comments to be written together the next event.
   */
  std::ostringstream eventStream;

private:

  /**
   * The default constructor should never be used.
   */
  Writer();

  /**
   * The copy constructor should never be used.
   */
  Writer(const Writer &);

  /**
   * The Writer cannot be assigned to.
   */
  Writer & operator=(const Writer &);

};

}

/** \example LHEFCat.cc This is a main function which simply reads a
    Les Houches Event File from the standard input and writes it again
    to the standard output. 
    This file can be downloaded from
    <A HREF="http://www.thep.lu.se/~leif/LHEF/LHEFCat.cc">here</A>. 
    There is also a sample
    <A HREF="http://www.thep.lu.se/~leif/LHEF/ttV_unweighted_events.lhe">event
    file</A>
    to try it on.
*/

/**\mainpage Les Houches Event File

Here are some example classes for reading and writing Les Houches
Event Files according to the
<A HREF="http://www.thep.lu.se/~torbjorn/lhef/lhafile2.pdf">proposal</A>
by Torbj&ouml;rn Sj&ouml;strand discussed at the
<A HREF="http://mc4lhc06.web.cern.ch/mc4lhc06/">MC4LHC</A>
workshop at CERN 2006.

The classes has now been updated to handle the suggested version 2.1 of
this file standard as discussed at the <a href="http://phystev.in2p3.fr/wiki/2013:groups:tools:lhef3">Les Houches workshop 2013</a> (The previous suggested version 2.0 was discussed at the <a href="http://www.lpthe.jussieu.fr/LesHouches09Wiki/index.php/LHEF_for_Matching">Les Houches workshop 2009</a>).

There is a whole set of classes available in a single header file
called <A HREF="http://www.thep.lu.se/~leif/LHEF/LHEF.h">LHEF.h</A>.

The two classes LHEF::HEPRUP and LHEF::HEPEUP are simple container
classes which contain the same information as the Les Houches standard
Fortran common blocks with the same names. They also contain the extra
information defined in version 2.1 in the standard. The other two main
classes are called LHEF::Reader and LHEF::Writer and are used to read
and write Les Houches Event Files

Here are a few <A HREF="examples.html">examples</A> of how to use the
classes:

\namespace LHEF The LHEF namespace contains some example classes for reading and writing Les Houches
Event Files according to the
<A HREF="http://www.thep.lu.se/~torbjorn/lhef/lhafile2.pdf">proposal</A>
by Torbj&ouml;rn Sj&ouml;strand discussed at the
<A HREF="http://mc4lhc06.web.cern.ch/mc4lhc06/">MC4LHC</A>
workshop at CERN 2006.



 */


#endif /* THEPEG_LHEF_H */
