//--------------------------------------------------------------------
// RBOOK -- A simple fortran interface to ROOT histogramming
//
// This file implements the C++ back-end functions that call the appropriate ROOT code
// 
//
// (C) 31/08/2006 NIKHEF / Wouter Verkerke
// -------------------------------------------------------------------
//
// Modified on 22/11/2013 by SF to use double precision; changes to
// ROPERA (made it more similar to MOPERA), and turned off storage
// of errors (do not make sense at NLO fixed order)
//

#include <iostream>
#include <map>
#include <math.h>
#include "TObject.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TFile.h"
#include "TROOT.h"

using namespace std;


// Global objects that store list of created histograms and file object
map<int,TObject*> _theList ;
TFile* _theFile = 0 ;
TROOT* _theRoot = 0 ;


// Declare all functions as 'extern C' so that we don't have to worry about C++ name
// mangling conventions in fortran side of code
extern "C" 
{
  int rinit_be__(char* , int*) ;
  void rbook_be__(int*, char*, int*, int*, double*, double*);
  void rbook2_be__(int*, char*, int*, int*, double*, double*, int*, double*, double*);
  void rfill_be__(int *, double*, double*);
  void rfill2_be__(int *, double*, double*, double*);
  void rclos_be__() ;
  void rwrit_be__() ;
  void ropera_be__(int*, char*, int*, int*, int*, double*, double*) ;
  void rcopy_be__(int*,int*) ;
}

// -----------------------------------
// RINIT back end -- Open a ROOT file
// -----------------------------------
int rinit_be__(char* fname, int* nfname)
{
  // Print Banner
  cout << "RINIT -- Initializing RBOOK fortran interface to ROOT histogramming -- (C) 2006 W. Verkerke" << endl ;

  // Copy FORTRAN string to C string
  char cname[256] ;
  memcpy(cname,fname,*nfname) ; cname[*nfname]=0 ;

  // Show what file was opened
  cout << "RINIT -- Opening ROOT file '" << cname << "' for histogram storage" << endl ;

  // Open ROOT file with given name
  _theFile = new TFile(cname,"RECREATE") ;

  // Check for success
  if (!_theFile) {
    cout << "RINIT -- There were problems opening the ROOT file, no histograms will be written" << endl ;
  }

  // Return success status 1=OK,0=failure
  return _theFile?1:0 ;
}

// --------------------------------------
// RWRIT back end -- Write to ROOT file
// --------------------------------------
void rwrit_be__()
{
  // Check that file was actually opened
  if (_theFile) {

    cout << "Writing events to ROOT file '" << _theFile->GetName() << "'" << endl ;

    // Flush contents
    _theFile->Write() ;


  } else {

    cout << "RWRIT -- Error: no ROOT file was opened" << endl ;
  }
}


// --------------------------------------
// RCLOSE back end -- Close the ROOT file
// --------------------------------------
void rclos_be__()
{
  // Check that file was actually opened
  if (_theFile) {

    cout << "RCLOS -- Closing ROOT file '" << _theFile->GetName() << "'" << endl ;

    // Close file and release memory
    _theFile->Close() ;
    delete _theFile ;

  } else {

    cout << "RCLOS -- Error: no ROOT file was opened" << endl ;
  }
}


// --------------------------------------
// RBOOK back end -- Create 1D histogram
// --------------------------------------
void rbook_be__(int* id, char* fname, int* nfname, int* nbins, double* xlo, double* xhi)
{
  // Convert fortran name to C++ string
  char cname[256] ;
  memcpy(cname,fname,*nfname) ; cname[*nfname]=0 ;

  // Print message
  cout << "RBOOK -- Creating TH1D " << Form("h%d",*id) << "[" << *xlo << "-" << *xhi << ":" << *nbins << "] : " << cname << endl ;

  // Create a TH1D object
  TH1D* h = new TH1D(Form("h%d",*id),cname,*nbins,*xlo,*xhi) ;
  
  // Tell TH1 to keep track of weights for sum(w2) errors
  //UNCOMMENT WHAT FOLLOWS TO REINSTATE ERRORS
  //h->Sumw2() ;

  // Save histogram in master list
  _theList[*id] = h ;
}



// ---------------------------------------
// RBOOK2 back end -- Create 2D histogram
// ---------------------------------------
void rbook2_be__(int* id, char* fname, int* nfname, int* xbins, double* xlo, double* xhi, 
		int* ybins, double* ylo, double* yhi)
{
  // Convert fortran name to C++ string
  char cname[256] ;
  memcpy(cname,fname,*nfname) ; cname[*nfname]=0 ;

  // Print message
  cout << "RBOOK2 -- Creating TH2D " << Form("h%d",*id) << "[" << *xlo << "-" << *xhi << ":" << *xbins << "," 
       << *ylo << "-" << *yhi << ":" << *ybins << "] : " << cname << endl ;

  // Create a TH1D object
  TH2D* h = new TH2D(Form("h%d",*id),cname,*xbins,*xlo,*xhi,*ybins,*ylo,*yhi) ;
  
  // Tell TH1 to keep track of weights for sum(w2) errors
  //UNCOMMENT WHAT FOLLOWS TO REINSTATE ERRORS
  //h->Sumw2() ;

  // Save histogram in master list
  _theList[*id] = h ;
}



// ------------------------------------
// RFILL back end -- Fill 1D histogram
// ------------------------------------
void rfill_be__(int* h, double* x, double* wgt) 
{
  // Pull the request histogram from the master list
  TH1D* hh = (TH1D*) _theList[*h] ;
  
  // Check if a histogram has been created for given ID
  if (hh) {
    // Fill the histogram
    hh->Fill(*x,*wgt) ;
  } else {
    cout << "RFILL -- Error: no histogram defined with ID " << *h << endl ;
  }
}



// ------------------------------------
// RFILL2 back end -- Fill 2D histogram
// ------------------------------------
void rfill2_be__(int* h, double* x, double* y, double* wgt) 
{
  // Pull the request histogram from the master list
  TH2D* hh = (TH2D*) _theList[*h] ;
  
  // Check if a histogram has been created for given ID
  if (hh) {
    // Fill the histogram
    hh->Fill(*x,*y,*wgt) ;
  } else {
    cout << "RFILL2 -- Error: no histogram defined with ID " << *h << endl ;
  }
}


// -------------------------------------------
// ROPERA back end -- Manipulate histograms
// ------------------------------------------
void ropera_be__(int* ih1, char* oper, int* operlen, int* ih2, int* ih3, double* x, double* y)
{
  // Convert fortran name to C++ string
  char op[256] ;
  memcpy(op,oper,*operlen) ; op[*operlen]=0 ;

  // Pull the request histogram from the master list
  TH1D* h1 = (TH1D*) _theList[*ih1] ;
  TH1D* h2 = (TH1D*) _theList[*ih2] ;
  TH1D* h3 = (TH1D*) _theList[*ih3] ;

  // Check histogram have been created for given IDs
  // Modified on 22/11/2013 by SF: do not stop if h1 or h2 are not booked,
  // just quit; if h3 is not booked, book it. This is in keeping with
  // the original MOPERA
  if (!h1) {
    //    cout << "ROPERA -- Error: no input histogram defined with ID " << *ih1 << endl ;
    return ;
  }
  if (!h2) {
    //    cout << "ROPERA -- Error: no input histogram defined with ID " << *ih2 << endl ;
    return ;
  }
  if (!h3) {
    //    cout << "ROPERA -- Error: no output histogram defined with ID " << *ih3 << endl ;

    // Clone h1 into h3, so the same title and binning will be used
    TH1D* h3 = (TH1D*)h1->Clone();

    // Rename the histogram (root finds histograms by name)
    h3->SetName(Form("h%d",*ih3));

    // Set all bins of the new histogram equal to zero
    h3->Scale(0.0);

    // Save the new histogram in master list
    _theList[*ih3] = h3 ;

    return ;
  }  

//  +  : sums       X*(hist I) with Y*(hist J) and puts the result in hist K;
//  -  : subtracts  X*(hist I) with Y*(hist J) and puts the result in hist K;
//  *  : multiplies X*(hist I) with Y*(hist J) and puts the result in hist K;
//  /  : divides    X*(hist I) with Y*(hist J) and puts the result in hist K;
//  F  : multiplies hist I by the factor X, and puts the result in hist K;
//  R  : takes the square root of  hist  I, and puts the result in hist K;if
//       the value at a given bin is less than or equal to 0, puts 0 in K
//  S  : takes the square      of  hist  I, and puts the result in hist K;
//  L  : takes the log_10 of  hist  I, and puts the result in hist K; if the
//       value at a given bin is less than or equal to 0, puts 0 in K

  switch(op[0]) {
  case '+': 
    h3->Add(h1,h2,*x,*y) ;
    break ;
  case '-': 
    h3->Add(h1,h2,*x,-1*(*y)) ;
    break ;
  case '*': 
    h3->Multiply(h1,h2,*x,*y) ;
    break ;
  case '/': 
    h3->Divide(h1,h2,*x,*y) ;
    break ;
  case 'F': 
    for (Int_t i=1 ; i<=h1->GetNbinsX() ; i++) {
      h3->SetBinContent(i, *x *  h1->GetBinContent(i)) ;
    }
    break ;
  case 'R': 
    for (Int_t i=1 ; i<=h1->GetNbinsX() ; i++) {
      Double_t v = h1->GetBinContent(i) ;
      h3->SetBinContent(i, v>0?sqrt(v):0.) ;
    }
    break ;
  case 'S': 
    for (Int_t i=1 ; i<=h1->GetNbinsX() ; i++) {
      Double_t v = h1->GetBinContent(i) ;
      h3->SetBinContent(i, v*v) ;
    }
    break ;
  case 'L': 
    for (Int_t i=1 ; i<=h1->GetNbinsX() ; i++) {
      Double_t v = h1->GetBinContent(i) ;
      h3->SetBinContent(i, v>0?log10(v):0.) ;
    }
    break ;
  case 'M':
  case 'V':
    cout << "ROPERA -- ERROR: options 'M' and 'V' not implemented" << endl ;
    break ;
  default:
    cout << "ROPERA -- ERROR: unknown option '" << op << "'" << endl ;  
    break ;
  }

  return ;
}


// -------------------------------------------
// RCOPY back end -- Copy histograms
// ------------------------------------------
void rcopy_be__(int* ih1, int* ih2) 
{
  // Pull the request histogram from the master list
  TH1D* h1 = (TH1D*) _theList[*ih1] ;
  TH1D* h2 = (TH1D*) _theList[*ih2] ;

  if (!h1) {
    cout << "RCOPY -- ERROR: input histogram " << *ih1 << " does not exist" << endl;
    return ;
  }
  if (h2) {
    delete h2 ;
  }

  // Delete ih2, clone ih1 and use it as replacement for ih2
  h2 = (TH1D*) h1->Clone(Form("h%d",*ih2)) ;
  _theList[*ih2] = h2 ;
  
}

