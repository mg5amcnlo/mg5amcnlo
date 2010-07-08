//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

//--------------------------------------------------------------------------
// Initialize process. 
  
void %(process_class_name)s::initProc() {

    %(initProc_lines)s
} 

//--------------------------------------------------------------------------
// Evaluate d(sigmaHat)/d(tHat), part independent of incoming flavour. 

void %(process_class_name)s::sigmaKin() { 

    %(sigmaKin_lines)s
}

//--------------------------------------------------------------------------
// Evaluate d(sigmaHat)/d(tHat), including incoming flavour dependence. 

double %(process_class_name)s::sigmaHat() {  

    %(sigmaHat_lines)s
}

//--------------------------------------------------------------------------
// Select identity, colour and anticolour.

void %(process_class_name)s::setIdColAcol() {

    %(setIdColAcol_lines)s
}

//--------------------------------------------------------------------------
// Evaluate weight for angles of decay products in process 

double %(process_class_name)s::weightDecay(Event& process, int iResBeg, int iResEnd) {

    %(weightDecay_lines)s
}

//--------------------------------------------------------------------------
// Evaluate d(sigmaHat)/d(tHat), part independent of incoming flavour. 

double %(process_class_name)s::matrix(int nhel[]) { 

    %(matrix_lines)s
}
