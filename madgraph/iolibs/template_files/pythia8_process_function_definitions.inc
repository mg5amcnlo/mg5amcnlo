//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

//--------------------------------------------------------------------------
// Initialize process. 
  
void %(process_class_name)s::initProc() {
    %(initProc_lines)s
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

void %(process_class_name)s::sigmaKin() { 
    %(sigmaKin_lines)s
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence. 

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

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for a given helicity

double %(process_class_name)s::matrix(int hel[]) { 
    %(matrix_lines)s
}

//--------------------------------------------------------------------------
// Set couplings and other parameters that are fixed during the run

void %(process_class_name)s::set_fixed_parameters() { 
    %(fixed_parameter_lines)s
}

//--------------------------------------------------------------------------
// Set couplings and other parameters that vary event by event

void %(process_class_name)s::set_variable_parameters() { 
    %(variable_parameter_lines)s
}

