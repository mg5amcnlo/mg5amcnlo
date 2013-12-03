/*
 *     dbin.h
 *
 *  C++ utility routines for the dbin package: see dbin.lex
 *
 *  N.B. The Strings class from the CLHEP library is used.
 *
 *       Torre Wenaus 04/01/1994
 *
 * Modifications:
 * 8/21/95   T. Wenaus Mod history started
 * 8/21/95   TW        Strings class removed from dbin generated code.
 * 8/22/95   TW        Strings class removed from dbinc.cc
 *
 * November 1995: some clean up to be able to run this code and 
 * standard dbin simulateneously..
 * Make some routine & variable static, and change the name of routine 
 * called from the outside, following the Nirvana/mcfio conventions.
 * 
 */
void mcf_ntubldRead(char* fname);
