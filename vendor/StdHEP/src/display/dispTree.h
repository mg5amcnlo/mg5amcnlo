/*******************************************************************************
*									       *
* disptree.h -- Nirvana Event Tree Display				       *
*									       *
* Copyright (c) 1991 Universities Research Association, Inc.		       *
* All rights reserved.							       *
* 									       *
* This material resulted from work developed under a Government Contract and   *
* is subject to the following license:  The Government retains a paid-up,      *
* nonexclusive, irrevocable worldwide license to reproduce, prepare derivative *
* works, perform publicly and display publicly by or for the Government,       *
* including the right to distribute to other Government contractors.  Neither  *
* the United States nor the United States Department of Energy, nor any of     *
* their employees, makes any warrenty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
* Fermilab Nirvana GUI Library						       *
* October 30, 1991							       *
*									       *
* Written by Paul  Lebrun & Marc Edel						       * 
*									       *
*******************************************************************************/
/*
 * Define the structure for a node in the Event  tree.
 */

typedef struct _nodehep {
  int 	   stdid;   		/* The standard Particle id */
  int	   multiplicity;        /* The number of particles in this node.. */
  int      *stdIndex;           /* A pointer to an array containing the Indices
  				    to the hepevt_ structure. */
  struct _nodehep  *daughter;   /* The pointer to the last born child */
  struct _nodehep  *sister;     /* the pointer to the sister  */
  struct _nodehep  *backsister;  /* back pointer to the previous sister */
  struct _nodehep  *mother;     /* The "back pointer" to the mother */
  int      *stwidget;           /* The pointer to the "tree widget" structure */
  StdHepWindow *window;           /* The pointer to the original window. */
} nodehep;

#define TREEDISPREAL 1
#define TREEDISPCODE 2

/*
* Top level routine 
*/
    void DispTree(Widget w, StdHepWindow *window);
    void FreeTree(nodehep *node);
    void ShowTree(StdHepWindow *window,
	                       nodehep *pnode, Widget super_node);
    nodehep *SearchNodes(nodehep *node, int stdi);                       
    void CloseHepTreeWindow(Widget w, StdHepWindow *window);	    
