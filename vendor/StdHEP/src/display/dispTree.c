/*******************************************************************************
* dispTree.c  Display a Standard HEP event tree. 			       *
* This is based on the Constraint Tree Widget , from Douglas Young.	       *
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
* March 18, 1991							       *
*									       *
* Written by Paul Lebrun						       *
*									       *
*******************************************************************************/

#include <stdio.h>
#include <X11/StringDefs.h>
#include <X11/Intrinsic.h> 
#include <Xm/Xm.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <stdlib.h>
#include <Xm/ScrolledW.h>
#include <Xm/DialogS.h>
#include <Xm/BulletinB.h>
#include "spin/Spin.h"
#include "tree.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "dispTree.h"
#include "pick.h"
#include "drawEvent.h"
#include "listNodeHep.h"
#include "util/stringUtils.h"


#define MAXCHAR_HEPNAM 30  /* for use with hepnam_ */
/*   #define DEBUG */		
/*
* Function declarations
*/
static nodehep *hepevt_tree(StdHepWindow *window);
static nodehep *make_node(StdHepWindow *window, int *list, int nn  );
static void drawNodehep(nodehep *ptr);
static void listnodeCB(Widget w, nodehep *ptr, caddr_t callData); 
static void closeCB(Widget w, StdHepWindow *window);
static void closecolorCB(Widget w, StdHepWindow *window);
         
void DispTree(Widget w, StdHepWindow *window)
{
  Widget     treediag, sw,  tree;
  int        i, nn;
  int idum = 0;
  nodehep      *head = NULL;
  int        ac,digit;  
  Arg        wargs[10];
  char title[40];
  
   if ((window->modetreedisp != TREEDISPREAL) &&
       (window->modetreedisp != TREEDISPCODE)) {
         printf (" Internal error, don't know which tree to set \n");
         return;
         }
  /*
  * First create an application shell to hold the scrollable window.
  */
   if (window->modetreedisp == TREEDISPREAL) {
     if (window->dtree == NULL) {
       sprintf(title," Tree for StdHep event %d ",window->event.userData); 
       treediag = XmCreateDialogShell(w, title, wargs,0);
       AddMotifCloseCallback(treediag, closeCB, (void *) window);
       window->dtree = treediag;
       /*
       * Put the tree in a scrolled window, to handle 
       * large trees.
       */
        ac = 0;
        XtSetArg(wargs[ac], XmNscrollingPolicy, XmAUTOMATIC); ac++;
        XtSetArg(wargs[ac], XmNwidth, 300); ac++;
        XtSetArg(wargs[ac], XmNheight, 500); ac++;
         sw = XtCreateManagedWidget("swindow",
                              xmScrolledWindowWidgetClass,
                              treediag, wargs, ac);
        window->stree = sw;
       } else {
          sprintf(title," Tree for StdHep event %d ",window->event.userData); 
          XtVaSetValues(window->dtree, XmNtitle, title,
                 XmNiconName, "Tree", 0);
          treediag = window->dtree;
          sw = window->stree;
       }
          
       /*
       * Always recreate the tree..
       */
       if (window->tree != NULL) XtDestroyWidget(window->tree);
       /*
        * Create the tree widget.
        */
       tree = XtCreateManagedWidget("tree", XstreeWidgetClass, 
                                sw, NULL, 0);
       window->tree = tree; 
       /*
       * Create the HEP tree from the STD HEP structure.
       */
       if (window->treehead == NULL) {
          head = hepevt_tree(window);
          window->treehead = (int *) head;
         } else head = (nodehep *) window->treehead;
 
       } else {
       
/* Color code.. Make it also scrollable window */

       if (window->dtreecolorcode == NULL) {
       sprintf(title," Color Code Table for Phase and Tree Display ");
       treediag = XmCreateDialogShell(w, title, wargs,0);
       AddMotifCloseCallback(treediag, closecolorCB, (void *) window);
       window->dtreecolorcode = treediag;
        ac = 0;
        XtSetArg(wargs[ac], XmNscrollingPolicy, XmAUTOMATIC); ac++;
        XtSetArg(wargs[ac], XmNwidth, 500); ac++;
        XtSetArg(wargs[ac], XmNheight, 600); ac++;
         sw = XtCreateManagedWidget("ColorsParticles",
                              xmScrolledWindowWidgetClass,
                              treediag, wargs, ac);
        window->streecolorcode = sw;
       } else {
          treediag = window->dtreecolorcode;
          sw = window->streecolorcode;
       }
       if (window->treecolorcode != NULL) 
           XtDestroyWidget(window->treecolorcode);
       tree = XtCreateManagedWidget("tree", XstreeWidgetClass, 
                                sw, NULL, 0);
       window->treecolorcode = tree;
       if (window->treeheadcolorcode == NULL) {
         head = hepevt_tree(window);
         window->treeheadcolorcode = (int *) head;
       } else head = (nodehep *) window->treeheadcolorcode;
    } 
   /*
    * Create the widgets representing the tree.
    */
   
   ShowTree(window, head, NULL);
   return;
}

static nodehep *hepevt_tree(StdHepWindow *window)
{
   int   index, indexm, nn, create;
   int   i, j, k, ndone, nprev, imo;
   int   *nlist, *nl1;
   PhaseParticle *p1, *p2;
   PhaseEvent *evt;
   
  nodehep *top, *cur, *mother, *backcur;
  /*
   * At the top of the tree, we place a dummy key 
   */
   nn = -1;
  
  top = make_node(window, NULL,nn);
  top->stdid = window->event.eventNum;
  /*
  * Now all the particles nodes. 
  */
  if (window->modetreedisp == TREEDISPREAL) evt = &(window->event);
  else evt = &(window->colorcode);
   
  create= -1;
  while (create) {
   p1 = evt->particles;
   for(i=0; i < evt->nParticles ; i++, p1++){
    create = 0;
    /* 
    * Look if this node has been already defined... If not defined, do it 
    * In addition, the mother better be created...
    */
    imo = p1->mother - 1;
    if (p1->mother == 0) 
          mother = top;
       else {
          mother = SearchNodes(top, imo);
#ifdef DEBUG
          printf(" make tree... Part %d has mother %d , at node %x \n",
            i, imo, mother);
#endif
     }    
    if ((SearchNodes(top,i) == NULL) & (mother != NULL)) {
      /* 
       * We will create a node for this particle..
       * Collect all the particles belonging to this node.  Conditions
       * are (i) the particle is off the same type. (ii) the particle
       * have the same mother (iii) the particles have no daughters
       */
       nn = 1;
       p2 = evt->particles;
       for(j=0; j < evt->nParticles  ; j++, p2++){
         if ( (i != j) && (p1->id == p2->id) && (p1->mother == p2->mother)) {
               if (p1->firstdaughter == 0) nn++;
           }
        }
       nlist = (int *) XtMalloc(sizeof(int) * nn);
       *nlist = i;
       nl1 = nlist;
       nlist++;
       p2 = evt->particles;
       if (nn > 1) {
         for(j=0; j < evt->nParticles  ; j++, p2++ ){
          if ( (i != j) && (p1->id == p2->id) && (p1->mother == p2->mother)) {
               if (p1->firstdaughter == 0) {
                   *nlist = j;
                   nlist++;
                  }
           }
          }
        }
        /*
        * Make a node and connect it...
        */
        cur = make_node(window, nl1,nn);
        cur->mother = mother;
        create = -1;
        if(mother->daughter != NULL) {
#ifdef DEBUG 
           printf(" make tree.. daughter %x becomes sister of %x \n", 
           mother->daughter, cur);
#endif
           cur->sister = mother->daughter;
           backcur = cur->sister;
           backcur->backsister = cur;
           
       }   
       mother->daughter = cur;
#ifdef DEBUG
        printf(" make tree.. mother %x has daughter %x \n \n", 
           mother, cur);
#endif

     }
   }
  }    
  return (top);
}

static nodehep *make_node(StdHepWindow *window, int *list, int nn)
{
  int index;
  int nnp, j, k;
  int *l, *jid;
  PhaseParticle *p1;
  PhaseEvent *evt;
  nodehep  *ptr = (nodehep *) malloc(sizeof(nodehep));
  
    if (window->modetreedisp == TREEDISPREAL) evt = &(window->event);
  else evt = & (window->colorcode);

  if (list != NULL) {
   index = *list;
   p1 = evt->particles; 
   for(k=0; k< index; k++, p1++); 
   ptr->stdid = p1->id;
   ptr->stdIndex = list;
  } else ptr->stdIndex = NULL;
   ptr->multiplicity = nn;
   ptr->daughter = ptr->sister = ptr->mother = NULL;
   ptr->backsister = NULL;
   ptr->stwidget = NULL;
   ptr->window = window;
#ifdef DEBUG
   printf (" make_node... multiplicity \%d id. %d ",nn, ptr->stdid);
#endif
   nnp = nn;
  if (nnp > 10) nnp=10; 
   if (nnp >0 ) {
#ifdef DEBUG
     printf(" Hepevt Indices..");
#endif
     l = list;
     for ( j=0; j <nnp; j++) {
     	index = *l;
#ifdef DEBUG
   	printf ("  %d", index); l++;
#endif 
     }
#ifdef DEBUG
     printf (" \n");
#endif
   }
  return (ptr);
}

nodehep *SearchNodes(nodehep *node, int stdi)
{
	int i, nn;
	int *l;
	nodehep *next;
  /* 
   * Generic search through a tree... 
   */
   if (node == NULL) return(NULL);
   nn = node->multiplicity;
   l = node->stdIndex;
   for (i=0; i <nn; i++){ 
   	if ( *l == stdi) return (node);
   	l++;
   }
   if (node->daughter != NULL) { 
      next = SearchNodes(node->daughter,stdi);
      if (next != NULL) return(next);
    }
   if (node->sister != NULL) { 
      next = SearchNodes(node->sister,stdi);
      if (next != NULL) return(next);
    }
    return(NULL);   
}

void FreeTree(nodehep *node)
{
	nodehep *mother, *cur, *sister, *next;
	int *l;
	
	/*
	* If point to null, assume we are done...
	*/
	if (node == NULL) return;
	/*
	* Make sure node has no daughter, and if it has, move to 
	* the last granddaughter in the family..
	*/
	cur = node;
	while (cur->daughter != NULL) {
		cur = cur->daughter;
	}
	/*
	* reassing the sister to be the first daughter ( if any),
	* and set the next node to be free to that daughter.  if no 
	* such daughter, then set next to be the mother.
	*/
	mother = cur->mother;
	sister = cur->sister;
	if (sister == NULL) { 
	   if (mother != NULL) mother->daughter = NULL;
	   next = mother;
	  }
	  else {
	    mother->daughter = sister;
	    next = sister;
	  }
#ifdef DEBUG 
           printf(" freeing tree at  %x \n", cur);
#endif
	 /*
	 * Now free the node at cur...
	 */
	 l = cur->stdIndex;
	 if (l != NULL) free(l);
	 free(cur);
	 /*
	 * and get going to the next one ..
	 */
	 FreeTree(next);
}
 
void ShowTree(StdHepWindow *window,
                        nodehep *mother, Widget super_node)
{
  Widget   w, sw, parent;
  Arg      wargs[3];
  int nm, i, j;
  int      n = 0;
  int backup;
  nodehep *lastsister, *curtmp, *cursister, *newmother;
  Widget *mw;
  
   if (window->modetreedisp == TREEDISPREAL) parent = window->tree;
   else parent = window->treecolorcode;
  /*
   * We draw the daughter and her sisters of the mother nodehep 
   * We assume that the mother has already been drawn.. First ensure
   * we have a real mother...
   */
  if (!mother) return;
  /* 
  * first time around : if the super node is null, there is no mother 
  * to mother, we assume that we have to draw the top of the tree...
  */
  if(!super_node){
  	n = 0;
  	XtSetArg(wargs[n], XtNsuperNode, super_node); n++;
 	w  =  XtCreateManagedWidget("node", xmPushButtonWidgetClass, 
                              parent, wargs, n);
  	mother->stwidget = (int *) w;
    	drawNodehep(mother);
  	sw = w;
    	}
    	else {
    	sw = super_node;
    	}
    
  cursister = mother->daughter;
  /* 
  * Find the last sister... For esthetique reasons, we want to 
  * to draw the tree backwards, since it was created backwards for 
  * reason of convenience... 
  */
  lastsister = cursister;
  if (cursister != NULL) {
    while ( cursister->sister != NULL) {
    	cursister = cursister->sister;
    	}
    lastsister = cursister;
    while ( cursister != NULL) {
      /*
       * Create a widget for the node, specifying the
       * given super_node constraint.
       */
        n = 0;
  	XtSetArg(wargs[n], XtNsuperNode, sw); n++;
 	w  =  XtCreateManagedWidget("node", xmPushButtonWidgetClass, 
                              parent, wargs, n);
  	cursister->stwidget = (int *) w;
    	drawNodehep(cursister);
    	cursister = cursister->backsister;
    	}
/*    	cursister = mother->daughter;  */
        cursister = lastsister;
   }
   /*
   * Now find the next mother, to set the recurrence.
   *
   */
   if (cursister == NULL) {
      backup = -1;
      curtmp = mother;
      while (backup == -1) {
        newmother = curtmp->backsister;
        if (newmother != NULL) 
          backup = 0;
        else {
          if (curtmp->mother == NULL) {
               /* 
                * end backing up, we are at the top of the tree...
                */
                newmother = NULL;
                backup = 0;
                }
                else {
                /*
                *  backup one level...
                */
                curtmp = curtmp->mother;
                }
           } 
         }
       }
       else {
         /* 
         * Look for the most backward daughter 
         */
         newmother = lastsister;
         }
          
      if ( newmother != NULL) 
      	sw =  ((Widget) newmother->stwidget);
      else
        sw = NULL;  
     ShowTree(window,  newmother, sw);
}

static void drawNodehep( nodehep *ptr){

	char strnum[20], nampart[MAXCHAR_HEPNAM];
	XmString xstrnum, xnampart, xminfo;
	Arg 	wargs[10];
	int i, n, lnp, icol, id, selected, itrsel;
	int *list;
	Widget w;
	StdHepWindow *window;
        Pixel  black, white;
        Display *display;
        Screen *screen;
        XVisualInfo info;
        XmFontList fontList;
        
        
	w = ( Widget) ptr->stwidget;
	display = XtDisplay(w);
	screen = XtScreen(w); 
/*
* Check that first if this is a selected node..
*/
	window = ptr->window;
	selected = 0;
	itrsel = 0;
	if (window->selectedTrack != NO_TRACK) {
	   list = ptr->stdIndex;
	   for (i=0; i<ptr->multiplicity; i++, list++)
	     if (*list == window->selectedTrack) {
	       selected = 1;
	       break; 
	       }
	   }
	   
	
/*
* Place the number of particles, first part the Label.. Then convert 
* the particle id to a the multifont name, then to Xm muilt font string..
* Complete the label
*/	
	if ( ptr->multiplicity == -1){
	  sprintf(strnum, "Top");
	  xminfo =  XmStringLtoRCreate(strnum, XmSTRING_DEFAULT_CHARSET);
   	  icol = 11;
   	  id = 0;
	  }
	else {
	  sprintf(strnum, " %d ", ptr->multiplicity);
  	  xstrnum =  XmStringLtoRCreate(strnum, XmSTRING_DEFAULT_CHARSET);  	
          hepnam_(&ptr->stdid,nampart, MAXCHAR_HEPNAM );
          /* If greek font is available, use greek letters in particle name */
             GET_ONE_RSRC(w, XmNfontList, &fontList);
             if (CharsetAvailable("greek", fontList)) {
    	         hepnmg_(&ptr->stdid, nampart );
    	         }
    	          else {
                    hepnam_(&ptr->stdid,nampart, MAXCHAR_HEPNAM );
                    *(char *)strchr(nampart, ' ') = '\0';
    	         }
          xnampart = MultiFontString(nampart);
          xminfo = XmStringConcat(xstrnum, xnampart);  
	  XmStringFree (xnampart);
	  XmStringFree (xstrnum);
/* 
* Set the color of the button...
*/
	  id = ptr->stdid;     
   	  icol = ParticleColorIndex(id);
   	  }
/* 
*  Load this crap...
*/ 
        n = 0;
        if (selected == 1) {
          window->selectedNode = (int *) ptr;
          XtSetArg(wargs[n], XmNshadowThickness, 5); n++;
        }
        
  	XtSetArg(wargs[n], XmNlabelString, xminfo); n++; 
        if (XMatchVisualInfo(display, XScreenNumberOfScreen(screen),
	    DefaultDepthOfScreen(screen), PseudoColor, &info)) {
 	     XtSetArg(wargs[n], XmNbackground, ParticleColors[icol] ); n++; 
             black = BlackPixelOfScreen(screen);
             if (ParticleColors[icol] == black) {
	       white = WhitePixelOfScreen(screen);
               XtSetArg(wargs[n], XmNforeground, white); n++;
            }
         } 
  	XtSetValues(w, wargs, n);  
	XmStringFree (xminfo);
/*
* Now add the pointer to the node to the argument forthe call back routine.
*/
	XtAddCallback(w, XmNactivateCallback,
	              (XtCallbackProc)  listnodeCB, ptr);

}
   
static void listnodeCB(Widget w, nodehep *ptr, caddr_t callData)
{
	StdHepWindow *wim;
	int i, j, im, *l, *lp;
	PhaseParticle *p, *pm;
	int nn, takeit;

/* Do nothin if the user click on the tree top.. */
	
	wim = ptr->window;
	if ((nodehep *) wim->treehead == ptr) return;
	
        if (wim->modetreedisp == TREEDISPREAL) {
	  if (wim->selnodeTracks != NULL) XtFree((char *) wim->selnodeTracks);
	  lp = ptr->stdIndex; 
	  if (wim->highlightTrackMode == TRACK_NODE) { 
	    l = (int *) XtMalloc(sizeof(int) * ptr->multiplicity);
	    wim->selnodeTracks = l;
	    for (i=0; i < ptr->multiplicity; i++, l++, lp++)  *l = *lp;
	    wim->selnodeNumTrack = ptr->multiplicity;
	    
	  } else if (wim->highlightTrackMode == TRACK_DESCENDANTS) {
	    /* This is a bit wasteful in memory usage, but faster.. */
	    l = (int *) XtMalloc(sizeof(int) * wim->event.nParticles);
	    wim->selnodeTracks = l;
	    p = wim->event.particles;
	    nn = 0;
	    for (i=0; i < wim->event.nParticles; i++, p++) {
	      takeit = False;
	      pm = p;
	      while ((!takeit) && (pm->mother != 0)) {
	         if ((pm->mother - 1) == *lp) takeit = True;
	         im = (pm->mother) - 1;
	         pm = wim->event.particles; pm += im;
	         } 
	         if (takeit) { 
	           nn++; 
	           *l = i;
	           l++;
	         }
	    }
	    wim->selnodeNumTrack = nn;
	    
	  } else if (wim->highlightTrackMode == TRACK_DAUGHTERS) {
	    /* This is a bit wasteful in memory usage, but faster.. */
	    l = (int *) XtMalloc(sizeof(int) * wim->event.nParticles);
	    wim->selnodeTracks = l;
	    p = wim->event.particles;
	    nn = 0;
	    for (i=0; i < wim->event.nParticles; i++, p++) {
	      if (( p->mother - 1) == *lp ) { 
	         nn++; 
	         *l = i;
	         l++;
	         }
	    }
	    wim->selnodeNumTrack = nn;
	    
	    
	  } else if (wim->highlightTrackMode == TRACK_ANCESTORS) {
	    /* This is a bit wasteful in memory usage, but faster.. */
	    l = (int *) XtMalloc(sizeof(int) * wim->event.nParticles);
	    wim->selnodeTracks = l;
	    nn = 0;
	    p = wim->event.particles;
	    p += *lp;
	    pm = p;
	    while (pm->mother != NULL) {
	           im = (pm->mother) - 1; 
	           pm = wim->event.particles; pm += im;
	           nn++; 
	           *l = im;
	           l++;
	    }
	    wim->selnodeNumTrack = nn;
	    
	  } else if (wim->highlightTrackMode == TRACK_MOTHER) {
	    p = wim->event.particles;
	    p += *lp;
	    if (p->mother == NULL) {
	      wim->selnodeTracks = NULL;
	       wim->selnodeNumTrack = 0;
	      } else {
	       im = (p->mother) -1;
	       wim->selnodeTracks = (int *) XtMalloc(sizeof(int));
	       *wim->selnodeTracks = im;
	       wim->selnodeNumTrack = 1;
	     }
	  }   
	  DrawEvent(wim, FALSE);
          ListNodeHep(wim->dtree, ptr); 
         } else ListNodeHep(wim->dtreecolorcode, ptr);
} 

static void closeCB(Widget w, StdHepWindow *window)
{
	CloseHepTreeWindow(w, window);
}	

void CloseHepTreeWindow(Widget w, StdHepWindow *window)
{
	nodehep *head;

/* Set the selected Node to null and redraw the event to remore the 
	possible highlights..*/
	
	window->selectedNode = NULL;
	window->selnodeNumTrack = 0;
	if (window->selnodeTracks != NULL) 
	    XtFree((char *) window->selnodeTracks);
	window->selnodeTracks = NULL;
	DrawEvent(window, FALSE);

	head = (nodehep *) window->treehead;
	if (head != NULL)  {
	    FreeTree(head);
	    window->treehead  = NULL;
	    }
	window->selectedNode = NULL;
	if (window->dtree != NULL) XtDestroyWidget(window->dtree);
	window->nodeWindowShell = NULL;
	window->dtree = NULL;
	window->stree = NULL;
	window->tree = NULL;
	return;
} 
	
static void closecolorCB(Widget w, StdHepWindow *window)
{
	XtDestroyWidget(window->dtreecolorcode);
	window->dtreecolorcode = NULL;
	window->streecolorcode = NULL;
	window->treecolorcode = NULL;
	return;
} 
	
