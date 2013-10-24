/*******************************************************************************
*									       *
* mcf_NTuBldMenu.c -- Stuff for the usual menu bar on top of the main window   *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <sys/param.h>
#include <limits.h>
#include <time.h>
#include <Xm/Xm.h>
#include <Xm/RowColumn.h>
#include <Xm/Text.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#if XmVersion == 1002
#include <Xm/PrimitiveP.h>
#endif
#include "misc.h"
#include "DialogF.h"
#include "help.h"
#include "getfiles.h"
#include "mcf_nTupleDescript.h"
#include "mcf_nTupleBuild.h"
#include "mcf_NTuBuildWindow.h"
#include "mcf_NTuBldMenu.h"
#include "mcf_NTuBldHelp.h"
#include "mcf_NTuBldFiles.h"
#include "mcf_NTuIOFiles.h"

extern nTuBuildWindow *NTuBuildWindowList;
extern char *VarTypesNamesF77[N_VAR_TYPES];
extern char *VarTypesNamesC[N_VAR_TYPES];

static void exitCB(Widget w, Widget parent, caddr_t callData);
static void closeCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void fileOpenCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void saveCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void saveAsCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void newCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void cloneCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void generateFIncCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void generateCdothCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void generateDbinCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void setLangEnvCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
static void setOrgStyleCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
#ifdef HISTO
static void showOneDHistCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData);
#endif

static char *makeCommonName(nTuBuildWindow *window, int orgStyle);

/*
** Create the menu bar for Generalized NTuple Buil-windows
*/
Widget CreateNTuBldMenuBar(Widget parent, nTuBuildWindow *window)
{    

    Widget menuBar, menuPane, menuTmp;

    /* create the menu bar widget */
    menuBar = XmCreateMenuBar(parent, "menuBar", NULL, 0);
     
    /* Create the File menu */
    menuPane = AddSubMenu(menuBar, "fileMenu", "File", 'F');
    if (window->isReadOnly == False) {
       AddMenuItem(menuPane, "open", "Open ddl File...", 'O', "Ctrl<Key>o",
    	    "Ctrl O", (XtCallbackProc)fileOpenCB, window);
       AddMenuSeparator(menuPane, "s0");
       AddMenuItem(menuPane, "newStruct", "New Window", 'N', "Ctrl<Key>n",
    	    "Ctrl N", (XtCallbackProc)newCB, window);
       AddMenuItem(menuPane, "cloneWin", "Clone this Window", 'A', NULL,
    	    "", (XtCallbackProc)cloneCB, window);
       AddMenuSeparator(menuPane, "s1");
       window->saveItem = AddMenuItem(menuPane, "save", "Save...", 'A',
    	    NULL, "", (XtCallbackProc)saveCB, window);
       XtSetSensitive(window->saveItem, False);
    }   
    window->saveAsItem = AddMenuItem(menuPane, "saveAs", "Save As...", 'A',
    	    NULL, "", (XtCallbackProc)saveAsCB, window);
    XtSetSensitive(window->saveAsItem, False);
    window->generateDbin = 
        AddMenuItem(menuPane, "GenDbin", "Generate ddl Template",
            'A', NULL,
    	    "", (XtCallbackProc)generateDbinCB, window);
    window->generateF77 = 
        AddMenuItem(menuPane, "GenInc", "Generate F77 Include",
            'I', "Ctrl<Key>i",
    	    "Ctrl I", (XtCallbackProc)generateFIncCB, window);
    window->generateC = 
        AddMenuItem(menuPane, "GenCdoth", "Generate C Include",
            'H', "Ctrl<Key>h",
    	    "Ctrl H", (XtCallbackProc)generateCdothCB, window);
    XtSetSensitive(window->generateF77, False);
    XtSetSensitive(window->generateC, False);
    XtSetSensitive(window->generateDbin, False);
    AddMenuSeparator(menuPane, "s2");
    window->closeItem = AddMenuItem(menuPane, "close", "Close", 'C',
    	    "Ctrl<Key>w", "Ctrl W", (XtCallbackProc)closeCB, window);
    AddMenuItem(menuPane, "exit", "Exit", 'E', "<Key>F3:",
    	    "F3", (XtCallbackProc)exitCB, window);
    /*
    ** Add the histogramming utility is called from the Browser or 
    ** Analyser. 
    */	    
    if (window->isReadOnly == True) {
        menuPane = AddSubMenu(menuBar, "analyzeMenu", "Analysis", 'A');
#ifdef HISTO        
        menuTmp = AddMenuItem(menuPane, "on1DHistMainM", 
             "Define 1D histograms...", 'A', NULL,
    	    "", (XtCallbackProc)showOneDHistCB, NULL);
#endif    	    
    }
    /* Create the Preferences menu */
    menuPane = AddSubMenu(menuBar, "preferencesMenu", "Preferences", 'P');
       AddMenuItem(menuPane, "setLangEnv", "Set Language Env...", 'A', NULL,
    	    "", (XtCallbackProc)setLangEnvCB, window);
        
    if (window->isReadOnly == False) 
       AddMenuItem(menuPane, "setOrgStyle",
            "Set Data Strct. Organisation...", 'A', NULL,
    	    "", (XtCallbackProc)setOrgStyleCB, window);
    	        
    /* Create the Help menu */
    CreateHelpPulldownMenu(menuBar, NTuBldMenuHelp);

    /* Add close callback to the window border menu in addition to File menu */
    AddMotifCloseCallback(window->shell, (XtCallbackProc)closeCB, window);

    XtManageChild(menuBar);
    return menuBar;
}
static void exitCB(Widget w, Widget parent, caddr_t callData)
{
    while (NTuBuildWindowList != NULL) 
       if ( CloseNTuBuildWindow(NTuBuildWindowList) == GFN_CANCEL) return ;
}    
static void fileOpenCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData)
{
   nTuBuildWindow *currentWindow;
   char *ffname, title[80], *text;
   
  ffname = (char *) malloc(FILENAME_MAX*sizeof(char));
  if (GetExistingFilename (wInfo->shell,
                       "ntuBuild ddl File Name:",ffname ) 
                         == GFN_CANCEL) {
                            free(ffname);
                            return;
                         }
   /*
   ** Check that the window is reasonably empty..,
   **  if need be, create a new window 
   */
    text = XmTextGetString(wInfo->titleW);
    if (text[0] != '\0') wInfo->titleBlank = False;
    XtFree(text);
    if(wInfo->titleBlank || wInfo->multiplicityBlank ||
       wInfo->nameIndexBlank) currentWindow = wInfo;
     else {
        sprintf(title, "NTuple Builder, window number %d", (wInfo->id + 1));
        currentWindow = CreateNTuBuildWindow(XtDisplay(wInfo->shell),
                                               title, False);
    }
    GetFileNTuBuildWindow(currentWindow, ffname);
    XtVaSetValues(currentWindow->shell, XmNtitle, ffname, 0);

    free(ffname);                                           
                                                
}
static void closeCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData)
{
    CloseNTuBuildWindow(wInfo);
}    
static void saveCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData)
{
   if (VerifyStruct(wInfo, True) != True) return;
   if (wInfo->dbinFileName == NULL) {
        wInfo->dbinFileName = (char *) malloc(FILENAME_MAX*sizeof(char));
        if (GetNewFilename (wInfo->shell, "dbin/ntuBuild File Name:",
                wInfo->dbinFileName ) == GFN_CANCEL) {
            free(wInfo->dbinFileName);
            wInfo->dbinFileName = NULL;
            return;
        }
   }
   SaveFileNTuBuildWindow(wInfo);
   wInfo->isSaved = True;    
}
static void saveAsCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData)
{
   if (VerifyStruct(wInfo, True) != True) return;
    if (wInfo->dbinFileName != NULL) {
         free(wInfo->dbinFileName);
         wInfo->dbinFileName = NULL;
    }
    wInfo->dbinFileName = (char *) malloc(FILENAME_MAX*sizeof(char));
    if (GetNewFilename (wInfo->shell, "dbin/ntuBuild File Name:",
                wInfo->dbinFileName ) == GFN_CANCEL) {
        free(wInfo->dbinFileName);
        wInfo->dbinFileName = NULL;
        return;
   }
   SaveFileNTuBuildWindow(wInfo);    
   wInfo->isSaved = True;    
}
static void newCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData)
{
    char title[80];
    
    sprintf(title, "NTuple Builder, window number %d", (wInfo->id + 1));
    CreateNTuBuildWindow(XtDisplay(wInfo->shell), title, False);
}    
static void cloneCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData)
{
    char title[80];
    nTuBuildWindow  *newWindow;

    sprintf(title, "NTuple Builder, window number %d", (wInfo->id + 1));
    newWindow = 
         CreateNTuBuildWindow(XtDisplay(wInfo->shell), title, False);
    CloneNTuBuildWindow(wInfo, newWindow);
}

static void generateFIncCB(Widget w, nTuBuildWindow *window, caddr_t callData)
{
    char *nameCom, line[FILENAME_MAX+500], *tmp, *version, *text, *tc, *tc2;
    char *nameEffCom, shortNameCom[24], nameMaxIndex[32], nameTmpIndex[32];
    int i, j, l, kmode, nc, ncTot, nl, iv;
    time_t clock;
    FILE *Ffp;
    varGenNtuple *var;
    descrGenNtuple *dNTu = window->descrNtu;
    
    if (window->isSaved != True) {
            DialogF(DF_ERR, window->shell, 1,
         " You first must save this DDL! (Save pull down menu above)",
         "Acknowledgged" );
          return;
    }
    if (VerifyStruct(window, True) != True) return;
    nameCom = makeCommonName(window, dNTu->orgStyle);
    if (nameCom == NULL) {
        DialogF(DF_ERR, window->shell, 1,
         "The Title/Name can not be unambiguously transformed\n\
to a valid FORTRAN name, please change title", "Acknowledged");
        return;
    }
    strcpy(line, window->dbinFileName);
    tc = strchr(line, '.');
    if (tc == NULL) {
         l = strlen(window->dbinFileName);
         tc = line; tc+=l;
    }     
    strcpy(tc,".inc"); 
    if (GetNewFilename(window->shell, "FORTRAN Include file", 
                        line) == GFN_CANCEL) return;  
    Ffp = fopen(line, "w");
    fprintf(Ffp,"C     ntuBuild\n");
    time(&clock);
    tmp = line; sprintf(tmp,
                "C     Creation Date : %n", &l); tmp += l;
    strncpy(tmp,ctime(&clock), 24); tmp += 24; *tmp='\n'; tmp++; *tmp = '\0';
    fprintf(Ffp,line);
    fprintf(Ffp,"C     User Comments\n");
    text = NULL;
    text = XmTextGetString(window->descriptW);  tc = text; 
    if (*tc == '\0') 
       fprintf(Ffp,
                "C     No user comments\n");
    else {
       ncTot = strlen(tc); nc =0;
       while (nc < ncTot) {
            tc2 = strchr(tc,'\n');
            nl = (int) (tc2-tc)/sizeof(char);
            if ((tc2 == NULL) || (nl > 75)) nl = 75;
            strncpy(line, tc, nl); line[nl] = '\0';
            fprintf (Ffp,
                "C     %s\n", line);
            tc += nl; nc += nl;
            if (*tc == '\n') {
               tc++;
               nc++;
            }
       }
    }
    XtFree(text); text = NULL;
    /*
    ** Write the Variable description as Comments
    */
    fprintf(Ffp,"C    \n");
    fprintf(Ffp,"C     Variable description\n");
    fprintf(Ffp,"C    \n");
    for (i=0; i<dNTu->numAvailable; i++) {
        var = dNTu->variables[i];
        if (var->nameBlank != True) {
            if (var->description != NULL) {
               tc = (char *) malloc(sizeof(char) * 72); tc2 = tc;
               sprintf(tc,"%s: %n", var->name, &j); tc +=j;
               l = strlen(var->description); 
               if (l > (65 - j)) l = 65 - j;
               strncpy(tc, var->description, l); tc+=l;
               if (strlen(var->description) > (65 - j))
                    strcpy(tc, "... ");
               else       
                    strcpy(tc, " ");
               fprintf (Ffp, "C     %s\n", tc2); 
                   /*         123456 */
               free(tc2);
            }
         }
     }
    fprintf(Ffp,"C    \n");
    if (window->nameIndexBlank == True) { 
         sprintf(nameTmpIndex, "Idum_%s",nameCom);
         
    } else {
       text = XmTextGetString(window->nameIndexW);
       if (strcmp(text,"dummyIndex") == 0)
           sprintf(nameTmpIndex, "Idum_%s",nameCom);
       else strcpy(nameTmpIndex, text);
       XtFree(text); text = NULL;
    }   
    if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) 
        fprintf(Ffp,"      integer %s  \n", nameTmpIndex);
    l = strlen(nameTmpIndex); 
    if (l > 26) {
            strncpy(nameMaxIndex, nameTmpIndex, 26);
            sprintf(&nameMaxIndex[26],"_max");
    } else
            sprintf(nameMaxIndex, "%s_max", nameTmpIndex);
    fprintf(Ffp,"      integer %s  \n", nameMaxIndex );
    fprintf(Ffp,"      Parameter (%s = %d) \n", 
                          nameMaxIndex, dNTu->maxMultiplicity);
            
    version = XmTextGetString(window->versionW);
    if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) {
        if (strlen(nameCom) < 23)
            nameEffCom = nameCom; 
        else {
            strncpy(shortNameCom, nameCom, 23); shortNameCom[23] = '\0';
            nameEffCom = shortNameCom;
        }
    
        /*
        ** The first 64 bits contain the version token, as a char[8] string
        ** and the multiplicty variable.
        */
        fprintf(Ffp,"      character*8 version_%s \n", nameEffCom);
        /*           123456 */
        fprintf(Ffp,"      integer pad_%s \n", nameEffCom);
             
       for (iv=0; iv< dNTu->numVariables; iv++) {
           i = 0;
           while (dNTu->varOrdering[i] != iv) i++;
           var = dNTu->variables[i];
           kmode = 0; if (var->isFixedSize != True) kmode = 1;
           tc = line;
           sprintf(tc,"      %s  %n",
                        VarTypesNamesF77[var->type], &l);
           tc +=l;
           if ((var->numDim == 0) && (kmode ==0))
                  sprintf(tc,"  %s", var->name);
           else if (var->numDim == 0) 
                   sprintf(tc," %s(%s)", var->name, nameMaxIndex ); 
                      
           else {
             if (var->type != CHARACTER_NTU) {  
                  sprintf(tc," %s(%n",var->name, &l); tc+=l;
                  for (j=0; j<var->numDim; j++, tc+=l) {
                       if ((j == var->numDim-1) && (kmode == 0))
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                       else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                  }
                  if (kmode == 1) {
                      sprintf(tc, "%s%n", nameMaxIndex, &l);
                      tc +=l;
                  }
                  sprintf(tc,") "); 
              } else { /* Special handling for Character variables */
                  if (var->numDim == 1) {
                    sprintf(tc,"*%d %s%n", var->dimensions[0], var->name, &l);
                    tc +=l;
                    if (kmode == 1)  
                         sprintf(tc, "(%s)", nameMaxIndex);
                 } else {
                    sprintf(tc,"*%d %s(%n", var->dimensions[0], var->name, &l);
                    tc+=l;
                    for (j=1; j<var->numDim; j++, tc+=l) {
                       if ((j == var->numDim-1) && (kmode == 0))
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                       else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                     }
                     if (kmode == 1) {
                        sprintf(tc, "%s%n", nameMaxIndex, &l);
                        tc +=l;
                     }
                     sprintf(tc,") "); 
                  }
               }
             }   
             fprintf(Ffp,"%s\n", line);
       }
       fprintf(Ffp,"      integer fence_%s(2) \n", nameEffCom);
        /*
        ** Now compose the Common block itself..
        */    
       fprintf(Ffp,"      COMMON/%s/version_%s, \n",
                           nameCom,  nameEffCom);
       fprintf(Ffp,"     & %s, pad_%s,\n", nameTmpIndex, nameEffCom);
       tc = line;
       sprintf(tc, "     & %n", &l); tc +=l; nc = l;
       for (iv=0; iv< dNTu->numVariables; iv++) {
           i = 0;
           while (dNTu->varOrdering[i] != iv) i++;
           var = dNTu->variables[i];
           if ((nc + strlen(var->name)+ 2) > 71 ) {
              fprintf(Ffp, "%s\n", line); tc=line;
	      sprintf(tc, "     & %n", &l); tc +=l; nc = l;
           }
           sprintf(tc,"%s, %n", var->name, &l);
           tc +=l; nc +=l;
       }
       fprintf(Ffp, "%s\n", line);
       fprintf(Ffp, "     & fence_%s\n", nameEffCom); 
       fprintf(Ffp,"C      End\n");
       /*
       ** End of FORTRAN 77, COMMON block based implementation.
       */
     } else {
       /* 
       **  The other type of organisation, using structure.
       */
        if (strlen(nameCom) < 22)
            nameEffCom = nameCom; 
        else {
            strncpy(shortNameCom, nameCom, 21); shortNameCom[21] = '\0';
            nameEffCom = shortNameCom;
        }
    
        fprintf(Ffp, "      structure /%s_v_struct/ \n", nameEffCom);
       for (iv=0; iv< dNTu->numVariables; iv++) {
           i = 0;
           while (dNTu->varOrdering[i] != iv) i++;
           var = dNTu->variables[i];
           if (var->isFixedSize == False) {
                 tc = line;
                 sprintf(tc,"      %s  %n",
                        VarTypesNamesF77[var->type], &l);
                 tc +=l;            
                 if (var->numDim == 0) {
                    sprintf(tc," %s ", var->name); 
                 } else {
                    if (var->type != CHARACTER_NTU) { 
                      sprintf(tc," %s(%n",var->name, &l); tc+=l;
                      for (j=0; j<var->numDim; j++, tc+=l) {
                       if (j == var->numDim-1)
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                       else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                      }
                      sprintf(tc,")");
                   
                    } else {
                      sprintf(tc,"*%d %s%n",
                               var->dimensions[0], var->name, &l); tc+=l;
                      if (var->numDim > 1) { 
                        sprintf(tc,"(%n", &l); tc+=l;        
                        for (j=1; j<var->numDim; j++, tc+=l) {
                           if (j == var->numDim-1)
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                           else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                        }
                        sprintf(tc,")");     
                      }
                    }
                 }   
                 fprintf(Ffp,"      %s\n", line);
              }
         }
        fprintf(Ffp,"      end structure\n", line);
        /*
        ** the mother structure now
        */
        fprintf(Ffp, "C       \n");
        fprintf(Ffp, "      structure /%s_struct/ \n", nameEffCom);
        fprintf(Ffp, "          character*8 version\n");
        text = XmTextGetString(window->nameIndexW);
        strcpy(nameTmpIndex, text);
        fprintf(Ffp,"           integer %s  \n", text);
        fprintf(Ffp, "          integer pad_%s\n",nameEffCom); 
       for (iv=0; iv< dNTu->numVariables; iv++) {
           i = 0;
           while (dNTu->varOrdering[i] != iv) i++;
           var = dNTu->variables[i];
           if (var->isFixedSize == True) {
               tc = line;
               sprintf(tc,"      %s  %n",
                        VarTypesNamesF77[var->type], &l);
               tc +=l;            
               if (var->numDim == 0) {
                    sprintf(tc," %s ", var->name); 
               } else {
                    if (var->type != CHARACTER_NTU) { 
                      sprintf(tc," %s(%n",var->name, &l); tc+=l;
                      for (j=0; j<var->numDim; j++, tc+=l) {
                       if (j == var->numDim-1)
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                       else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                      }
                      sprintf(tc,")");
                    } else {
                      sprintf(tc,"*%d %s%n",
                               var->dimensions[0], var->name, &l); tc+=l;
                      if (var->numDim > 1) { 
                        sprintf(tc,"(%n", &l); tc+=l;        
                        for (j=1; j<var->numDim; j++, tc+=l) {
                           if (j == var->numDim-1)
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                           else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                        }
                        sprintf(tc,")");     
                      }
                    }
               }
               fprintf(Ffp,"      %s\n", line);
            }
        }
        fprintf(Ffp,"            record /%s_v_struct/ %s_var(%s)\n", 
                             nameEffCom, nameEffCom, nameMaxIndex);
        fprintf(Ffp,"            integer fence(2) \n");
        fprintf(Ffp,"      end structure\n", line);
        fprintf(Ffp,"C         \n");
        /*
        ** Now compose the Common block itself..
        */    
        fprintf(Ffp,"C         \n");
	fprintf(Ffp,"       record /%s_struct/ %s_c \n", 
	                            nameEffCom, nameEffCom);
        fprintf(Ffp,"C         \n");
        
        fprintf(Ffp,"      COMMON/%s/%s_c \n", nameCom,  nameEffCom);
        fprintf(Ffp,"C      End\n");
        
     }
    free(nameCom);
    fclose(Ffp);
    
}
static void generateCdothCB(Widget w, nTuBuildWindow *window, caddr_t callData)
{
    char *nameCom, line[FILENAME_MAX+500], *tc, *text;
    int i, l;
    varGenNtuple *var;
    descrGenNtuple *dNTu = window->descrNtu;
    
    if (window->isSaved != True) {
            DialogF(DF_ERR, window->shell, 1,
         " You first must save this DDL! (Save pull down menu above)",
         "Acknowledged" );
          return;
    }
    if (VerifyStruct(window, True) != True) return;
    /*
    ** verify that no unknown ddl data type are defined 
    */
    for (i=0; i<dNTu->numAvailable; i++) { 
        var = dNTu->variables[i];
        if ( (var->nameBlank != True)  && (var->type == COMPLEX_NTU)) 
            DialogF(DF_WARN, window->shell, 1,
         "Complex is a valid data type in ANSI C \n\
It will be replaced by float[2]", "Acknowledged" );
 
        if ( (var->nameBlank != True)  && (var->type == DBL_COMPLEX_NTU)) 
            DialogF(DF_WARN, window->shell, 1,
         "Double precision Complex is a valid data type in ANSI C \n\
It will be replaced by dbl[2]", "Acknowledged" );
        
    }
    nameCom = makeCommonName(window, dNTu->orgStyle);
    if (nameCom == NULL) {
        DialogF(DF_ERR, window->shell, 1,
         "The Title/Name can not be unambiguously transformed\n\
to a valid structure name, please change title", "Acknowledged");
        return;
    }
    strcpy(line, window->dbinFileName);
    tc = strchr(line, '.');
    if (tc == NULL) {
         l = strlen(window->dbinFileName);
         tc = line; tc+=l;
    }     
    strcpy(tc,".h");
    if (GetNewFilename(window->shell, "FORTRAN Include file", 
                        line) == GFN_CANCEL) return;  
    
    if (dNTu->title != NULL) free(dNTu->title);
    dNTu->title = XmTextGetString(window->titleW);
    
    if (dNTu->description != NULL) free(dNTu->description);
    dNTu->description = XmTextGetString(window->descriptW);
    text = XmTextGetString(window->versionW);
    strcpy(dNTu->version, text);
    if (window->nameIndexBlank == True) 
      strcpy(dNTu->nameIndex,"dummyIndex");
    else {
      text = XmTextGetString(window->nameIndexW);
      strcpy(dNTu->nameIndex, text);
      XtFree(text);
    }
    
    mcf_ComposeDoth(dNTu, line);
    free(nameCom);   
}
/*
** Saving a particular instance of this structure as dbin template.
*/
static void generateDbinCB(Widget w, nTuBuildWindow *window, caddr_t callData)
{
    char *nameCom, line[FILENAME_MAX+500], *tmp, *version, *text, *tc, *tc2;
    char nullDescr[4], *descrTmp;
    int i, j, l, kmode, nc, ncTot, nl, iv;
    time_t clock;
    FILE *Ffp;
    varGenNtuple *var;
    descrGenNtuple *dNTu = window->descrNtu;
    
    if (window->isSaved != True) {
            DialogF(DF_ERR, window->shell, 1,
         " You first must save this DDL! (Save pull down menu above)",
         "Acknowledged" );
          return;
    }
    if (VerifyStruct(window, True) != True) {
            DialogF(DF_ERR, window->shell, 1,
         " This Ntuple is badly defined ! (Please Report)",
         "Acknowledged" );
         return;
    }
    /*
    ** verify that no unknown dbin data type are defined 
    */
    for (i=0; i<dNTu->numAvailable; i++) { 
        var = dNTu->variables[i];
        if ( (var->nameBlank != True)  &&
           ((var->type != INTEGER_NTU) && (var->type != REAL_NTU) &&
            (var->type != DBL_PRECISION_NTU))) { 
            DialogF(DF_ERR, window->shell, 1,
         "Variable type %s is not supported by dbin\n\
this data structure can not be saved as dbin template", "Acknowledged", 
          VarTypesNamesF77[var->type]);
          return;
        }
    }
    nameCom = makeCommonName(window, dNTu->orgStyle);
    if (nameCom == NULL) {
        DialogF(DF_ERR, window->shell, 1,
         "The Title/Name can not be unambiguously transformed\n\
to a valid structure name, please change title", "Acknowledged");
        return;
    }
    strcpy(nullDescr, "? ");
    strcpy(line, window->dbinFileName);
    tc = strchr(line, '.');
    if (tc == NULL) {
         l = strlen(window->dbinFileName);
         tc = line; tc+=l;
    }     
    strcpy(tc,"_template.db");   
    Ffp = fopen(line, "w");
    fprintf(Ffp,"# ntuBuild\n");
    time(&clock);
    tmp = line; sprintf(tmp,"#Creation Date : %n", &l); tmp += l;
    strncpy(tmp,ctime(&clock), 24); tmp += 24; *tmp='\n'; tmp++; *tmp = '\0';
    fprintf(Ffp,line);
    fprintf(Ffp,"# User Comments\n");
    text = NULL;
    text = XmTextGetString(window->descriptW);
    tc = text; 
    if (*tc == '\0') 
       fprintf(Ffp,"# no user comments\n");
    else {
       ncTot = strlen(tc); nc =0;
       while (nc < ncTot) {
            tc2 = strchr(tc,'\n');
            nl = (int) (tc2-tc)/sizeof(char);
            if ((tc2 == NULL) || (nl > 75)) nl = 75;
            strncpy(line, tc, nl); line[nl] = '\0';
            fprintf (Ffp,"# %s\n", line);
            tc += nl; nc += nl;
            if (*tc == '\n') {
               tc++;
               nc++;
            }
       }
    }
    XtFree(text); text = NULL;
    
    fprintf(Ffp, "mode internal\n");
    version = XmTextGetString(window->versionW);
    
    if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) {
        fprintf(Ffp, "incname %s_struct \n", nameCom);
        fprintf(Ffp, "template %s \n", nameCom);
        /*
        ** The first 64 bits contain the version token, as a char[8] string
        ** Then a padding word (integer)
        ** Unfortunatly, this can not put in a dbin template. 
        ** Cheat for now, place it as a integer... (Not real good!)
        */ 
        fprintf(Ffp,"    int iversion(2) !Version token \n");
        text = XmTextGetString(window->nameIndexW);
        fprintf(Ffp,"    int %s !Generalized Ntuple Multiplicity value \n",
                                                                      text);
                                                                      
        fprintf(Ffp,"    int pad_%s !Padding to support 64 bit archit. \n",
                                                                      text);
        XtFree(text); text = NULL;
        for (iv=0; iv< dNTu->numVariables; iv++) {
           i = 0;
           while (dNTu->varOrdering[i] != iv) i++;
           var = dNTu->variables[i];
           kmode = 0; if (var->isFixedSize != True) kmode = 1;
           if (var->description == NULL) descrTmp = nullDescr;
                  else descrTmp = var->description;
           tc = line;
           switch(var->type) {
                   case (INTEGER_NTU):                   
                     sprintf(tc,"    int %n", &l);
                     break;
                   case (REAL_NTU):
                     sprintf(tc,"    real %n", &l);
                     break;
                   case (DBL_PRECISION_NTU):
                     sprintf(tc,"    double %n", &l);
                     break;
                   default:
                     break; /* ain't supposed to happen, see above */
            }
            tc +=l;            
            if ((var->numDim == 0) && (kmode ==0)) sprintf(tc," %s !%s",
                                              var->name, descrTmp);
            else if (var->numDim == 0) {
                 sprintf(tc," %s(%d) !%s",
                        var->name, dNTu->maxMultiplicity, descrTmp); 
            } else { 
                  sprintf(tc," %s(%n",var->name, &l); tc+=l;
                  for (j=0; j<var->numDim; j++, tc+=l) {
                       if ((j == var->numDim-1) && (kmode == 0))
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                       else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                  }
                  if (kmode == 1) {
                      sprintf(tc, "%d%n", dNTu->maxMultiplicity, &l);
                      tc +=l;
                  }    
                  sprintf (tc,") !%s", descrTmp);
            }
            fprintf(Ffp,"%s\n", line);
          }
        fprintf(Ffp," int fence_%s", nameCom);
        fprintf(Ffp,"end template\n", line);
        fprintf(Ffp,"# end \n", line);
     }else { /* the other type of organisation, using structure */
        fprintf(Ffp, "incname %s_v_struct \n", nameCom);
        fprintf(Ffp, "template %s_v \n", nameCom);
        for (iv=0; iv< dNTu->numVariables; iv++) {
           i = 0;
           while (dNTu->varOrdering[i] != iv) i++;
           var = dNTu->variables[i];
           if (var->isFixedSize == False) {
               tc = line;
               if (var->description == NULL) descrTmp = nullDescr;
                    else descrTmp = var->description;
               switch(var->type) {
                   case (INTEGER_NTU):                   
                     sprintf(tc,"    int %n", &l);
                     break;
                   case (REAL_NTU):
                     sprintf(tc,"    real %n", &l);
                     break;
                   case (DBL_PRECISION_NTU):
                     sprintf(tc,"    double %n", &l);
                     break;
                   default:
                     break; 
               }
               tc +=l;            
               if (var->numDim == 0) {
                    sprintf(tc," %s !%s",
                        var->name, descrTmp); 
               } else { 
                    sprintf(tc," %s(%n",var->name, &l); tc+=l;
                    for (j=0; j<var->numDim; j++, tc+=l) {
                       if (j == var->numDim-1)
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                       else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                    }
                    sprintf(tc, ") !%s", descrTmp);
               }
               fprintf(Ffp,"%s\n", line);
            }
        }
        fprintf(Ffp,"end template\n", line);
        fprintf(Ffp,"#  \n", line);
        /*
        ** the mother structure now
        */
        fprintf(Ffp, "incname %s_struct \n", nameCom);
        fprintf(Ffp, "template %s \n", nameCom);
        fprintf(Ffp,"    int iversion(3) !Version token \n");
        for (iv=0; iv< dNTu->numVariables; iv++) {
           i = 0;
           while (dNTu->varOrdering[i] != iv) i++;
           var = dNTu->variables[i];
           if (var->isFixedSize == True) {
               if (var->description == NULL) descrTmp = nullDescr;
                    else descrTmp = var->description;
               tc = line;
               switch(var->type) {
                   case (INTEGER_NTU):                   
                     sprintf(tc,"    int %n", &l);
                     break;
                   case (REAL_NTU):
                     sprintf(tc,"    real %n", &l);
                     break;
                   case (DBL_PRECISION_NTU):
                     sprintf(tc,"    double %n", &l);
                     break;
                   default:
                     break;
               }
               tc +=l;            
               if (var->numDim == 0) {
                sprintf(tc," %s !%s", var->name, descrTmp); 
               } else { 
                    sprintf(tc," %s(%n",var->name, &l); tc+=l;
                    for (j=0; j<var->numDim; j++, tc+=l) { 
                       if (j == var->numDim-1)
                             sprintf(tc,"%d%n", var->dimensions[j], &l); 
                       else                                 
                             sprintf(tc,"%d,%n", var->dimensions[j], &l);
                    }           
                    sprintf(tc, ") !%s", descrTmp);
                } 
                fprintf(Ffp,"%s\n", line);
             }
          }
          fprintf(Ffp,"provide %s_v_struct \n", nameCom);
          fprintf(Ffp," dimension %s_max %d \n", 
                                 nameCom, dNTu->maxMultiplicity);
          fprintf(Ffp," record %s_v (%s_max) \n", 
                                 nameCom, nameCom);                          
          fprintf(Ffp," int fence_%s", nameCom);
          fprintf(Ffp,"#  \n", line);
     }
    free(nameCom);
    fprintf(Ffp,"#  \n", line);
    fprintf(Ffp,"end \n");
    fclose(Ffp);
    
}
static void setLangEnvCB(Widget w, nTuBuildWindow *window, caddr_t callData)
{
    int i, answer;
    XmString s1;
    
    answer  = DialogF(DF_QUES, window->shell, 2, 
"Array and Variable names are obviously language sensitive\n\
Do you wish Fortran or C syntax ?", "FORTRAN", "C");
    window->langEnv = (answer-1);
    for (i=0; i<N_VAR_TYPES; i++) {
        if (window->langEnv == F77_LANGUAGE )
            XtVaSetValues(window->typeBtns[i], 
                      XmNlabelString, 
                      s1=XmStringCreateSimple(VarTypesNamesF77[i]), 0);
        else 
            XtVaSetValues(window->typeBtns[i], 
                      XmNlabelString, 
                      s1=XmStringCreateSimple(VarTypesNamesC[i]), 0);
        XmStringFree(s1);
    }
    UpdateVariableList(window, window->selectedIndex);
 
}             

static void setOrgStyleCB(Widget w, nTuBuildWindow *window, caddr_t callData)
{
    int resp;  
     resp = DialogF(DF_QUES, window->shell, 2, 
"There are two ways of organizing a particular instance \n\
of this type of Data Structure, either as // arrays, suitable \n\
in F77 COMMON blocks or VAX FORTRAN data structure, easier for\n\
 C writer ", "F77 COMMON", "VAX FORTRAN D/S");
    window->descrNtu->orgStyle = resp - 1;
    /*
    ** Check if this structure is selfconsistent
    */
    if (VerifyStruct(window, True) == True) {
        XtSetSensitive(window->saveItem, True);
        XtSetSensitive(window->saveAsItem, True);
    }
    XtSetSensitive(window->generateF77, False);
    XtSetSensitive(window->generateC, False);
    XtSetSensitive(window->generateDbin, False);
    window->hasChanged = True;
} 

#ifdef HISTO
static void showOneDHistCB(Widget w, nTuBuildWindow *wInfo, caddr_t callData)
{
   mcfioC_ShowBrowserOneDHist();
}
#endif            
static char *makeCommonName(nTuBuildWindow *window, int orgStyle)
{
    char *text, *out;
    int i, l, nMax;
    
    text = NULL;
    text = XmTextGetString(window->titleW);
    l = strlen(text);
    nMax = 22; /* Shorten the length for index names */
    if (l > nMax) l = nMax;
    out = (char *) malloc(sizeof(char) * (l+1));
    strncpy(out, text, l); out[l]='\0';
    XtFree(text); 
    
    for (i=0; i<l; i++) if (out[i] == ' ') out[i] == '_';
    if (VerifyVarGenName(out) == True) return out;
    else{ 
       free(out);
       return NULL;
    } 
}       
/*
** Verify that a String will be accepted by a reasonable compiler
*/    
int VerifyVarGenName(char *string) 
{
    static char validChars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
abcdefghijklmnopqrstuvwxyz1234567890_";
    static char invalidFirstChars[] = "1234567890";
    int i, ll;

    if (string == NULL) return False;
    if (string[0] == '\0') return False;
    ll = strlen(string);
    for (i=0; i<ll; i++) 
        if (strchr(validChars, string[i]) == NULL) return False;
    if (strchr(invalidFirstChars, string[0]) != NULL) return False;
    return True;
}       

