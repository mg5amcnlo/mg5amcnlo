/*******************************************************************************
*									       *
* mcf_NTuBldFiles.c -- Utilities to manipulate file within ntuBuild stuff      *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <limits.h>
#include <time.h>
#include <Xm/Xm.h>
#include <Xm/Text.h>
#include "DialogF.h"
#include "mcf_nTupleDescript.h"
#include "mcf_nTupleBuild.h"
#include "mcf_NTuBldFiles.h"
#include "mcf_ntubld_db.h"


extern struct line_title_c line_title_c_;
extern struct header_c header_c_;
extern struct variable_c variable_c_;

extern char *VarTypesNamesF77[N_VAR_TYPES];

int GetFileNTuBuildWindow(nTuBuildWindow *window, char* filename)
{
    descrGenNtuple *dNTu = window->descrNtu;
    int i, j;
    char *text, *tc;
    varGenNtuple *varTmp;
    
    
    header_c_.n_obj_header = 0;
    line_title_c_.n_obj_line_title = 0;
    mcf_ntubldRead(filename);
    if ((line_title_c_.n_obj_line_title < 1)  ||
        (header_c_.n_obj_header != 1)) {
        DialogF(DF_ERR, window->shell, 1,
                " This file was not created by this application!",
                 "Acknowldged");
        return 1;
    }
    if (strcmp(line_title_c_.line_title[0].line,
               "ntuBuild Database, v1.0") != 0) {
        DialogF(DF_ERR, window->shell, 1,
                " This file was created by a wrong version of ntuBuild!",
                 "Acknowldged");
        return 2;
    }
    /*
    ** There are 80 character per lines in dbin..
    */
    text = (char *)
            malloc(sizeof(char) * 80 * (line_title_c_.n_obj_line_title -1));
    for (i=1, tc=text; i<line_title_c_.n_obj_line_title; i++) {
        strcpy(tc, line_title_c_.line_title[i].line); 
        tc += strlen(line_title_c_.line_title[i].line);
        *tc = '\n'; tc++;
    }
    *tc = '\0';
    XmTextSetString(window->descriptW, text);
    XmTextSetString(window->titleW, header_c_.header[0].title);
    XmTextSetString(window->versionW, header_c_.header[0].version);
    XmTextSetString(window->nameIndexW, header_c_.header[0].namemaxindex);
    SetIntText(window->multiplicityW, header_c_.header[0].maxmult);
    dNTu->orgStyle = header_c_.header[0].orgstyle;
    dNTu->numVariables = header_c_.header[0].nvar;
    while  (dNTu->numAvailable < dNTu->numVariables)
            ExtendVariableList(window);
    /*
    ** Now the variables
    */
    for (i=0; i<variable_c_.n_obj_variable; i++) {
        varTmp = dNTu->variables[i];
        varTmp->nameBlank = False;
        if (varTmp->name != NULL) free(varTmp->name);
        varTmp->name = (char *)
            malloc(sizeof(char) * (strlen(variable_c_.variable[i].name) + 1));
        strcpy(varTmp->name, variable_c_.variable[i].name);
        
        if (varTmp->description != NULL) free(varTmp->description);
        if ((strlen(variable_c_.variable[i].description) > 1) ||
                    variable_c_.variable[i].description[0] != ' ') { 
           varTmp->description = (char *) malloc(sizeof(char) * 
                (strlen(variable_c_.variable[i].description) + 1));
           strcpy(varTmp->description, variable_c_.variable[i].description);
        }   
        varTmp->type = variable_c_.variable[i].type;
        varTmp->isFixedSize = True;
        if (strncmp(variable_c_.variable[i].isfixedsize,"Yes",3))
            varTmp->isFixedSize = False;
        varTmp->numDim = variable_c_.variable[i].numdim;
        if (varTmp->numDim > 0) 
           for (j=0; j< varTmp->numDim; j++)
               varTmp->dimensions[j] = variable_c_.variable[i].dimensions[j];
                
    }    
    UpdateVariableList(window, 1);
    UpdateDialogVFields(window);
    /*
    ** This verification should not be needed at this point
    */
    if (VerifyStruct(window, True) == True) {
        window->isSaved = True;
        XtSetSensitive(window->generateF77, True);
        XtSetSensitive(window->generateC, True);
        XtSetSensitive(window->generateDbin, True);
        if (window->dbinFileName != NULL) free(window->dbinFileName);
        window->dbinFileName = (char *) 
                              malloc((strlen(filename)+1) * sizeof(char));
        strcpy(window->dbinFileName, filename);
    } else return 0;
    /*
    ** Set the ordering. Trivial in this case, it has been ordered in 
    ** the save routine.
    */
    dNTu->varOrdering = (int *) malloc(sizeof(int) * dNTu->numAvailable);
    for (i=0; i<dNTu->numAvailable; i++)
       dNTu->varOrdering[i] = dNTu->numAvailable + 1; 
    
    for (i=0; i<dNTu->numVariables; i++)
       dNTu->varOrdering[i] = i;
    return 0; 
}
void SaveFileNTuBuildWindow(nTuBuildWindow *window) 
{
    FILE *Ffp;
    int i,j, k, kmode, itype, l, nl, dd[5], ncTot, nc, nVar, iv, iSpace, nm;
    size_t sizeV;
    char *text, *tc, *tc2, *env, line[255];
    descrGenNtuple *dNTu;
    varGenNtuple **varGenOrdered;
    varGenNtuple *varTmp, *var1, *var2, *varEnd;
    time_t clock;
    
    dNTu = window->descrNtu;
    env = NULL;
    env = getenv("MCFIO_DIR");
    /*
    ** Compute the variable ordering: Rule: if parallel array, first 
    ** fixed size, then all the indexed ones. within such groups, we order 
    ** variable by types.  If substrutures, same idea, variable are ordered
    ** by types. Blank variables are placed last
    */
    if (dNTu->varOrdering != NULL) free(dNTu->varOrdering);
    dNTu->varOrdering = (int *) malloc(sizeof(int) * dNTu->numAvailable);
    for (i=0; i<dNTu->numAvailable; i++)
       dNTu->varOrdering[i] = dNTu->numAvailable + 1; 
    for (itype=0, l=0; itype<N_VAR_TYPES; itype++) {
      for (i=0; i<dNTu->numAvailable; i++) {
        varTmp = dNTu->variables[i];
        if ((varTmp->nameBlank != True) && (varTmp->type == itype) &&
                (varTmp->isFixedSize == True)) {
                       dNTu->varOrdering[i] = l; l++; 
                }
          }
    }
    for (itype=0; itype<N_VAR_TYPES; itype++) {
      for (i=0; i<dNTu->numAvailable; i++) {
        varTmp = dNTu->variables[i];
        if ((varTmp->nameBlank != True) && (varTmp->type == itype) &&
                (varTmp->isFixedSize == False)) {
                       dNTu->varOrdering[i] = l; l++;
                }
          }
    }
    nVar = l;
    /*
    ** Display this new order, so the user knows what he is saving..
    ** recreate the variables structure, it is easier...
    */
    varGenOrdered =
       (varGenNtuple **) malloc(sizeof(varGenNtuple *) * nVar);
    for (iv=0; iv< nVar; iv++) {
       for (j=0; j<dNTu->numAvailable; j++)  
          if (dNTu->varOrdering[j] == iv)  i = j; 
       varTmp = dNTu->variables[i];
       varGenOrdered[iv] = (varGenNtuple *) malloc(sizeof(varGenNtuple));
       varGenOrdered[iv]->name = NULL;
       varGenOrdered[iv]->description = NULL;
       CopyVarGenNtuple(varTmp, varGenOrdered[iv]);
    }
    for (j=0; j<dNTu->numAvailable; j++)  
            DestroyVarGenNtuple(dNTu->variables[j]);
    free(dNTu->variables);
    dNTu->variables = varGenOrdered;
    dNTu->numVariables = nVar;
    dNTu->numAvailable = nVar;
    for (iv=0; iv< nVar; iv++) dNTu->varOrdering[iv] = iv;
    UpdateVariableList(window, 0);
    /*
    ** Check the alignement of the structure. Variable must aligned on 
    ** a word boundary. We got 8+4 bytes offset due to the version and 
    ** the multiplicity value (which is an int ).  So we always add a 
    ** padding variable.  Note: ultimately, this
    ** machine dependant code, but seems to be ok for 32 bit to 64 
    ** bit conversion. One way to avoid this is to start the structure
    ** with the longest type, however, it still unsafe as we want
    ** to keep the variable part sepqarated from the arrays of 
    ** substrures.  
    */
    for (iv=0, iSpace=2*sizeof(int) + 8*sizeof(char); iv< nVar; iv++) {
        varTmp = dNTu->variables[iv];
        nm = 1;
        for (k=0; k<varTmp->numDim; k++) nm = nm * varTmp->dimensions[k];
        if ((dNTu->orgStyle == PARALLEL_ARRAY_NTU) && 
            (varTmp->isFixedSize == False)) nm = nm * dNTu->maxMultiplicity;
        switch (varTmp->type) {
           case BYTE_NTU: case CHARACTER_NTU:
                 sizeV = sizeof(char); /* This better be one byte */
                 break;
              case INTEGER2_NTU:
                 sizeV = sizeof(short);
                 break;
              case LOGICAL_NTU: case INTEGER_NTU:
                  sizeV = sizeof(int);
                  break;
              case REAL_NTU:
                  sizeV = sizeof(float);
                  break;
              case DBL_PRECISION_NTU:
                  sizeV = sizeof(double);
                  break;
              case COMPLEX_NTU:
                  sizeV = 2 * sizeof(float);
                  break;
              case DBL_COMPLEX_NTU:
                  sizeV = 2 * sizeof(double);
                  break;
              case POINTER_NTU:
                  sizeV = sizeof(void *); /* This different on 32 vs 64 bit */
                  break;
             default : 
                  fprintf(stderr, " mcf_ComputNTuLength, internal error \n");
                  sizeV = 0;
                  break;
          }
          if ((iSpace%((int) sizeV)) != 0) {
                 nc = iSpace%((int) sizeV);
                 switch (nc) {
                      case 2:
                         k = INTEGER2_NTU;
                         nl = 1;
                         break;
                      case 4:
                         k = INTEGER_NTU;
                         nl = 1;
                         break;
                      default:
                         k = BYTE_NTU;
                         nl = nc;
                         break;
                 }        
                 sprintf(line, " This structure is misaligned! \n\
 Variable %s does not start on a word boundary. \n\
 One must add %d bytes upstream of this variable. \n\
 Suggestion: insert %d %s at location %d.\n", varTmp->name,
                          nc, nl, VarTypesNamesF77[k], (iv+1));
    
                 DialogF(DF_ERR, window->shell, 1, 
                            line, "Acknowledged");
                 varEnd = dNTu->variables[dNTu->numAvailable-1];
                 if (varEnd->nameBlank != True) ExtendVariableList(window);
                 for (i=dNTu->numAvailable-1; i>iv; i--) {
                        var2 =  dNTu->variables[i];
                        var1 = dNTu->variables[i-1];
                        if ((var1->nameBlank != True) || 
                            (var2->nameBlank != True)) 
                               CopyVarGenNtuple(var1, var2);
                  }
                  varTmp = dNTu->variables[iv];
                  varTmp->nameBlank = True;
                  if (varTmp->name != NULL) {
                       free(varTmp->name); 
                       varTmp->name = NULL;
                  }
                  if (varTmp->description != NULL) {
                       free(varTmp->description); 
                       varTmp->description = NULL;
                  }
                  varTmp->type = k; 
                  varTmp->numDim = 0;    
                  XmListSelectPos(window->listW, (iv+1));
                  UpdateVariableList(window, 0);
                  return;

          }
          iSpace += (nm * sizeV);
    }                    
    Ffp = fopen(window->dbinFileName, "w");
    fprintf(Ffp,"# ntuBuild\n");
    time(&clock);
    tc = line; sprintf(tc,"#Creation Date : %n", &j); tc += j;
    strncpy(tc,ctime(&clock), 24); tc += 24; *tc='\n'; tc++; *tc = '\0';
    fprintf(Ffp,line);
    text = XmTextGetString(window->titleW); 
    strcpy(line, text);
    XtFree(text);
    for (i=0; i<strlen(text); i++)
        if ((line[i] == ' ') || (line[i] == '\t')) line[i] = '_';
    fprintf(Ffp,"Database %s 0100\n",line);
    fprintf(Ffp,"# \n");
    /*
    ** This is no longer needed... 
    */
    /*  fprintf(Ffp,"include $MCFIO_DIR/mcf_NTuBld.db\n"); */
    fprintf(Ffp,"# \n");
    fprintf(Ffp,"make line_title \"ntuBuild Database, v1.0\"\n");
    text = XmTextGetString(window->descriptW);
    tc = text; 
    if (*tc == '\0') 
       fprintf(Ffp,"make line_title \" \"\n");
    else {
       ncTot = strlen(tc); nc =0;
       while (nc < ncTot) {
            tc2 = strchr(tc,'\n');
            nl = (int) (tc2-tc)/sizeof(char);
            if ((tc2 == NULL) || (nl > 75)) nl = 75;
            strncpy(line, tc, nl); line[nl] = '\0';
            fprintf (Ffp,"make line_title \"%s\"\n", line);
            tc += nl; nc += nl;
            if (*tc == '\n') {
               tc++;
               nc++;
            }
       }
    }
    XtFree(text);
    
    text = XmTextGetString(window->titleW); 
    fprintf(Ffp,"make header \"%s\" /\n", text);
    XtFree(text);
    
    text = XmTextGetString(window->versionW); 
    fprintf(Ffp," \"%s\" /\n", text);
    XtFree(text);
    
    if (window->nameIndexBlank) 
      fprintf(Ffp," \"dummyIndex\" /\n");
    else {
      text = XmTextGetString(window->nameIndexW); 
      fprintf(Ffp," \"%s\" /\n", text);
      XtFree(text);
    }
    /*
    ** Count the number of variable really defined 
    */
    dNTu->numVariables = 0;
    for (i=0; i< dNTu->numAvailable; i++)
         if (dNTu->variables[i]->nameBlank == False)
               dNTu->numVariables = dNTu->numVariables +1;
    
    fprintf(Ffp, " %d %d %d \n", dNTu->maxMultiplicity,  dNTu->orgStyle, 
                               dNTu->numVariables);
                               
    fprintf(Ffp,"# \n");
    for (iv=0; iv< nVar; iv++) {
       for (j=0; j<dNTu->numVariables; j++)  
          if (dNTu->varOrdering[j] == iv)  i = j; 
       varTmp = dNTu->variables[i];
       fprintf(Ffp,"# \n");
       fprintf(Ffp, "make variable \"%s\" /\n", varTmp->name);
       if (varTmp->description == NULL) 
             fprintf(Ffp,"\" \" /\n");
       else 
             fprintf(Ffp, " \"%s\" /\n", varTmp->description);
       fprintf(Ffp, " %d ", varTmp->type);
       for (j=0; j<5; j++) {
             if (j< varTmp->numDim) dd[j] = varTmp->dimensions[j];
             else dd[j] = 0;
       }      
       if (varTmp->isFixedSize)
                fprintf(Ffp, "\"Yes\" %d %d %d %d %d %d \n", varTmp->numDim, 
                          dd[0], dd[1], dd[2], dd[3], dd[4]);
          else 
                fprintf(Ffp, "\"No\" %d %d %d %d %d %d \n", varTmp->numDim, 
                          dd[0], dd[1], dd[2], dd[3], dd[4]);
    }
    fprintf(Ffp,"# \n");
    fprintf(Ffp,"end \n");
    XtSetSensitive(window->generateF77, True);
    XtSetSensitive(window->generateC, True);
    XtSetSensitive(window->generateDbin, True);
    window->isSaved = True;
    fclose(Ffp);                          
}
/*
** Extend the list of available variable, by a fixed amount..
**     (NUM_START_VARIABLES for now..)
*/
void ExtendVariableList(nTuBuildWindow *window)
{
    varGenNtuple **varLTmp;
    descrGenNtuple *dNTu = window->descrNtu;
    int i,j;

    varLTmp =
       (varGenNtuple **) malloc(sizeof(varGenNtuple *) * 
              (NUM_START_VARIABLES + window->descrNtu->numAvailable));
    for (i=0; i<(window->descrNtu->numAvailable + NUM_START_VARIABLES); i++) {
       if (i < window->descrNtu->numAvailable)
            varLTmp[i] = window->descrNtu->variables[i];
       else {
          varLTmp[i] = (varGenNtuple *) malloc(sizeof(varGenNtuple));
       varLTmp[i]->nameBlank = True;
       varLTmp[i]->name = NULL;
       varLTmp[i]->description = NULL;
       varLTmp[i]->type = INTEGER_NTU;
       varLTmp[i]->isFixedSize = False;
       varLTmp[i]->numDim = 0;
       for (j=0; j<MAX_VAR_DIMENSIONS; j++) 
            varLTmp[i]->dimensions[j] = -1;
       varLTmp[i]->offset = 0;
       varLTmp[i]->offsetXDR = NULL;     
       }
    }
    free(window->descrNtu->variables);
    window->descrNtu->variables = varLTmp;
         
    window->descrNtu->numAvailable += NUM_START_VARIABLES;
    UpdateVariableList(window, 0);
    
}
int VerifyStruct(nTuBuildWindow *window, int help)
{
    char *text;
    varGenNtuple *var1, *var2;
    descrGenNtuple *dNTu = window->descrNtu;
    int i,j;
    
    if (window == NULL) return False;
    if((window->titleBlank || window->multiplicityBlank ) ||
       (window->nameIndexBlank && window->descrNtu->maxMultiplicity > 0)) {
          if (help == True) DialogF(DF_ERR, window->shell, 1, 
" You must fill the name/title, \n,\
  the name for the main index and \n\
  the maximum main index value (0 if not applicable)", "Acknowledged");
      return False;
    }
    text = XmTextGetString(window->versionW);
    if (strlen(text) < 3) { 
          if (help == True) DialogF(DF_ERR, window->shell, 1, 
" The version string is suspect. \n Expecting string y.xx", "Acknowledged");
      XtFree(text);
      return False;
    } 
    XtFree(text);
    text = NULL;
    if (window->nameIndexBlank == FALSE) text =
                             XmTextGetString(window->nameIndexW); 
    for(i=0; i<dNTu->numAvailable; i++) {
         if (dNTu->variables[i]->nameBlank) continue;
         if (text != NULL) {
           if (strcmp(text, dNTu->variables[i]->name) == 0) {
            if (help == True) DialogF(DF_ERR, window->shell, 1,
" The variable name for the multiplicity conflicts with variable name %s",
"Acknowledged", dNTu->variables[i]->name);
            XtFree(text);
            return False;
            }
        }
        for (j=i+1; j<dNTu->numAvailable; j++) {
           if (dNTu->variables[j]->nameBlank) continue;
           if (strcmp(dNTu->variables[j]->name, 
                      dNTu->variables[i]->name) == 0) {
            if (help == True) DialogF(DF_ERR, window->shell, 1,
" Duplicate Variable name  %s", "Acknowledged", dNTu->variables[i]->name);
            if (text != NULL) XtFree(text);
            return False;
            }
        }
    }
    XtFree(text);
    return True;
}
