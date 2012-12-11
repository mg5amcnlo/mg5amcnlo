/*******************************************************************************
*									       *
* mcf_BrowseUtil1.c -- Utilities and auxillary Panels for NTuple Browser.      *
*									       *
* Copyright (c) 1995, 1996 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include <ctype.h>
#include <limits.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/SelectioB.h>
#include <Xm/RowColumn.h>
#include <Xm/ToggleB.h>
#include <Xm/Form.h>
#include <Xm/Label.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/Text.h>
#include "DialogF.h"
#include "getfiles.h"
#include "misc.h"
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Dict.h"
#include "mcfio_Util1.h"
#include "mcfio_Direct.h"
#include "mcf_nTupleBuild.h"
#include "mcf_NTuIOUtils.h"
#include "mcf_BrowseUtil1.h"
#include "mcf_BrowseMainPanel.h"
#ifdef HISTO
#include "histoscope.h"
#endif
extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;

#define SIZEOFLIST_BROWSERANAL 20

nTuBrowserInfo **NTupleBrowserList = NULL;

static void createBrowserDataDump(nTuBrowserInfo * nTuBr);
static void dismissDataDumpCB(Widget w, nTuBrowserInfo * nTuBr,
                               caddr_t call_data);
static void printValuesBrief(char *cDat, int type,
                             int n_instance, char **text);
                             
static void printValuesFull(char *cDat, varGenNtuple *var, char **text);

static void briefCB(Widget w, nTuBrowserInfo * nTuBr ,
                             XmToggleButtonCallbackStruct *callData);

static void modifiedFromDataCB(Widget w,  
                               nTuBrowserInfo * nTuBr, caddr_t call_data);
                               
static void modifiedToDataCB(Widget w,  
                               nTuBrowserInfo * nTuBr, caddr_t call_data);

/*
** Create a list of little Data structure for place holder of analysis 
** information.
*/
void mcfioC_CreateBrowserAnalysis()
{
   int i,j;
     if (NumOfNTuples < 1 ) return;
     NTupleBrowserList =  
        (nTuBrowserInfo **) malloc(sizeof(nTuBrowserInfo *) * NumOfNTuples);
     for (i=0; i<NumOfNTuples; i++) {
        NTupleBrowserList[i] = 
           (nTuBrowserInfo *) malloc(sizeof(nTuBrowserInfo));
        NTupleBrowserList[i]->id = i+1;   
        NTupleBrowserList[i]->templateW = NULL;
        NTupleBrowserList[i]->dumpDataW = NULL; 
        NTupleBrowserList[i]->dumpDataFormW = NULL; 
        NTupleBrowserList[i]->dumpDataShellW = NULL; 
        NTupleBrowserList[i]->nHistoItems = 0;
        NTupleBrowserList[i]->nHisto1D = 0;
        NTupleBrowserList[i]->nHisto2D = 0;
        NTupleBrowserList[i]->nHistoNtuples = 0;
        NTupleBrowserList[i]->sizeOfLists = SIZEOFLIST_BROWSERANAL;
        NTupleBrowserList[i]->hsItemList =
             (nTuBroHsGeneral **) malloc(sizeof(nTuBroHsGeneral *) 
                                              * SIZEOFLIST_BROWSERANAL);
        for (j=0; j<SIZEOFLIST_BROWSERANAL; j++)
             NTupleBrowserList[i]->hsItemList[j] = NULL;
        NTupleBrowserList[i]->currentData = False;    
        NTupleBrowserList[i]->data = NULL;    
     }
}

void mcfioC_DestroyBrowserAnalysis()
{
     int i,j;
     nTuBroHsGeneral *hsG;
     nTuBroHs1D *h1;
     nTuBroHs2D *h2;
     
     if (NTupleBrowserList == NULL) return;
     for (i=0; i<NumOfNTuples; i++) {
       if (NTupleBrowserList[i]->templateW != NULL) 
           if (CloseNTuBuildWindow(NTupleBrowserList[i]->templateW) ==
           GFN_CANCEL) return; 
       if (NTupleBrowserList[i]->dumpDataW != NULL) 
            XtDestroyWidget(NTupleBrowserList[i]->dumpDataW);
       for (j=0; j<NTupleBrowserList[i]->sizeOfLists; j++) {
          hsG = NTupleBrowserList[i]->hsItemList[j]; 
          if (hsG != NULL) {
#ifdef HISTO          
             if(hsG->id >0) hs_delete(hsG->id); 
             switch (hsG->type) {
               case HS_1D_HISTOGRAM:
                  h1 = (nTuBroHs1D *) hsG;
                  if (h1->varIndices != NULL) free (h1->varIndices);
                  free(h1);
                  break;
             case  HS_2D_HISTOGRAM:
                h2 = (nTuBroHs2D *) hsG;
                if (h2->xVarIndices != NULL) free (h2->xVarIndices);
                if (h2->yVarIndices != NULL) free (h2->yVarIndices);
                free(h2);
                break;
             default : 
                printf(" McfioC_DestroyBrowserAnalysis : internal Error\n");
                break;
           }
#endif            
           free (hsG); NTupleBrowserList[i]->hsItemList[j] = NULL;
         }
       }
       if (NTupleBrowserList[i]->data != NULL)
           free(NTupleBrowserList[i]->data);
       free(NTupleBrowserList[i]);
     }
     free(NTupleBrowserList);
     NTupleBrowserList = NULL;
}
void mcfioC_createBrowserData(nTuBrowserInfo *nTuBr)
{
    int nBytes, lastFixed, i;
    nTuDDL *ddl;
    descrGenNtuple *dNTu;
     
    if (nTuBr == NULL) return;
    if (nTuBr->data != NULL) free(nTuBr->data);
     ddl = mcf_GetNTuByPtrID(nTuBr->id);
     if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
     /*
     ** 24 = 8 char. for version +  the mulplicity variable +  Padding + 
     **       fence (2 short words = 1 long)
     */
     nBytes = 8*sizeof(char) + 4 * sizeof(int);
     if(dNTu->firstIndexed == -1) lastFixed = dNTu->numVariables;
        else lastFixed = dNTu->firstIndexed;
     for (i=0; i<lastFixed; i++) 
        if (dNTu->variables[i]->lengthW != 0) 
              nBytes += dNTu->variables[i]->lengthB;
     if (dNTu->firstIndexed != -1) {
       for (i=dNTu->firstIndexed; i<dNTu->numVariables; i++) 
        if (dNTu->variables[i]->lengthW != 0)
             nBytes += (dNTu->maxMultiplicity * dNTu->variables[i] ->lengthB);
     }
     nTuBr->data = (void *) malloc(nBytes);       
}            
           
     
void mcfioC_ExtendBrowserAnalysis(nTuBrowserInfo * nTuBr)
{
     int j, nnew, ivar, nn, i;
     int *il, **ill;
     nTuDDL *ddl;
     descrGenNtuple *dNTu;
     nTuBroHsGeneral **hsGList;
     
     if (nTuBr == NULL) return;
     ddl = mcf_GetNTuByPtrID(nTuBr->id);
     if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
     nnew = nTuBr->sizeOfLists + SIZEOFLIST_BROWSERANAL;
     hsGList =
             (nTuBroHsGeneral **) malloc(sizeof(nTuBroHsGeneral *) 
                                              * nnew);
     for (i=0; i<nnew; i++) hsGList[i] = NULL;
     memcpy((void *) il, (void *) nTuBr->hsItemList,
               (sizeof(nTuBroHsGeneral *) * nTuBr->nHistoItems));
     nTuBr->sizeOfLists = nnew;
     free(nTuBr->hsItemList);
     nTuBr->hsItemList = hsGList; 
}              
void mcfioC_ShowBrowserDataDump(nTuBrowserInfo * nTuBr) {

   char *text, *t1, *t2;
   int i, j, in, nChar, len, nMult, n, lastFixed, n1, n2;
   int iFrom, iTo;
   char *cDat;
   char titleW[NTU_MAX_TITLE_LENGTH+15];
   char version[8];
    nTuDDL *ddl;
    descrGenNtuple *dNTu;
   varGenNtuple *var;

   if (nTuBr->data == NULL ) {
      mcfioC_createBrowserData(nTuBr);
      nTuBr->currentData = False;
   }   
   if (nTuBr->currentData == False) {
      for(i=0, n1=-1; i<NumOfNTuples; i++) 
        if(NTupleBrowserList[i] == nTuBr) n1 = i;
      if (n1 == -1) {
        DialogF(DF_WARN, McfioMainPanelW, 1,
       "Internal error in ShowBrowserDataDump\nPlease report.",
       "Acknowledged");
         return;
      }   
      if (mcfioC_NTuple(1, (n1+1), 
	              (char *) nTuBr->data) == False) {
       DialogF(DF_WARN, McfioMainPanelW, 1,
    "No data for this event!.\nPlease choose other Ntuple or skip this event",
       "Acknowledged");
       return;
      }
      nTuBr->currentData = True;
   }
   if (nTuBr->dumpDataShellW == NULL) createBrowserDataDump(nTuBr);
   ddl = mcf_GetNTuByPtrID(nTuBr->id);
   sprintf(titleW,"Data for %s",ddl->title); 
   XtVaSetValues(nTuBr->dumpDataShellW, XmNtitle, titleW, 0);
   if (nTuBr->currentData == False) {
       XmTextSetString(nTuBr->dumpDataW,
    "Undecoded Data, Press View Data again to decode the requested buffer ");
       XtManageChild(nTuBr->dumpDataFormW);
       return;
   }
   if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
   if(dNTu->firstIndexed == -1) lastFixed = dNTu->numVariables;
           else lastFixed = dNTu->firstIndexed;
   cDat = (char *) nTuBr->data;
   strncpy(version, cDat, 8);  cDat += dNTu->multOffset; 
   /*
   ** set the text widget From, To, to go for a specific track  
   ** or substructures
   */
   nMult = *((int *) cDat);
   iFrom = nTuBr->dumpFrom;
   if (iFrom > nMult) iFrom = nMult;
   iFrom--;
   iTo = nTuBr->dumpTo;
   if (iTo > nMult) iTo = nMult;
   /*
   ** Estimate of the amount of text we have to generate 
   */
   if (XmToggleButtonGetState(nTuBr->dumpDataBriefW)) {
      /*
      ** Assume 160 char. max. per line 
      */
      n2 =  dNTu->numVariables - lastFixed; 
      text = (char *) malloc(sizeof(char ) * (160*(7 + lastFixed + n2*nMult)));
   } else {
      /*
      ** Assume 32 characters maximum per printed values
      */
      if (lastFixed != 0) { 
          for (i=0, n1=0; i<lastFixed; i++) {
            var = dNTu->variables[i]; n = 1;
            for (j=0; j<var->numDim; j++) n = n * var->dimensions[j];
            n1 += n*32; n1 += 80*var->numDim;  
          }
       } else n1 = 0;
       if (dNTu->firstIndexed != -1) {
           for (i=dNTu->firstIndexed, n2=0; i<dNTu->numVariables; i++) {
              var = dNTu->variables[i]; n = 1;
              for (j=0; j<var->numDim; j++) n = n * var->dimensions[j];
              n2 += n*32;  n2 += 80*var->numDim;  
           }
       } else n2 = 0;   
       text = (char *) malloc(sizeof(char ) * (1120 + n1 + nMult*n2));
    }   
   t1 = text;
   sprintf(t1,
             " Version for this Data block instance : %s \n%n", version, &len);
   t1 += len;
   if (dNTu->firstIndexed != -1 ) {
          sprintf(t1, " Multiplicity, %s = %d \n%n", dNTu->nameIndex, nMult,
          &len); t1+=len;
   }
   sprintf(t1, "               -----------------------------\n%n",&len); 
   t1+=len;
   if (lastFixed != 0) { 
       sprintf(t1, " Fixed size part of the block\n\n%n", &len);
       t1 += len;
       for (i=0; i<lastFixed; i++) {
            var = dNTu->variables[i]; n = 1;
            for (j=0; j<var->numDim; j++) n = n * var->dimensions[j];
            sprintf (t1, "%s = %n", var->name, &len ); t1 +=len;
            cDat = (char *) nTuBr->data;
            cDat += dNTu->variables[i]->offset;
            if (XmToggleButtonGetState(nTuBr->dumpDataBriefW)) 
                       printValuesBrief(cDat, var->type, n, &t1);
            else   printValuesFull(cDat, var, &t1);         
        } 
    }
    sprintf(t1,
      " ---- End of fixed size part --------------------------------- \n%n",
        &len);  t1 += len;
    if (iFrom == -1) { 
       sprintf(t1,
      " ---- Multiplicity 0, end of Dump -------- -------------------- \n%n",
        &len);  t1 += len;
    } else { 
    if ((dNTu->firstIndexed != -1) && 
         (dNTu->orgStyle == PARALLEL_ARRAY_NTU)) {
            sprintf(t1,
     " Variable size, organisation is of type parallel arrays \n\n%n", &len); 
            t1 += len;
            for (i=dNTu->firstIndexed; i<dNTu->numVariables; i++) {
              var = dNTu->variables[i]; n = 1;
              for (j=0; j<var->numDim; j++) n = n * var->dimensions[j];
              sprintf (t1, "%s :  \n%n", var->name, &len ); t1 +=len;
              cDat = (char *) nTuBr->data;
              cDat += var->offset;
              for (in = iFrom; in< iTo; in++) {
                 sprintf(t1, "  Index_%s = %d -> %n",dNTu->nameIndex, in, &len);
                 t1 += len;
                 if (XmToggleButtonGetState(nTuBr->dumpDataBriefW)) 
                       printValuesBrief(cDat, var->type, n, &t1);
                 else  printValuesFull(cDat, var, &t1);         
                 cDat += var->lengthB;
             } 
         }
     } else if  ((dNTu->firstIndexed != -1) && 
         (dNTu->orgStyle != PARALLEL_ARRAY_NTU)) {
            sprintf(t1,
     " Variable size, organisation is of type substructures \n%n", &len); 
            t1 += len;
            for (in = iFrom; in< iTo; in++) {
              sprintf(t1, "Index_%s = %d : \n%n",dNTu->nameIndex, (in+1), &len);
              t1 += len;
              for (i=dNTu->firstIndexed; i<dNTu->numVariables; i++) {
                var = dNTu->variables[i]; n = 1;
                for (j=0; j<var->numDim; j++) n = n * var->dimensions[j];
                cDat = (char *) nTuBr->data;
                cDat += ( var->offset + dNTu->subOffset[in]);
                sprintf (t1, "    %s = %n", var->name, &len ); t1 +=len;
                 if (XmToggleButtonGetState(nTuBr->dumpDataBriefW)) 
                       printValuesBrief(cDat, var->type, n, &t1);
                 else   printValuesFull(cDat, var, &t1);         
              }
         }
     }         
   }    
   XmTextSetString(nTuBr->dumpDataW,text);
   XtManageChild(nTuBr->dumpDataFormW);
   free(text);
       
   
}

static void createBrowserDataDump(nTuBrowserInfo * nTuBr)
{
    Arg args[50];
    int ac, nMult;
    Widget form, dismissBtn, fullBtn, radioBox, labelFromW, labelToW;
    char *cDat;
    nTuDDL *ddl;
    descrGenNtuple *dNTu;
   varGenNtuple *var;
    XmString s1;

   ddl = mcf_GetNTuByPtrID(nTuBr->id);
   if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
    ac = 0;
    XtSetArg(args[ac], XmNautoUnmanage, False); ac++; 
    XtSetArg(args[ac], XmNresizePolicy, XmRESIZE_NONE); ac++; 
    form = XmCreateFormDialog(McfioMainPanelW, "form", args, ac);
    nTuBr->dumpDataShellW = XtParent(form);
    nTuBr->dumpDataFormW = form;
    XtVaSetValues(nTuBr->dumpDataShellW, XmNtitle,
                                "Data Dump of an Ntuple", 0);
    AddMotifCloseCallback(XtParent(form),
                         (XtCallbackProc)dismissDataDumpCB, nTuBr);
                         
                         
    dismissBtn = XtVaCreateManagedWidget("dismissBtn",
    	    xmPushButtonGadgetClass, form,
    	    XmNlabelString, s1=XmStringCreateSimple("Dismiss"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 70,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 90, 0);
    XmStringFree(s1);
    XtAddCallback(dismissBtn, XmNactivateCallback,
    	    (XtCallbackProc)dismissDataDumpCB, (void *)nTuBr); 
    
    radioBox = XtVaCreateManagedWidget("radioBox",
	    	    xmRowColumnWidgetClass, form,
	    	    XmNradioBehavior, True,
	    	    XmNradioAlwaysOne, True,
	    	    XmNorientation, XmHORIZONTAL,
    		    XmNleftAttachment, XmATTACH_FORM,
    		    XmNbottomAttachment, XmATTACH_FORM, 0);

     nTuBr->dumpDataBriefW = 
          XtVaCreateManagedWidget("radioA",xmToggleButtonWidgetClass,
	    	    radioBox, 
	    	    XmNset, True,
    		    XmNlabelString,s1=XmStringCreateSimple("Brief"),0);
    XmStringFree(s1);
    XtAddCallback( nTuBr->dumpDataBriefW, XmNvalueChangedCallback,
		    (XtCallbackProc)briefCB, (caddr_t) nTuBr);
 	    
    fullBtn = 
          XtVaCreateManagedWidget("radioB",xmToggleButtonWidgetClass,
	    	    radioBox, 
	    	    XmNset, False,
    		    XmNlabelString,s1=XmStringCreateSimple("Full"),0);
    XmStringFree(s1);
    XtAddCallback(fullBtn, XmNvalueChangedCallback,
		    (XtCallbackProc)briefCB, (caddr_t) nTuBr);
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Index Range, From:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftOffset,5); ac++;
    XtSetArg(args[ac], XmNleftWidget, radioBox); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    labelFromW = 
       XmCreateLabelGadget(form, "indexRangeFrom", args,ac);
    XmStringFree(s1);
    XtManageChild(labelFromW);
    
    nTuBr->fromDataTextW  = XtVaCreateManagedWidget("fromDataText",
    	    xmTextWidgetClass, form,
            XmNcolumns, 5,
            XmNmaxLength, 5,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,labelFromW, 0);
    RemapDeleteKey(nTuBr->fromDataTextW );
    XmTextSetString(nTuBr->fromDataTextW, "1");
    nTuBr->dumpFrom = 1;
    XtAddCallback(nTuBr->fromDataTextW, XmNvalueChangedCallback,
    	    (XtCallbackProc)modifiedFromDataCB, (void *) nTuBr);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("To:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftOffset,1); ac++;
    XtSetArg(args[ac], XmNleftWidget, nTuBr->fromDataTextW ); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    labelToW = 
       XmCreateLabelGadget(form, "indexRaneTo", args,ac);
    XmStringFree(s1);
    XtManageChild(labelToW);
    
    nTuBr->toDataTextW  = XtVaCreateManagedWidget("toDataText",
    	    xmTextWidgetClass, form,
            XmNcolumns, 5,
            XmNmaxLength, 5,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,labelToW, 0);
    RemapDeleteKey(nTuBr->toDataTextW );
    cDat = (char *) nTuBr->data;
    cDat +=dNTu->multOffset; 
    nMult = *((int*) cDat);
    if (nMult > 3) {
      XmTextSetString(nTuBr->toDataTextW, "4");
      nTuBr->dumpTo = 4;
    } else {
      SetIntText(nTuBr->toDataTextW, nMult);
      nTuBr->dumpTo = nMult;
    }  
    XtAddCallback(nTuBr->toDataTextW, XmNvalueChangedCallback,
    	    (XtCallbackProc)modifiedToDataCB, (void *) nTuBr);
    
    ac = 0;
    XtSetArg(args[ac], XmNrows, 8); ac++;
    XtSetArg(args[ac], XmNcolumns, 80); ac++;
    XtSetArg(args[ac], XmNeditMode, XmMULTI_LINE_EDIT); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_POSITION); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget, dismissBtn); ac++;
    XtSetArg(args[ac], XmNdeleteResponse, XmDO_NOTHING); ac++;
    nTuBr->dumpDataW = XmCreateScrolledText(form,
                                          "SolutionText", args, ac);
    XtManageChild(nTuBr->dumpDataW);
    XtManageChild(form);
    

}    
static void dismissDataDumpCB(Widget w, nTuBrowserInfo * nTuBr,
                               caddr_t call_data)
{
    XtUnmanageChild(nTuBr->dumpDataFormW);
}
static void briefCB(Widget w, nTuBrowserInfo * nTuBr ,
                             XmToggleButtonCallbackStruct *callData)
{                             
    if (!callData->set)
    	return;
    mcfioC_ShowBrowserDataDump(nTuBr);
}
static void modifiedFromDataCB(Widget w,  
                               nTuBrowserInfo * nTuBr, caddr_t call_data)
{
   int ival;

    if (GetIntText(w, &ival) != TEXT_READ_OK) return;
    nTuBr->dumpFrom = ival; 
    mcfioC_ShowBrowserDataDump(nTuBr);
                                   
}                               
static void modifiedToDataCB(Widget w,  
                               nTuBrowserInfo * nTuBr, caddr_t call_data)
{
   int ival;

    if (GetIntText(w, &ival) != TEXT_READ_OK) return;
    nTuBr->dumpTo = ival; 
    mcfioC_ShowBrowserDataDump(nTuBr);
}
static void printValuesBrief(char *start, int type, 
                             int n_instance, char **text) 
{
   int *iDat;
   float *fDat; 
   double *gDat;
   char *cDat;
   short *sDat;
   void *pDat;
   int len;
   int j, n;
   char *t1 = *text;
   
   n = n_instance;
   switch (type) {
         case BYTE_NTU:
            if (n > 10) n = 10;
             for (j=0, cDat = start; j<n; j++, t1+=len, cDat++) 
                           sprintf (t1,"%x, %n",(*cDat), &len);
             break;
         case CHARACTER_NTU:
             if (n > 40) n = 40;
             for (j=0, cDat = start; j<n; j++, cDat++) 
                if (isgraph(*cDat)) { 
                     sprintf (t1,"%c%n", *cDat, &len);
                      t1+=len;
                 } else if (isspace(*cDat)) {
                     sprintf (t1," %n", &len);
                      t1+=len;
                 }          
             break;
         case POINTER_NTU:
             if (n > 2) n = 2;
             /* This is a bad idea,
             * we do not know what kind of pointer that is... Skip!
             */
             /*  pDat = (void *) cDat;
             for (j=0; j<n; j++, t1+=len, pDat++) 
                           sprintf (t1,"%x, %n", pDat, &len); */
             break;
         case LOGICAL_NTU:
             if (n > 10) n = 10;
             for (j=0, iDat = (int *) start; j<n; j++, t1+=len, iDat++) {
                  if (*iDat) sprintf (t1,"T,%n", &len);
                  else sprintf (t1,"F, %n", &len);
             }    
             break;
         case INTEGER2_NTU:
             if (n > 8) n = 8;
             for (j=0, sDat = (short *) start; j<n; j++, t1+=len, sDat++) 
                           sprintf (t1,"%d, %n",(*sDat), &len);
             break;
         case INTEGER_NTU:
             if (n > 4) n = 4;
             for (j=0, iDat = (int *) start; j<n; j++, t1+=len, iDat++) 
                           sprintf (t1,"%d, %n",(*iDat), &len);
             break;
         case REAL_NTU:
             if (n > 4) n = 4;
             for (j=0, fDat = (float *) start; j<n; j++, t1+=len, fDat++) 
                           sprintf (t1,"%g, %n",(*fDat), &len);
             break;
         case DBL_PRECISION_NTU:
             if (n > 4) n = 4;
             for (j=0, gDat = (double *) start; j<n; j++, t1+=len, gDat++) 
                           sprintf (t1,"%g, %n",(*gDat), &len);
             break;
         case COMPLEX_NTU:
             if (n > 2) n = 2;
             for (j=0, fDat = (float *) start; j<n; j++) {
                      sprintf (t1,"(%g, %n",(*fDat), &len);  t1+=len, fDat++;
                      sprintf (t1,"%g), %n",(*fDat), &len);  t1+=len, fDat++;
              } 
              break;
          case DBL_COMPLEX_NTU:
              if (n > 2) n = 2;
               for (j=0, gDat = (double *) start; j<n; j++) {
                   sprintf (t1,"(%g, %n",(*gDat), &len);  t1+=len, gDat++;
                   sprintf (t1,"%g), %n",(*gDat), &len);  t1+=len, gDat++;
               } 
               break;
          }
         if (n_instance != n) sprintf (t1, "...\n%n", &len);
                 else sprintf (t1, "\n%n", &len);
          t1 +=len;
          *text = t1;
}
static void printValuesFull(char *start, varGenNtuple *var, char **text)
{
   int *iDat;
   float *fDat; 
   double *gDat;
   char *cDat;
   short *sDat;
   void *pDat;
   int len;
   int idim, j, jl, n, ntot, nlines;
   char *t1;
   static char varDimSymbols[] = "ijklmn";
   
   t1 = *text;
   if (var->numDim < 1) {
         printValuesBrief(start, var->type, 1, text);
         return;
   }
   sprintf (t1,"\n%n",&len); t1 += len;
   for (idim=0, ntot=1, nlines = 1; idim<var->numDim; idim++) {
       sprintf (t1, "       Dimension, %c = 0, %d\n%n",varDimSymbols[idim],
                     var->dimensions[idim], &len ); t1 += len;
       ntot = ntot *  var->dimensions[idim];
       if (idim <(var->numDim-1) ) nlines = nlines *   var->dimensions[idim];
   }
   for (jl = 0; jl<nlines; jl++) { 
      sprintf (t1,"          %n",&len); t1 += len;
      n = var->dimensions[(var->numDim-1)]; 
      switch (var->type) {
         case BYTE_NTU:
             for (j=0, cDat = start; j<n; j++, t1+=len, cDat++) 
                           sprintf (t1,"%x, %n",(*cDat), &len);
             break;
         case CHARACTER_NTU:
             for (j=0, cDat = start; j<n; j++, cDat++) 
                if (isgraph(*cDat)) { 
                     sprintf (t1,"%c%n", *cDat, &len);
                      t1+=len;
                 } else if (isspace(*cDat)) {
                     sprintf (t1," %n", &len);
                      t1+=len;
                 }          
             break;
         case POINTER_NTU:
             /* This is a bad idea,
             * we do not know what kind of pointer that is... Skip!
             */
             /*  pDat = (void *) cDat;
             for (j=0; j<n; j++, t1+=len, pDat++) 
                           sprintf (t1,"%x, %n", pDat, &len); */
             break;
         case LOGICAL_NTU:
             for (j=0, iDat = (int *) start; j<n; j++, t1+=len, iDat++) {
                  if (*iDat) sprintf (t1,"T,%n", &len);
                  else sprintf (t1,"F, %n", &len);
             }    
             break;
         case INTEGER2_NTU:
             for (j=0, sDat = (short *) start; j<n; j++, t1+=len, sDat++) 
                           sprintf (t1,"%d, %n",(*sDat), &len);
             break;
         case INTEGER_NTU:
             for (j=0, iDat = (int *) start; j<n; j++, t1+=len, iDat++) 
                           sprintf (t1,"%d, %n",(*iDat), &len);
             break;
         case REAL_NTU:
             for (j=0, fDat = (float *) start; j<n; j++, t1+=len, fDat++) 
                           sprintf (t1,"%g, %n",(*fDat), &len);
             break;
         case DBL_PRECISION_NTU:
             for (j=0, gDat = (double *) start; j<n; j++, t1+=len, gDat++) 
                           sprintf (t1,"%g, %n",(*gDat), &len);
             break;
         case COMPLEX_NTU:
             for (j=0, fDat = (float *) start; j<n; j++) {
                      sprintf (t1,"(%g, %n",(*fDat), &len);  t1+=len, fDat++;
                      sprintf (t1,"%g), %n",(*fDat), &len);  t1+=len, fDat++;
              } 
              break;
          case DBL_COMPLEX_NTU:
               for (j=0, gDat = (double *) start; j<n; j++) {
                   sprintf (t1,"(%g, %n",(*gDat), &len);  t1+=len, gDat++;
                   sprintf (t1,"%g), %n",(*gDat), &len);  t1+=len, gDat++;
               } 
               break;
          }
         sprintf (t1, "\n%n", &len); t1 +=len;
      }
      *text = t1;
}


 
