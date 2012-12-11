/*******************************************************************************
*									       *
* mcf_NTuBuildWindow.h --    						       *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
int GetFileNTuBuildWindow(nTuBuildWindow *wInfo, char*filename);
void SaveFileNTuBuildWindow(nTuBuildWindow *wInfo); 
void ExtendVariableList(nTuBuildWindow *wInfo);
int VerifyStruct(nTuBuildWindow *window, int help);
int VerifyVarGenName(char *string); 
