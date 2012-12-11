/*******************************************************************************
*									       *
* mcf_NTuIOFiles.h -- Utilities to manipulate files within the MCFIO Gen.      *
*        				Ntuple schema                          *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
int mcfioC_DeclareNtuple(int uid, char *title, char *category, 
                                int stream, char *filename);
int mcfioC_EndDeclNTuples(int stream);                                
nTuDDL *mcf_GetFileNTuDDL(char*filename);
void    mcf_ComputeNTuOffsets(nTuDDL *ddl); 
void    mcf_ComputeNTuLengths(nTuDDL *ddl); 
void    mcf_ComposeDoth(descrGenNtuple *dNtu, char *filename);
