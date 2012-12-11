/*******************************************************************************
*									       *
* mcf_NTuIOUtil.h -- Utilities to manipulate files within the MCFIO Gen.      *
*        				Ntuple schema                          *
*									       *
*	P. Lebrun, October 1995.					       *
*									       *
*******************************************************************************/
nTuDDL *mcf_GetNTuByPtrID(int id);
nTuDDL *mcf_GetNTuByStreamID(int stream, int id);
int mcf_CheckValidCat(char *category, int dotDotDot);
char *mcf_ValidStr(char *string, int max_length, char *strKind);
int mcf_NTuId(int uid, char *category);
int mcfioC_GetNTupleIds(int stream, int *ids, int max);
int mcfioC_GetNTupleUID(int stream, int id);
void mcfioC_GetNTupleCategory(int stream, int id, char **answer);
void mcfioC_GetNTupleTitle(int stream, int id, char **answer);
void mcfioC_GetNTupleName(int stream, int id, char **answer);
void AddNTuDDLtoList(nTuDDL *ddl);
void DestroyNTuDDL(nTuDDL *ddl);
void DestroyVarGenNtuple(varGenNtuple *var);
void CopyVarGenNtuple(varGenNtuple *vFrom, varGenNtuple *vTo);
void DestroyGenNtuple(descrGenNtuple  *dNTu);
void mcfioC_SetForSaveDecoding(int val);
