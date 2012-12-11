/*
 *     dbin.cc
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include "mcf_ntuBldDbinc.h"
#include "mcf_ntubld_db.h"

static void dbin_debug();
static void lineparse();
static void dbinparse(char* str, char* typ, char* nam, 
                      char* var, char* com, char* dim);
static void  getmembers(long nmems);
static void getvalues();
static char* stlower(char*);
static void chrcat(char* str, char chr);
static char* token(char** str, char* sep);
static int testsep(char chr, char *sep);
static void mcf_ntubld_interpret();

static char varname[40], objname[40], curstruct[40];
static char chvalues[500], *values, dim[20];
static char tok1[30], tok2[30], tok3[100], com[100];
static char line[1000];
static int n_instance_line_title, n_instance_header, n_instance_variable;

static double dvar[100];
static float rvar[100];
static char chvar[100][80];
static char dbpath[FILENAME_MAX+1], filename[FILENAME_MAX+1];
static long nvars, morevalues, n_templates;
static long inc_depth, n_instance, debug_on;
static int isl;
static long n_significant, lnlen=0; 
static FILE *inFile, *curFile1,  *curFile2,  *curFile3,  *curFile4,  *curFile5;
static const char *fnamep;
static void dbin_getrec(char* fname[],void (*)(void));

static void dbin_getrec(char* fname[],void (*interpret)(void))
{
  char chr;
  int istat;
  long inc_depth_old = 0;
  const int nchmx = 300;
  /*
  ** Start be intializing all these globals, to be able to call this routine
  ** more than once.. 
  */
  inc_depth = 0;
  n_instance = 0;
  lnlen = 0; 
  /* extract path from filename */
  strcpy(filename,*fname);
  if (strrchr(filename,'/') != NULL)  {
      strcpy(dbpath,filename);
      *(strrchr(dbpath,'/')+1)='\0';
  } else {
      dbpath[0] = '\0';
  }

  /* open file */
  inFile = fopen(*fname,"r");
  if (inFile == NULL) {
    fprintf(stdout,"Error opening %s\n",*fname);
    return;
  }
  else
      {
          if (debug_on) fprintf(stdout,"Opened %s\n",*fname);
      }
  /*  read a line */
  while (inc_depth>=0) {
    istat=1;
    while (istat!=EOF) {
        if (inc_depth > inc_depth_old) /*  need to open new file */
        {
          long ifstat;
          ifstat=1;
          fnamep = (const char *)filename;
          if (inc_depth==1) {curFile1 = fopen(fnamep,"r");
            if (curFile1==NULL) {
              fprintf(stdout,"Error opening %s\n",fnamep);
              ifstat=0;
            }
            else {
                if (debug_on) fprintf(stdout,"Opened %s\n",fnamep);
            }
          }
          if (inc_depth==2) {curFile2 = fopen(fnamep,"r");
            if (curFile2==NULL) {
              fprintf(stdout,"Error opening %s\n",fnamep);
              ifstat=0;
            }
            else {
                if (debug_on) fprintf(stdout,"Opened %s\n",fnamep);
            }
          }
          if (inc_depth==3) {curFile3 = fopen(fnamep,"r");
            if (curFile3==NULL) {
              fprintf(stdout,"Error opening %s\n",fnamep);
              ifstat=0;
            }
            else {
                if (debug_on) fprintf(stdout,"Opened %s\n",fnamep);
            }
          }
          if (inc_depth==4) {curFile4 = fopen(fnamep,"r");
            if (curFile4==NULL) {
              fprintf(stdout,"Error opening %s\n",fnamep);
              ifstat=0;
            }
            else {
                if (debug_on) fprintf(stdout,"Opened %s\n",fnamep);
            }
          }
          if (inc_depth==5) {curFile5 = fopen(fnamep,"r");
            if (curFile5==NULL) {
              fprintf(stdout,"Error opening %s\n",fnamep);
              ifstat=0;
            }
            else {
                if (debug_on) fprintf(stdout,"Opened %s\n",fnamep);
            }
          }
        }
      inc_depth_old = inc_depth;
      if (inc_depth==0) istat=fgetc(inFile);
      if (inc_depth==1) istat=fgetc(curFile1);
      if (inc_depth==2) istat=fgetc(curFile2);
      if (inc_depth==3) istat=fgetc(curFile3);
      if (inc_depth==4) istat=fgetc(curFile4);
      if (inc_depth==5) istat=fgetc(curFile5);
      chr = istat;
        if (chr == '\t') chr = ' '; /*  remove tabs */
        if (chr == '\n') { /* line is complete; process it */
            if (morevalues == 1) {  /* line extension containing values */
                /* if final significant char is '/', mark next
                   line as values continuation */
                int i;
                isl=0;
                for (i=0;i<strlen(line);i++) {
                    if (line[i] == '!') i=strlen(line);
                    if (line[i] == '/') isl=i;
                }
                if (isl != 0) {
                    n_significant = 0;
                    for (i=isl;i<strlen(line);i++) {
                        if (line[i] == '!') i=strlen(line);
                        if (line[i]!='/' && line[i]!=' ' && line[i]!='\t'
                            && i < strlen(line) ) n_significant++;
                    }
                    if (n_significant != 0) morevalues = 0;
                } else {
                    morevalues = 0;
                }
                strcat(values," "); 
                if (morevalues == 0) {
                    strcat(values,line);
                } else {
                    strncat(values,line,isl-1);
                }
            } else { /* line is not an extension. Parse it. */
                dbinparse(line, tok1, tok2, tok3, com, dim);
            }
            if (morevalues == 0) { 
                /* no more line extensions to read. Process line. */
                /* now interpret the line */
                if (tok1[0] != '\0') {
                    if (debug_on) fprintf(stdout,"%s %s %s\n",tok1,tok2,values);
                    lineparse();
                    (*interpret)();
                }
            }
            line[0] = '\0';
            lnlen = 0;
        } else {
            /* add to line */
            if (chr != '\r') { line[lnlen++]=chr; line[lnlen]='\0'; }
        }
    }
    inc_depth--; line[0] = '\0';
  }
  return;
}

/****************************************************************************/
static void lineparse()
{
  char* tokn, *env, *envName, *tmp1, *tmp2;
  long l, in_template;
  varname[0] = '\0';
  objname[0] = '\0';
  if (!strcmp(tok1,"end")) {strcpy(curstruct,"--"); in_template = 0;}
  if (!strcmp(tok1,"structure")) {strcpy(curstruct,tok2);}
  if (!strcmp(tok1,"database")) ;
  if (!strcmp(tok1,"incname")) ;
  if (!strcmp(tok1,"index")) ;
  if (!strcmp(tok1,"provide")) ;
  if (!strcmp(tok1,"parent")) ;
  if (!strcmp(tok1,"child")) ;
  if (!strcmp(tok1,"dimension")) ;
  if (!strcmp(tok1,"template")) {in_template = 1; strcpy(curstruct,tok2);
    n_instance = 0;}
  if (!strcmp(tok1,"command")) {in_template = 1; strcpy(curstruct,tok2);}
  if (!strcmp(tok1,"include")) {  /* switch input to specified file */
 /*
 ** Commented out, we use absolute path name in the includes. 
 ** This allows us to go to more than one directory..
 */
/*    strcpy(filename,dbpath); */ 
/*    strcat(filename,tok2); */
/*
** We now implement translation of environmental variable
**
*/
      if (tok2[0] == '$') {
         tmp1 = strchr(&tok2[1], '/');
         if (tmp1 == NULL) {
              fprintf(stderr, "DBin error, Unkonw path %s\n", tok2);
              return;
         }
         envName = (char *) malloc(sizeof(char) * (strlen(tok2)+1));
         strcpy(envName, &tok2[1]);
         tmp2 = strchr(envName, '/'); *tmp2 = '\0';
         env = getenv(envName);
         free(envName);
         
         if (env == NULL) { 
              fprintf(stderr, "DBin error, Unkonw path %s\n", tok2);
              return;
         }
         strcpy(filename,env); l = strlen(env); filename[l] = '/'; l++;
         strcpy(&filename[l], tmp1);
            
      } else  strcpy(filename, tok2);
    inc_depth++;
  } 
  if (!strcmp(tok1,"make")) {
    n_instance++;
    strcpy(varname,"TEMPLATE_");
    strcat(varname,stlower(tok2));
  }
  if (!strcmp(tok1,"define")) {
      /* get first token (name) from values list */
    tokn = token(&values," \t");
    strcpy(varname,"TEMPLATE_");
    strcat(varname,tok2);
    strcpy(objname,tok2);
    strcat(objname,"_");
    strcat(objname,tokn);
  }
  if (!strcmp(tok1,"call")) {
      /* get first token (name) from values list */
    tokn = token(&values," \t");
    strcpy(varname,"COMMAND_");
    strcat(varname,tok2);
  }
  if (!strncmp(tok1,"int",3) || !strcmp(tok1,"real") || !strcmp(tok1,"double") ||
      !strncmp(tok1,"char",4) || !strcmp(tok1,"material") ) {
    if ((! strncmp(curstruct,"--",2)) && (in_template == 0)) {
      fprintf(stdout,"dbin: Parameter \"%s\" not in structure; ignored:\n",
              tok2);
      fprintf(stdout,"      %s\n",line);
    } else {
        /* parse values */
      strcpy(varname,curstruct);
      strcat(varname,".");
      strcat(varname,tok2);
      getvalues();
    }
  }
}

/****************************************************************************/
static void dbinparse(char* str, char* typ, 
                      char* nam, char* var, char* com, char* dim)
{
/* Parse from line the type, name, value, comment */
  int i;
  long nc = 0;

  nvars = 1;
  chvalues[0] = dim[0] = typ[0] = nam[0] = var[0] = com[0] = '\0';
  values = chvalues;
  
/* if final significant char is '/', mark next line as values continuation */
  isl=strlen(str);
  for (i=0;i<strlen(str);i++) {
      if (str[i] == '!') i=strlen(str);
      if (str[i] == '/') isl=i;
  }
  morevalues = 0;
  if (isl != strlen(str)) {
      n_significant = 0;
      for (i=isl;i<strlen(str);i++) {
          if (str[i] == '!') i=strlen(str);
          if (str[i]!='/' && str[i]!=' ' && str[i]!='\t'
               && i < strlen(line) ) n_significant++;
      }
      if (n_significant == 0) morevalues = 1;
  }

    /* initial whitespace, type, whitespace */
  while ((str[nc] == ' ') || (str[nc] == '\t')) ++nc;
  while ((str[nc] != ' ') && (str[nc] != '\t')
         && (nc < strlen(str))) chrcat(typ,str[nc++]);
  while ((str[nc] == ' ') || (str[nc] == '\t')) ++nc;
    /* name, whitespace, dimension? */
  while ((str[nc] != ' ') && (str[nc] != '\t') && (str[nc] != '(' )
         && (nc < strlen(str))) chrcat(nam,str[nc++]);
  while ((str[nc] == ' ') || (str[nc] == '\t')
         && (nc < strlen(str))) ++nc;
  if (str[nc] == '(') {  /* have a dimensioned array */
      /* get dimension */
    while (str[++nc] != ')') chrcat(dim,str[nc]); nc++;
    nvars = atol(dim);
  }
    /* skip over value(s) to comment */
  while (  (str[nc] != '!')   && 
           (str[nc] != '/' || ( morevalues && (nc != isl) ) )   &&
           ( (nc < strlen(str)) || ( morevalues && (nc < isl)) )   ) chrcat(chvalues,str[nc++]);
    /* comment */
  while (((str[nc] == '!') || (str[nc] == '\t'))
         && (nc < strlen(str))) ++nc;
  while (nc <= strlen(str)) { chrcat(com,str[nc++]); }
    /* turn mnemonic num into variable name var */
  var = nam;
}

/****************************************************************************/

static void getvalues()
{
    char* tokn;
    long nv=0; while (nv < nvars) {
          /* get next token and trim it from the values list. */
        if (!strncmp(tok1,"char",4) || !strncmp(tok1,"material",8) ) {
            char *iq1, *iq2;
            iq1 = strchr(values,'"');
            iq2 = strrchr(values,'"');
            if (iq1 != NULL) {
                strncpy(chvar[nv],iq1+1,iq2-iq1-1);
                chvar[nv][iq2-iq1-1] = '\0';
            }
            else
                strcpy(chvar[nv],values);
        } else {
            tokn = token(&values," \t");
            if (tokn != NULL) {
                if (!strncmp(tok1,"int",3)) rvar[nv] = atol(tokn);
                if (!strcmp(tok1,"real"))   rvar[nv] = atof(tokn);
                if (!strcmp(tok1,"double")) dvar[nv] = atof(tokn);
            }
        }
        nv++;
    }
}

/****************************************************************************/

static void getmembers(long nmems)
{
    char *tokn, *iq1, *iq2;
    long n, nq, nv=0;
    /* fill string interiors with '@' so they are delineated as tokens */
    n=0; nq=0; while (n<strlen(values)) {
        if (values[n]=='"') nq++;
        if ((values[n]==' '||values[n]=='\t') && nq%2==1) values[n] = '@';
        n++;
    }
    while (nv < nmems) {
          /* get next token and trim it from the values list. */
        tokn = token(&values," \t");
        if ( tokn[0]=='"' ) {
            n=0; while (n<strlen(tokn))
                { if (tokn[n]=='@') tokn[n] = ' '; n++; }
            iq1 = strchr(tokn,'"');
            iq2 = strrchr(tokn,'"');
            strncpy(chvar[nv],iq1+1,iq2-iq1-1);
            chvar[nv][iq2-iq1-1] = '\0';
        } else {
            strcpy(chvar[nv],tokn);
        }
        rvar[nv] = atof(tokn);
        nv++;
    }
}

/****************************************************************************/
static void dbin_debug()
{
    debug_on = 1;
}

/****************************************************************************/
static void chrcat(char* str, char chr)
{
    int ln;
    ln = strlen(str);
    str[ln] = chr;
    str[ln+1]='\0';
}

/****************************************************************************/
static char * stlower(char* st) {
  int i=0;
  while (st[i] != '\0') {
    if (st[i] >= 'A' && st[i] <= 'Z') st[i] = st[i] + 'a' - 'A';
    i++;
  }
  return st;
}

/****************************************************************************/
static char* token(char** str, char* sep)
{
    int i=0;
    char *if1=NULL, *if2=NULL, *strend = *str + strlen(*str);
    /* if1 = rel. pointer to 1st token char */
    i=0; while (if1 == NULL && i<strlen(*str)) {
        if (!testsep(*(*str+i),sep))
            if1= *str+i;
        i++;
    }
    if (if1 == NULL) return if1;
    /* if2 = 1st char past the token */
    i=0; while (if2 == NULL && i<strlen(if1)) 
        { if (testsep(if1[i],sep)) if2=&if1[i]; i++; }
    if (if2<strend && if2 != NULL) {
        if (if2 != NULL) *if2 = '\0';
        *str = if2+1;
    } else {
        *str = strend;
    }
    return if1;
}

/****************************************************************************/
static int testsep(char chr, char *sep)
{
    int ist=0, i=0;
    while (sep[i] != '\0')
        if (sep[i++] == chr || chr == '\0' || chr == '\n' ) ist=1;
    return ist;
}
/***  Database read routine  ***/
/***    Generated automatically using the dbin tool. */
/***    Not to be modified by user. */
/*
** Modifiedt by P.L., to abe able to load all the templates into 
** one file... And included in this file, to avoid defining too many 
** global symbols.  This clearly breaks the dbin mold, to be discussed 
** later.. 
*/
void mcf_ntubldRead(char* fname)
{
  void (*pf)(); /* pointer to interpreter */
  inc_depth = 0;
  n_instance =0;
  lnlen=0; 
  debug_on = 0;  
  pf = &mcf_ntubld_interpret;
  mcf_ntubldInit();
  n_instance_line_title = 0;
  n_instance_header = 0;
  n_instance_variable = 0;
  dbin_getrec(&fname,pf);
}
static void mcf_ntubld_interpret()
{
int inum, index, i, ndim, iok;
iok=0;
if ( !strcmp(varname,"TEMPLATE_line_title") ) {
  inum = 0; iok = 1;
  getmembers(n_el_line_title);
  index = n_instance_line_title;
  *n_obj_line_title = n_instance_line_title+1;
  strcpy(line_title[index].line,chvar[inum++]);
  n_instance_line_title++;
}
if ( !strcmp(varname,"TEMPLATE_header") ) {
  inum = 0; iok = 1;
  getmembers(n_el_header);
  index = n_instance_header;
  *n_obj_header = n_instance_header+1;
  strcpy(header[index].title,chvar[inum++]);
  strcpy(header[index].version,chvar[inum++]);
  strcpy(header[index].namemaxindex,chvar[inum++]);
  header[index].maxmult = rvar[inum++];
  header[index].orgstyle = rvar[inum++];
  header[index].nvar = rvar[inum++];
  n_instance_header++;
}
if ( !strcmp(varname,"TEMPLATE_variable") ) {
  inum = 0; iok = 1;
  getmembers(n_el_variable);
  index = n_instance_variable;
  *n_obj_variable = n_instance_variable+1;
  strcpy(variable[index].name,chvar[inum++]);
  strcpy(variable[index].description,chvar[inum++]);
  variable[index].type = rvar[inum++];
  strcpy(variable[index].isfixedsize,chvar[inum++]);
  variable[index].numdim = rvar[inum++];
  for (i=0;i<5;i++) variable[index].dimensions[i] = rvar[inum++];
  n_instance_variable++;
}
}
