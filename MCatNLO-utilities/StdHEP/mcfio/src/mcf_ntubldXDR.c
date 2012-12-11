/***  XDR code for dbin database I/O  ***/
/***  Generated automatically using the dbin tool. */
/***  Not to be modified by user. */
#include "mcf_ntubldXDRinc.h"
static int *idat, ln, i;
static u_int ui;
static float *dat;
static double *ddat;
static char* xdrstring;

bool_t xdr_line_title(XDR *xdrs, line_title_s *dbobj)
{
  ui = 0;
  if (xdrs->x_op == XDR_ENCODE) {
    ln = strlen(dbobj->line); }
  if (!xdr_int(xdrs,&ln)) return 0;
  xdrstring = dbobj->line;
  if (!xdr_string(xdrs,&xdrstring,ln)) return 0;
  return 1;
}

bool_t xdr_header(XDR *xdrs, header_s *dbobj)
{
  ui = 0;
  if (xdrs->x_op == XDR_ENCODE) {
    ln = strlen(dbobj->title); }
  if (!xdr_int(xdrs,&ln)) return 0;
  xdrstring = dbobj->title;
  if (!xdr_string(xdrs,&xdrstring,ln)) return 0;
  if (xdrs->x_op == XDR_ENCODE) {
    ln = strlen(dbobj->version); }
  if (!xdr_int(xdrs,&ln)) return 0;
  xdrstring = dbobj->version;
  if (!xdr_string(xdrs,&xdrstring,ln)) return 0;
  if (xdrs->x_op == XDR_ENCODE) {
    ln = strlen(dbobj->namemaxindex); }
  if (!xdr_int(xdrs,&ln)) return 0;
  xdrstring = dbobj->namemaxindex;
  if (!xdr_string(xdrs,&xdrstring,ln)) return 0;
  if (!xdr_int(xdrs,&(dbobj->maxmult))) return 0;
  ui = dbobj->maxmult;
  if (!xdr_int(xdrs,&(dbobj->orgstyle))) return 0;
  ui = dbobj->orgstyle;
  if (!xdr_int(xdrs,&(dbobj->nvar))) return 0;
  ui = dbobj->nvar;
  return 1;
}

bool_t xdr_variable(XDR *xdrs, variable_s *dbobj)
{
  ui = 0;
  if (xdrs->x_op == XDR_ENCODE) {
    ln = strlen(dbobj->name); }
  if (!xdr_int(xdrs,&ln)) return 0;
  xdrstring = dbobj->name;
  if (!xdr_string(xdrs,&xdrstring,ln)) return 0;
  if (xdrs->x_op == XDR_ENCODE) {
    ln = strlen(dbobj->description); }
  if (!xdr_int(xdrs,&ln)) return 0;
  xdrstring = dbobj->description;
  if (!xdr_string(xdrs,&xdrstring,ln)) return 0;
  if (!xdr_int(xdrs,&(dbobj->type))) return 0;
  ui = dbobj->type;
  if (xdrs->x_op == XDR_ENCODE) {
    ln = strlen(dbobj->isfixedsize); }
  if (!xdr_int(xdrs,&ln)) return 0;
  xdrstring = dbobj->isfixedsize;
  if (!xdr_string(xdrs,&xdrstring,ln)) return 0;
  if (!xdr_int(xdrs,&(dbobj->numdim))) return 0;
  ui = dbobj->numdim;
  idat = (int *) dbobj->dimensions;
  if (!xdr_array(xdrs,(char **) &idat,&ui,ui,sizeof(int),(xdrproc_t) xdr_int)) return 0;
  return 1;
}

