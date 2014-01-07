#ifndef _mcf_tmp_INC
#define _mcf_tmp_INC


/***** template line_title *****/

typedef struct _line_title_s {
  char line[80]; /*  */
} line_title_s;
static const int n_el_line_title=1;
extern struct line_title_c {
  int n_obj_line_title;
  int idmline_title;
  line_title_s line_title[500];
} line_title_c_;
static int *n_obj_line_title = &(line_title_c_.n_obj_line_title);
static line_title_s *line_title = &line_title_c_.line_title[0];

/***** template header *****/

typedef struct _header_s {
  char title[80]; /*  */
  char version[80]; /*  */
  char namemaxindex[80]; /*  */
  int maxmult; /*  */
  int orgstyle; /*  */
  int nvar; /*  */
} header_s;
static const int n_el_header=6;
extern struct header_c {
  int n_obj_header;
  int idmheader;
  header_s header[1];
} header_c_;
static int *n_obj_header = &(header_c_.n_obj_header);
static header_s *header = &header_c_.header[0];

/***** template variable *****/

typedef struct _variable_s {
  char name[80]; /*  */
  char description[80]; /*  */
  int type; /*  */
  char isfixedsize[80]; /*  */
  int numdim; /*  */
  int dimensions[5]; /*  */
} variable_s;
static const int n_el_variable=10;
extern struct variable_c {
  int n_obj_variable;
  int idmvariable;
  variable_s variable[100];
} variable_c_;
static int *n_obj_variable = &(variable_c_.n_obj_variable);
static variable_s *variable = &variable_c_.variable[0];

#endif
