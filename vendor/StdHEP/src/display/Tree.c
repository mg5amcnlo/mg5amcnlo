/**********************************************************************************
  * Tree.c: The Tree Widget Source File
  *         From:
  *                   The X Window System, 
  *            Programming and Applications with Xt
  *                   OSF/Motif Edition
  *         by
  *                Douglas Young
  *              Prentice Hall, 1990
  *
  *                 Example described on pages: 397-419
  *
  *
  *  Copyright 1989 by Prentice Hall
  *  All Rights Reserved
  *
  * This code is based on the OSF/Motif widget set and the X Window System
  *
  * Permission to use, copy, modify, and distribute this software for 
  * any purpose and without fee is hereby granted, provided that the above
  * copyright notice appear in all copies and that both the copyright notice
  * and this permission notice appear in supporting documentation.
  *
  * Prentice Hall and the author disclaim all warranties with regard to 
  * this software, including all implied warranties of merchantability and fitness.
  * In no event shall Prentice Hall or the author be liable for any special,
  * indirect or cosequential damages or any damages whatsoever resulting from 
  * loss of use, data or profits, whether in an action of contract, negligence 
  * or other tortious action, arising out of or in connection with the use 
  * or performance of this software.
  *
  * Open Software Foundation is a trademark of The Open Software Foundation, Inc.
  * OSF is a trademark of Open Software Foundation, Inc.
  * OSF/Motif is a trademark of Open Software Foundation, Inc.
  * Motif is a trademark of Open Software Foundation, Inc.
  * DEC is a registered trademark of Digital Equipment Corporation
  * HP is a registered trademark of the Hewlett Packard Company
  * DIGITAL is a registered trademark of Digital Equipment Corporation
  * X Window System is a trademark of the Massachusetts Institute of Technology
  **********************************************************************************/


#include	  <X11/Intrinsic.h>
#include	  <X11/IntrinsicP.h>
#include	  <X11/StringDefs.h>
#include	  <X11/CoreP.h>
#include  	<X11/CompositeP.h>
#include	  <X11/ConstrainP.h>
#include	  "tree.h"
#include	  "treeP.h"
#define   MAX(a,b) ((a) > (b) ? (a) : (b))

static void             Initialize();
static void             ConstraintInitialize();
static void             ConstraintDestroy();
static Boolean          ConstraintSetValues();
static Boolean          SetValues();
static XtGeometryResult GeometryManager();
static void             ChangeManaged();
static void             insert_new_node();
static void             delete_node();
static void             new_layout();
static void             Redisplay();
static TreeOffsetPtr    create_offset();
static int              compute_positions();
static void             shift_subtree();
static void             set_positions();
static void             reset();
static Position         current_position();
static void             set_current_position();
static Position         sum_of_positions();

static XtResource resources[] = {
 {XtNhorizontalSpace,XtCSpace,XtRDimension,sizeof(Dimension),
   XtOffset(XsTreeWidget, tree.h_min_space), XtRString,"15" },
 {XtNverticalSpace,XtCSpace, XtRDimension,sizeof (Dimension),
   XtOffset(XsTreeWidget, tree.v_min_space), XtRString,"5"  },
 {XtNforeground, XtCForeground, XtRPixel, sizeof (Pixel),
  XtOffset(XsTreeWidget, tree.foreground), XtRString,"Black"},
};

static XtResource treeConstraintResources[] = {
 {XtNsuperNode, XtCSuperNode, XtRPointer, sizeof(Widget),
   XtOffset(TreeConstraints, tree.super_node),
   XtRPointer, NULL},
};

XsTreeClassRec XstreeClassRec = {
  {
    /* core_class fields  */
    (WidgetClass) &constraintClassRec,/* superclass         */
    "Tree",                           /* class_name         */
    sizeof(XsTreeRec),                /* widget_size        */
    NULL,                             /* class_init         */
    NULL,                             /* class_part_init    */
    FALSE,                            /* class_inited       */	
    Initialize,                       /* initialize         */
    NULL,                             /* initialize_hook    */	
    XtInheritRealize,                 /* realize            */
    NULL,                             /* actions            */
    0,                                /* num_actions        */	
    resources,                        /* resources          */
    XtNumber(resources),              /* num_resources      */
    NULLQUARK,                        /* xrm_class          */
    TRUE,                             /* compress_motion    */	
    TRUE,                             /* compress_exposure  */	
    TRUE,                             /* compress_enterleave*/	
    TRUE,                             /* visible_interest   */
    NULL,                             /* destroy            */
    NULL,                             /* resize             */
    Redisplay,                        /* expose             */
    SetValues,                        /* set_values         */
    NULL,                             /* set_values_hook    */	
    XtInheritSetValuesAlmost,         /* set_values_almost  */
    NULL,                             /* get_values_hook    */	
    NULL,                             /* accept_focus       */
    XtVersion,                        /* version            */	
    NULL,                             /* callback_private   */
    NULL,                             /* tm_table           */
    NULL,                             /* query_geometry     */	
    NULL,                             /* display_accelerator*/
    NULL,                             /* extension          */
  },
  {
    /* composite_class fields */
    GeometryManager,                 /* geometry_manager    */
    ChangeManaged,                   /* change_managed      */
    XtInheritInsertChild,            /* insert_child        */	
    XtInheritDeleteChild,            /* delete_child        */	
    NULL,                            /* extension           */
  },
  { 
    /* constraint_class fields */
   treeConstraintResources,          /* subresources        */
   XtNumber(treeConstraintResources),/* subresource_count   */
   sizeof(TreeConstraintsRec),       /* constraint_size     */
   ConstraintInitialize,             /* initialize          */
   ConstraintDestroy,                /* destroy             */
   ConstraintSetValues,              /* set_values          */
   NULL,                             /* extension           */
   },
  {
    /* Tree class fields */
    0,                               /* ignore              */	
  }
};

WidgetClass XstreeWidgetClass = (WidgetClass) &XstreeClassRec;

static void Initialize(request, inew)
    XsTreeWidget request, inew;
{
  Arg       wargs[2];
  XGCValues values;
  XtGCMask  valueMask;
  /*
   * Make sure the widget's width and height are 
   * greater than zero.
   */
  if (request->core.width <= 0)
    inew->core.width = 5;
  if (request->core.height <= 0)
    inew->core.height = 5;
  /*
   * Create a graphics context for the connecting lines.
   */
  valueMask = GCForeground | GCBackground;
  values.foreground = inew->tree.foreground;
  values.background = inew->core.background_pixel;
  inew->tree.gc = XtGetGC ((Widget)inew, valueMask, &values);  
  /*
   * Create the hidden root widget.
   */
  inew->tree.tree_root = (Widget) NULL;
  XtSetArg(wargs[0], XtNwidth, 1);
  XtSetArg(wargs[1], XtNheight, 1);
  inew->tree.tree_root = 
          XtCreateWidget("root", widgetClass, (Widget)inew, wargs, 2);
  /*
   * Allocate the tables used by the layout
   * algorithm.
   */
  inew->tree.horizontal = create_offset(10);
  inew->tree.vertical   = create_offset(10);
} 

static void ConstraintInitialize(request, inew)
     Widget request, inew;
{
  TreeConstraints tree_const = TREE_CONSTRAINT(inew);
  XsTreeWidget tw = (XsTreeWidget) inew->core.parent;
  /*
   * Initialize the widget to have no sub-nodes.
   */
  tree_const->tree.n_sub_nodes = 0;
  tree_const->tree.max_sub_nodes = 0;
  tree_const->tree.sub_nodes = (WidgetList) NULL;
  tree_const->tree.x = tree_const->tree.y = 0; 
  /*
   * If this widget has a super-node, add it to that 
   * widget' sub-nodes list. Otherwise make it a sub-node of 
   * the tree_root widget.
   */
  if(tree_const->tree.super_node)
    insert_new_node(tree_const->tree.super_node, inew);
  else
    if(tw->tree.tree_root)
      insert_new_node(tw->tree.tree_root, inew);
} 

static Boolean SetValues(current, request, inew)
    XsTreeWidget current, request, inew;
{
 int       redraw = FALSE;
 XGCValues values;
 XtGCMask  valueMask;
 /*
  * If the foreground color has changed, redo the GC's
  * and indicate a redraw.
  */
 if (inew->tree.foreground != current->tree.foreground ||
     inew->core.background_pixel !=
                           current->core.background_pixel){
   valueMask         = GCForeground | GCBackground;
   values.foreground = inew->tree.foreground;
   values.background = inew->core.background_pixel;
   XtReleaseGC((Widget)inew, inew->tree.gc);
   inew->tree.gc    = XtGetGC ((Widget)inew, valueMask, &values);   
   redraw = TRUE;     
 }
 /*
  * If the minimum spacing has changed, recalculate the
  * tree layout. new_layout() does a redraw, so we don't
  * need SetValues to do another one.
  */
 if (inew->tree.v_min_space != current->tree.v_min_space ||
     inew->tree.h_min_space != current->tree.h_min_space){ 
   new_layout(inew);
   redraw = FALSE;
 }
 return (redraw);
}

static Boolean ConstraintSetValues(current, request, inew)
    Widget current, request, inew;
{
 TreeConstraints newconst = TREE_CONSTRAINT(inew);
 TreeConstraints current_const = TREE_CONSTRAINT(current);
 XsTreeWidget tw = (XsTreeWidget) inew->core.parent;
 /*
  * If the super_node field has changed, remove the widget
  * from the old widget's sub_nodes list and add it to the
  * new one.
  */
 if(current_const->tree.super_node !=
                                  newconst->tree.super_node){
   if(current_const->tree.super_node)
     delete_node(current_const->tree.super_node, inew);
   if(newconst->tree.super_node)
     insert_new_node(newconst->tree.super_node, inew);
   /*
    * If the Tree widget has been realized, 
    * compute new layout.
    */
   if(XtIsRealized((Widget)tw))
     new_layout(tw);
  }               
  return (False);
}

static void insert_new_node(super_node, node)
     Widget super_node, node;
{
  TreeConstraints super_const = TREE_CONSTRAINT(super_node);
  TreeConstraints node_const = TREE_CONSTRAINT(node);
  int index = super_const->tree.n_sub_nodes;
  
  node_const->tree.super_node = super_node;
  /*
   * If there is no more room in the sub_nodes array, 
   * allocate additional space.
   */  
  if(super_const->tree.n_sub_nodes ==
                             super_const->tree.max_sub_nodes){
    super_const->tree.max_sub_nodes += 
                    (super_const->tree.max_sub_nodes / 2) + 2;
    super_const->tree.sub_nodes = 
     (WidgetList) XtRealloc((char*)super_const->tree.sub_nodes, 
                           (super_const->tree.max_sub_nodes) *
                            sizeof(Widget));
  } 
  /*
   * Add the sub_node in the next available slot and 
   * increment the counter.
   */
  super_const->tree.sub_nodes[index] = node;
  super_const->tree.n_sub_nodes++;
}

static void delete_node(super_node, node)
    Widget  super_node, node;
{
  TreeConstraints node_const = TREE_CONSTRAINT(node);
  TreeConstraints super_const;
  int             pos, i;
  /*
   * Make sure the super_node exists.
   */
  if(!super_node) return;  
  
  super_const = TREE_CONSTRAINT(super_node);
  /*
   * Find the sub_node on its super_node's list.
   */
  for (pos = 0; pos < super_const->tree.n_sub_nodes; pos++)
    if (super_const->tree.sub_nodes[pos] == node)
      break;
  if (pos == super_const->tree.n_sub_nodes) return;
  /*
   * Decrement the number of sub_nodes
   */  
  super_const->tree.n_sub_nodes--;
  /*
   * Fill in the gap left by the sub_node.
   * Zero the last slot for good luck.
   */
  for (i = pos; i < super_const->tree.n_sub_nodes; i++) 
    super_const->tree.sub_nodes[i] = 
                            super_const->tree.sub_nodes[i+1];
 super_const->tree.sub_nodes[super_const->tree.n_sub_nodes]=0;
}

static void ConstraintDestroy(w) 
#ifdef JUST_LIKE_BOOK  /* Unimportant but perhaps confusing */
     XsTreeWidget w;
#else
     Widget w;
#endif
{ 
  TreeConstraints tree_const = TREE_CONSTRAINT(w);
  int i;
 /* 
  * Remove the widget from its parent's sub-nodes list and
  * make all this widget's sub-nodes sub-nodes of the parent.
  */
  /* This could break, since XtDestroyWidget will sometimes remove a 
     supernode before this contraint has the opportunity to act. 
     For now, there will be no rearrangement of the tree upon removal 
     of a node... */
 
/*   if(tree_const->tree.super_node) { 
    delete_node(tree_const->tree.super_node, w);
    for(i=0;i< tree_const->tree.n_sub_nodes; i++)
      insert_new_node(tree_const->tree.super_node, 
                      tree_const->tree.sub_nodes[i]);
  }
  new_layout(w->core.parent);
  */
}

static XtGeometryResult GeometryManager(w, request, reply)
    Widget               w;
    XtWidgetGeometry    *request;
    XtWidgetGeometry    *reply;
{

 XsTreeWidget tw = (XsTreeWidget) w->core.parent;
 /*
  * No position changes allowed!.
  */
 if ((request->request_mode & CWX && request->x!=w->core.x)
     ||(request->request_mode & CWY && request->y!=w->core.y))
  return (XtGeometryNo);
 /*
  * Allow all resize requests.
  */
 if (request->request_mode & CWWidth)
   w->core.width = request->width;
 if (request->request_mode & CWHeight)
   w->core.height = request->height;
 if (request->request_mode & CWBorderWidth)
   w->core.border_width = request->border_width;
 /*
  *  Compute the new layout based on the new widget sizes;
  */
 new_layout(tw);
 return (XtGeometryYes);
}

static void ChangeManaged(tw)
    XsTreeWidget tw;
{
  new_layout(tw);
}


static void Redisplay (w, event, region)
     XsTreeWidget   w;
     XEvent        *event;
     Region         region;
{
  int              i, j;
  TreeConstraints tree_const;
  Widget          child;
  /*
   * If the Tree widget is visible, visit each managed child.
   */
  if(w->core.visible)
   for (i = 0; i < w -> composite.num_children; i++){
     child = w -> composite.children[i];
     tree_const = TREE_CONSTRAINT(child);
     /*
      * Draw a line between the right edge of each widget
      * and the left edge of each of its sub_nodes. Don't
      * draw lines from the fake tree_root.
      */
     if(child != w->tree.tree_root && 
        tree_const->tree.n_sub_nodes)
       for (j = 0; j < tree_const->tree.n_sub_nodes; j++)
         XDrawLine(XtDisplay(w), XtWindow(w), 
                   w->tree.gc,
                   child->core.x + child->core.width, 
                   child->core.y + child->core.height / 2,
                   tree_const->tree.sub_nodes[j]->core.x,
                   tree_const->tree.sub_nodes[j]->core.y + 
                tree_const->tree.sub_nodes[j]->core.height/2);
    }
}

static void new_layout(tw)
     XsTreeWidget   tw;
{
  /*
   *  Reset the auxiliary tables.
   */
  reset(tw->tree.vertical);
  reset(tw->tree.horizontal);
  /*
   * Compute each widget's x,y position
   */
  compute_positions(tw, tw->tree.tree_root, 0);
  /*
   * Move each widget into place.
   */
  set_positions(tw, tw->tree.tree_root, 0, 0);
  /*
   * Trigger a redisplay of the lines connecting nodes.
   */
  if(XtIsRealized((Widget)tw))
    XClearArea(XtDisplay(tw), XtWindow(tw), 0, 0, 0, 0, TRUE);
}

 static int compute_positions(tw, w, level)
     XsTreeWidget tw;
     Widget       w;
     long         level;
{
 Position       current_hpos, current_vpos;
 int             i, depth = 0;
 TreeConstraints tree_const = TREE_CONSTRAINT(w);
 /*
  * Get the current positions for this level.
  */
 current_hpos = current_position(tw->tree.horizontal, level);
 current_vpos = current_position(tw->tree.vertical, level);
 /*
  * Set the current horizontal width to the max widths of all
  * widgets at this level.
  */
 set_current_position(tw->tree.horizontal, level, 
                      MAX(current_hpos, w->core.width));
 /*
  * If the node has no sub_nodes, just set the vertical 
  * position to the next available space.
  */
 if(tree_const->tree.n_sub_nodes == 0){
   tree_const->tree.y = current_vpos;
 }
 else {
   Widget          first_kid, last_kid;
   TreeConstraints const1, const2;
   Position        top, bottom;
  /*
   * If the node has sub_nodes, recursively figure the 
   * positions of each sub_node.
   */
   for(i = 0; i < tree_const->tree.n_sub_nodes; i++)
    depth = compute_positions(tw, 
                              tree_const->tree.sub_nodes[i],
                              level + 1);
  /*
   * Now that the vertical positions of all children are 
   * known, find the vertical extent of all sub_nodes.
   */
  first_kid= tree_const->tree.sub_nodes[0];
  last_kid = 
   tree_const->tree.sub_nodes[tree_const->tree.n_sub_nodes-1];
  const1   = TREE_CONSTRAINT(first_kid);
  const2   = TREE_CONSTRAINT(last_kid);
  top      = const1->tree.y + first_kid->core.height / 2; 
  bottom   = const2->tree.y + last_kid->core.height / 2;
  /*
   * Set the node's position to the center of its sub_nodes.
   */
  tree_const->tree.y = (top + bottom)/2 - (w->core.height/ 2);
  /*
   * If this position is less than the next available 
   * position, correct it to be the next available
   * position, calculate the amount by which all sub_nodes
   * must be shifted, and shift the entire sub-tree.
   */
   if(tree_const->tree.y < current_vpos){
     Dimension offset = current_vpos - tree_const->tree.y;
     for(i = 0; i < tree_const->tree.n_sub_nodes; i++)
       shift_subtree(tree_const->tree.sub_nodes[i], offset);
    /*
     * Adjust the next available space at all levels below
     * the current level.
     */
     for(i = level + 1; i <= depth; i++){
       Position pos = current_position(tw->tree.vertical, i);
       set_current_position(tw->tree.vertical, i, pos+offset);
     }
     tree_const->tree.y = current_vpos;
     }
   }
 /*
  * Record the current vertical position at this level.
  */
  set_current_position(tw->tree.vertical, level,
                       tw->tree.v_min_space + 
                       tree_const->tree.y + w->core.height);
  return (MAX(depth, level));
}

static void shift_subtree(w, offset)
     Widget     w;
     Dimension  offset;
{
  int             i;
  TreeConstraints tree_const = TREE_CONSTRAINT(w);
  /*
   * Shift the node by the offset.
   */
  tree_const->tree.y += offset; 
  /*
   * Shift each sub-node into place.
   */
  for(i=0; i< tree_const->tree.n_sub_nodes; i++)
    shift_subtree(tree_const->tree.sub_nodes[i], offset);
}

static void set_positions(tw, w, level)
     XsTreeWidget tw;
     Widget       w;
     int          level;
{
 int               i;
 Dimension         replyWidth = 0, replyHeight = 0;
 XtGeometryResult  result;
  
 if(w){
  TreeConstraints tree_const = TREE_CONSTRAINT(w);
 /*
  * Add up the sum of the width's of all nodes to this 
  * depth, and use it as the x position.
  */
  tree_const->tree.x = (level * tw->tree.h_min_space) + 
                sum_of_positions(tw->tree.horizontal, level);
 /*
  * Move the widget into position.
  */
  XtMoveWidget (w, tree_const->tree.x, tree_const->tree.y);
 /*
  * If the widget position plus its width or height doesn't
  * fit in the tree, ask if the tree can be resized.
  */
  if(tw->core.width < tree_const->tree.x + w->core.width ||
     tw->core.height < tree_const->tree.y + w->core.height){
    result = 
      XtMakeResizeRequest((Widget)tw, MAX(tw->core.width, 
                                  tree_const->tree.x + 
                                  w->core.width),
                              MAX(tw->core.height, 
                                  tree_const->tree.y + 
                                  w->core.height),
                          &replyWidth, &replyHeight);
    /*
     * Accept any compromise.
     */
     if (result == XtGeometryAlmost)
       XtMakeResizeRequest ((Widget)tw, replyWidth, replyHeight, 
                             NULL, NULL);
  }
 /*
  * Set the positions of all sub_nodes.
  */
  for(i=0; i< tree_const->tree.n_sub_nodes;i++)
    set_positions(tw, tree_const->tree.sub_nodes[i], level+1);
  }
}

static TreeOffsetPtr create_offset(size)
   long size;
{
 TreeOffsetPtr  offset = 
                 (TreeOffsetPtr) XtMalloc(sizeof(TreeOffset));
 offset->size = size;
 offset->array = 
             (Dimension *) XtMalloc(size * sizeof(Dimension));
 return (offset);
}

static void reset(offset)
   TreeOffsetPtr offset;
{
  long i;
  for(i=0; i< offset->size; i++)
    offset->array[i] = 0;
}

static Position current_position(offset, position)
   TreeOffsetPtr  offset;
   long          position;
{
  if(position >= offset->size)
    return (0);
  return (offset->array[position]);
 }

static void set_current_position(offset, index, value)
   TreeOffsetPtr offset;
   int           index;
   Dimension     value;
{
 if(index >= offset->size){
   offset->size = index + index / 2;
   offset->array =
    (Dimension *) XtRealloc((char*)offset->array, 
                            offset->size * sizeof(Dimension));
 }
 offset->array[index] = value;
}

static Position sum_of_positions(offset, index)
   TreeOffsetPtr  offset;
   long           index;
{
  int    i;
  Position  sum  = 0;
  long      stop = index;
  if(index > offset->size) 
    stop = offset->size;
  for (i=0;i < stop; i++)
    sum += offset->array[i];
  return (sum);
}

