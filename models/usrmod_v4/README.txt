*****************************************
*					*
*  h   h  ooo  w     w ttttttt ooo	*
*  h   h o   o w     w 	  t   o   o	*
*  hhhhh o   o w  w  w    t   o   o	*
*  h   h o   o w  w  w    t   o   o	*
*  h   h  ooo   ww ww     t    ooo	*
*					*
*					*
*    use madgraph's user mod...?	*
*					*
*       S.de Visscher FYNU/CP3		*
*     Louvain la neuve, 2/06/06		*
*					*
*****************************************

This text file will instruct you about what to do
step by step in order to use the "user mode" of madgraph.

All you need is in Model/usrmod directory.
What you canc see here is 

1/ Three ".dat" files you'll have to edit to enter
your own model...

2/ A directory wich contains part of files that will be used to create
".f" and ".inc" files needed by madgraph; this directory
is called Header_and_footer... you don't need to care about this

3/ ConversionScript.pl:
	The perl script wich mix information from Header_and_footer
	files and from .dat files...

4/ testprog.f
	 a little code that permti to check directly if all values
	 printed are the ones defined in param_card.dat.

5/couplingsvalue.f
        Slightly the same role than testprog.f vut produce a text file
        containing in a table the list of couplings and their values

To do things properly, make your own usermod directory to let the
original one intact, let' say you call it "z_prime"

**************************************************
1/ Editing the .dat files.

a/ particles.dat
----------------

This file lists all particles of the model.
The basis used here is Standard Model particles.

There are two very important tags you have to take into account:
MODEL EXTENSION and END.
Any new particle you want to use has to be described between those ones.

For example if you need a Z' you may write like the following:

#MODEL EXTENSION
zp      zp        V        W      ZPMASS ZPWIDTH S ZP   32
# END


ATTENTION: use space instead of TAB between chars...

The two first columns indicate particle and anti-particle names
The third is about particles type (Vector, Fermion or Scalar)
The fourth say how madgraph has to represent it on Feynman diagrams.
The fifth gives the variable name for the mass (max. 7 char.)
The sixth gives the variable name for the width (max.7 char.)
The seventh say if particles color type is singlet (S), triplet (T), or octet (O).
The eigth gives the label name in Feynman diagrams.
and finally the last one is the PDGCode of the particle.

*************
ABOUT PDGCODE:
For new kind of particle that do not possess PDG Code you can enter the number you want, just
ensure it is not yet used.
*************

Now once the new particles are entered, we have to deal with interactions.
So let's move on to interaction.dat file

b/ interaction.dat
------------------

As you can see, all Standard Model vertices are listed by type.
At the end of the file you can find the USRVertex part wich is the one
you have to edit.

You can do like te following:

#   USRVertex
zp zp z GNNZ QED

wich gives the vertex name (you choose the one you want - madgraph is
full of freedom!)  and type (QED) for Z'Z'Z interaction.

**************
ABOUT INTERACTION NAME:
Pay attention to the fact that interactions

f' f s and f f' s are different...and therefore need two differents
names! like for example: gfpfs and gffps...
*************

Ok this is fine but you still have to give the name of all parameters
that will have to be used in all vertices definitions.

c/ VariableName.dat
-------------------

Suppose you need an exotic coupling like:

GNNZ= param1/(param2**2)*sqrt(param3)

The way to declare you paramaters is very simple: do it like the following:

write this:

param1 #comment for the first param
param2 #comment for second param
param3 #comment for third param

in VariableName.dat... that's all. This will permit to declare
parameters in input.inc file, and write comments in param_card.dat
automatically.

Obviously, you cannot declare parameter that are already existing in
particle.dat like masses and width, nor ones declared in
Header_and_Footer/coupl_header file...

*****************************************************************
2/ Run the shell script ./ConversionScript.pl

CAUTION: it will ask you wether you want to keep old couplings.f and
 param_card.dat that you maybe changed in a previous run. The goam
 here is to avoid to loose all changes you would have done before, in
 other words: save you time and energy. In the case you answer by
 "yes", old files are stored in the OldFiles directory, with an
 incrementation.

*****************************************************************
3/ Look the results...
What did the perl script do?

It simply looked to what you entered in particles.dat,
interactions.dat and VariableName.dat and send those informations to
coupl.inc, couplings.f, input.inc, printout.f, ident_card.dat,
lha_reading.f and param_card.dat.

What you still need to do now is to edit two files:
couplings.f and param_card.dat.

Let's see what we find in couplings.f:

Near the end of the file you see something like:

c UserMode couplings


      GNNZ=1d0


Instead of 1d0 wich is obviously the default value you have to edit
the true couplings value you want...this is the hard work part for
you!
If the coupling is of the form VVV then it is declared as double
precision (All declarations are made in coupl.inc)
If the coupling is of the form VVSS, VVS, SSS, SSSS, VSS, VVV it is
declared as double complex and if the type is FFS or FFV, it is the
same thing BUT there is the separation between Vector and Axial part

Example:
a VVV interaction is written: gwwz = ee*cw/sw

a VVS interaction is written: gzzh = dcmplx( ee2/sc2*Half*v, Zero )
where the first term in parenthesis indicates the real part and the
second the imaginary.

a FFS interaction is written: ghtop(1) = dcmplx( -mtMS/v, Zero )
			      ghtop(2) = dcmplx( -mtMS/v, Zero )


To be sure to how to enter differents kind of vertices, check out the HELAS manual:
HELAS: HELicity Amplitude Subroutines for Feynman Diagram Evaluation
(Murayama,Watanabe,Hagiwara,Jan. 1992)


*******************************************************************
4/ Adapt param_card.dat to your business

Your masses and widths variables are automatically inserted in
param_card.dat. By default they are set at 100 and 1 GeV respectively.
Then you just have to change those ones for you convinience...

******************************************************************
5/ Run the following to make a testprogram and check that everything
compiles and that the masses, widths and couplings are set as they
should:

 make testprog
 ./testprog

 or

 make couplings
 ./couplingsvalue

It will show you wether good values are assigned to good variables...

*****************************************************************
6/ Change the proc_card.dat to do the process you wish:
for example:

pp>zpzp	#First Process
99	#Max QCD couplings
99	#Max QED couplings
With the model called z_prime

Now you're ready to generate the smatrix. 

***********************************************************
7/ Do ./bin/newprocess
*********************************************************
8/ Now you can generate some events: ./bin/generate_events
