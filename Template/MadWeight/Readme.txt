***********************************************************************
**                       MadWeight   Help                            **
***********************************************************************
**
** option format:
**    They are 3 possibilities:
**          $> ./bin/madweight.py A B C 
**                  launch only steps A,B and C
**          $> ./bin/madweight.py A+ 
**                  launch all steps after (and including) step A
**          $> ./bin/madweight.py A- 
**                  launch all steps preceeding (and including) step A
**
**    Where A/B/C are one of the following names:
**
**    1) param: Creates all the param_card.dat as required in MadWeight_card
**    2) analyzer: Analyzes the Feynman diagrams, the transfer function 
**                 and creates the fortran code for the integration
**    3) compilation: Final compilation for all the SubProcesses
**    4) event: Verification of the LHCO file. Select the events containing
**              the exact number of jets/electrons/muons for each SubProcesses.
**    5) dir: Creates the directory for each parralel run 
**            (one run by param_card and by event)
**    6) launch: Launches the computation of the weights
**    7) control: Launches control-status for the run
**    8) collect: Collects all the data
**    9) plot: Creates the plots for the likelihood
**
** additional options:
**
**   -help: Provides this help
**   -version: Gives the Madweight version number
**   relaunch: Relaunches failed (and zero results) runs. 
**   clean=NAME: Suppress the event by event log/result/... for the run NAME
**        example: $> ./bin/madweight.py clean=my_run
**   clean: Suppress the event by event log/result/... for the current run_name 
**        (name defined in run_card.dat)
**   status: give the number of idle/running/finish job at present time. This will not launch
**           any new job (not as 'control' will sometimes do)
**
***********************************************************************
