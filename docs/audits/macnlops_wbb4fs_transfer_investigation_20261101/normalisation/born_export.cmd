set automatic_html_opening False --no_save
set auto_update 0 --no_save
set nb_core 16 --no_save
set run_mode 2 --no_save
set lhapdf /nfs/home/rikkert/LHAPDF/bin/lhapdf-config --no_save
import model loop_sm
define p = g u d s c u~ d~ s~ c~
define j = g u d s c u~ d~ s~ c~
generate p p > w+ b b~ QCD=2 QED=1
output /export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/diagnosis_transfer_20261101/normalisation/Wbb_Born_LO -nojpeg
quit
