set automatic_html_opening False --no_save
set nb_core 4 --no_save
import model loop_sm-no_b_mass
define p = g u d s c b u~ d~ s~ c~ b~
generate p p > t t~ [QCD]
output /export/tmp/rikkert/delta_noemission_20261027/TT -f -nojpeg
quit
