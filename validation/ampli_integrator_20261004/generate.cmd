set automatic_html_opening False
set notification_center False
import model loop_sm
generate e+ e- > u u~ [QCD]
output /tmp/mg5-amplicol-backend-sq5iorrd/eejets -f -nojpeg
generate p p > e+ e- [QCD]
output /tmp/mg5-amplicol-backend-sq5iorrd/drell_yan -f -nojpeg
quit
