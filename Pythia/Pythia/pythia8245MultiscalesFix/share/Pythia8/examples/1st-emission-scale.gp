
# General settings:
   set output 'first-emission-scale.eps'
   set terminal postscript eps enhanced color 'Helvetica' 16
   set style data lines
   set size 1.0,1.0

   reset
   set xlabel "p_{{/Symbol \\136} evol}  (GeV)"

   #set ylabel "Deviation [%]"
   set format y "    %1.1f"
   set logscale y

   set key right top

#plot "t-old.dat" u 1:2 w l, "t-fix.dat" u 1:2 w l
plot "ttbar-t-old.dat" u 1:2 w l, "ttbar-t-fix.dat" u 1:2 w l

