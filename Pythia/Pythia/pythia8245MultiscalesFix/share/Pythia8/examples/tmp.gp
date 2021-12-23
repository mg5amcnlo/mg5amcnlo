
# General settings:
   set output 'pp2ttj.eps'
   set terminal postscript eps enhanced color 'Helvetica' 16
   set style data lines
   set size 1.0,2.0

   set multiplot

   reset
   set xlabel "p_{{/Symbol \\136} evol}  (GeV)"

   #set ylabel "Deviation [%]"
   #set format y "    %1.1f"
   set logscale y

   set key right top

   set origin 0.,0.

plot "pp2ttj_avgscale_vs_tS.dat" u 1:2 w l, "pp2ttj_avgscale_vs_tH.dat" u 1:2 w l


  set origin 0.,1.
plot "pp2ttj_tS.dat" u 1:2 w l, "pp2ttj_tH.dat" u 1:2 w l
