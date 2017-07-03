# General settings:
   set output 'ptj.eps'
   set terminal postscript eps enhanced color 'Helvetica' 16
   set style data lines
   set size 2.0,1.6

   set multiplot

   reset
   set xlabel "k_{/Symbol \\136 1}  (GeV)"
   set size 1.0,0.4
   set origin 0.0,0.1
   set bmargin 0
   set tmargin 0

   set ylabel "Deviation [%]"
   set format y "    %1.1f"
   set yrange[-100:100]
   set xrange[0:25]

   set xlabel "d_{ij}"
   set key right top

plot 0 noti w l lt 0 lw 6,\
"< paste ptj3_0.dat ptj3_1.dat ptj3_2.dat" using ($1):((($4)/($2) - 1.0)*100.0) w l lt 1 lw 1 title '(ptj3_0.dat) / (ptj3_1.dat)', \
"< paste ptj3_0.dat ptj3_1.dat ptj3_2.dat" using ($1):((($6)/($2) - 1.0)*100.0) w l lt 1 lw 1 title '(ptj3_0.dat) / (ptj3_2.dat)'

   reset
   set bmargin 0
   set tmargin 0

   set xlabel ""
   set format x ""
   set format y "%1.1t{/Symbol \\327}10^{%L}"
   set key right top

   set logscale y
   set xrange[0:25]
   set yrange[1e-3:1e0]

   set size 1.0,0.7
   set origin 0.0,0.8

   set ylabel "d{/Symbol s}/dd_{ij}   [mb]"
   set format y "%1.1t{/Symbol \\327}10^{%L}"

plot \
0 noti w l lt 0 lw 3,\
"ptj3_0.dat" using ($1):($2) w l lt 1 lw 1 title 'ptj3_0.dat',\
"ptj3_1.dat" using ($1):($2) w l lt 2 lw 1 title 'ptj3_1.dat',\
"ptj3_2.dat" using ($1):($2) w l lt 3 lw 1 title 'ptj3_2.dat'

set nomultiplot

