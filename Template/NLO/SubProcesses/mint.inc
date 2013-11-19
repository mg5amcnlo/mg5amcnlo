c Note that the number of intervals in the integration
c grids, 'nintervals', should be equal to
c "nintervals = min_inter * 2^n",
c where 'n' is smaller or equal to 'min_it'.

      integer nintervals,ndimmax,nintegrals
      parameter (nintervals=32,ndimmax=60,nintegrals=5)

      integer min_inter,min_it,max_it,max_points
      data min_inter /4/
      data min_it /4/
      data max_it /100/
      data max_points /100000/

      double precision accuracy
      common /to_accuracy/ accuracy
