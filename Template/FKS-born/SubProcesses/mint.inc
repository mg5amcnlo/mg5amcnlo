c Note that the number of intervals in the integration
c grids, 'nintervals', should be equal to
c "nintervals = nint_used * 2^n",
c where 'n' is smaller or equal to 'min_it'.

      integer nintervals,ndimmax
      parameter (nintervals=32,ndimmax=60)

      integer nint_used,min_it,max_it,max_points
      data nint_used /4/
      data min_it /4/
      data max_it /100/
      data max_points /100000/

      double precision accuracy
      common /to_accuracy/ accuracy
