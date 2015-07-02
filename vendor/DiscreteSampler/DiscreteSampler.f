!
!     Module      : DiscreteSampler
!     Author      : Valentin Hirschi
!     Date        : 29.10.2014
!     Description : 
!              A relatively simple and flexible module to do
!              sampling of discrete dimensions for Monte-Carlo
!              purposes.
!
!     List of public subroutines and usage :
!
!     DS_initialize(tolerate_zero_grid, verbose)
!       :: Initializes the general options of the DiscreteSampler.
!       :: tolerate_zero_grid is a logical that specifies whether
!       :: DiscreteSampler should crash when asked to sample a dimension
!       :: of zero norm.
!       :: Verbose is another logical setting the verbosity of the
!       :: module.
!
!     DS_register_dimension(name, n_bins,(all_grid|void))
!       ::  Register a new dimension with its name and number of bins
!       ::  If all_grid is specified and set to False, then only a 
!       ::  running grid is specified, it is useful for registering
!       ::  grid intended for convolution only.
!
!     DS_remove_dimension(name)
!       ::  Removes and clear the dimension of input name
!
!     DS_print_global_info(name|index|void)
!       ::  Print global information on a registered information, using
!       ::  either its name or index. or all if none is selected
!     
!     DS_clear
!       ::  Reinitializes the module and all grid data
!
!     DS_binID(integer_ID)
!       ::  Creates an object of binID type from an integer ID. Notice
!       ::  that you can also use the assignment operator directly.
!
!     DS_add_bin(dimension_name, (binID|integerID|void))
!       ::  Add one bin to dimension of name dimension_name.
!       ::  The user can add an ID to the added bin or just use the
!       ::  default sequential labeling
!    
!     DS_remove_bin(dimension_name, (binID|integerID))
!       ::  Remove a bin in a dimension by either specfiying its index
!       ::  in this dimension or its binID (remember you can create a
!       ::  binID on the flight from an integer using the function
!       ::  DS_binID )
!
!     DS_add_entry(dimension_name, (binID|integerID), weight, (reset|void))
!       ::  Add a new weight to a certan bin (characterized by either
!       ::  its binID or the integer of the binID). If the dimension or
!       ::  the binID do not exist at the time of adding them, they will
!       ::  be registered on the flight. By setting the option 'reset'
!       ::  to '.True.' (default is '.False.'), you can chose to hardset
!       ::  the weight of this bin (with n_entries=min_bin_probing_points 
!       ::  then) instead of adding it.
!
!     DS_update_grid((dim_name|void))
!       ::  Update the reference grid of the dimension dim_name or 
!       ::  update all of them at the same time without argument.
!       ::  It uses the running grid for this and it reinitilizes it.
!
!     DS_write_grid((file_name|stream_id), (dim_name|void), (grid_type|void))
!       :: Append to file 'file_name' or a given stream the data for 
!       :: the current reference grid for dimension dim_name or
!       :: or all of them if called without dim_name.
!       :: You can specify grid_type to be written out to be either 'ref'
!       :: or 'run', and it is 'ref' by default.
!
!     DS_load_grid((file_name|stream_id), (dim_name|void))
!       :: Reset the run_grid for dimension dim_name and loads in
!       :: the data obtained from file 'file_name' or stream_id for this
!       :: dimension or all of them if called without dim_name.
!       :: The data is loaded in the running grid only (which is
!       :: re-initialized before this), so you have to call 
!       :: 'DS_update_grid' if you want it pased to the ref_grid.
!
!     DS_get_point(dim_name, random_variable, 
!       (binIDPicked|integerIDPicked), jacobian_weight, (mode|void),
!                                        (convoluted_grid_names|void))
!       :: From a given random variable in [0.0,1.0] and a dimension
!       :: name, this subroutine returns the picked bin or index and
!       :: the Jacobian weight associated.
!       :: The mode is by default 'norm' but can also be 'variance'
!       :: which means that the sampling is done either on the
!       :: variance in each bin or the absolute value of the weights.
!       :: mode == 'norm'
!       :: Jacobian = 1.0d0 / normalized_abs_wgt_in_selected_bin
!       :: mode == 'variance'
!       :: Jacobian = 1.0d0 / normalized_variance_in_selected_bin
!       :: mode == 'uniform'
!       :: Jacobian = 1.0d0 / n_bins_in_dimension
!       :: Setting the option 'convoluted_grid_names' to an array of
!       :: dimension names already registered in the module will have
!       :: the subroutine DS_get_point return a point sampled as the
!       :: grid 'dim_name' *convoluted with all grids specified in
!       :: convoluted_grid_names*.
!       :: Example:
!       ::  call DS_get_point('MyDim',0.02,out_binPicked,out_jac,mode='norm',
!       :: & convoluted_grid_names = (/'ConvolutionDim1','ConvolutionDim2'/))
! 
!     DS_set_min_points(min_point, (dim_name|void))
!       :: Sets the minimum number of points that must be used to probe
!       :: each bin of a particular dimension (or all if not specified) 
!       :: before DS_get_point uses a uniform sampling on the bins 
!       :: (possibly with convolution) when the reference grid is empty. 
!       :: By default it is 10.
!
!     DS_get_dim_status(dim_name)
!       :: Return an integer that specifies the status of a given dimension
!       :: dim_status = -1 : Dimension is not registered.
!       :: dim_status =  0 : Dimension exists but the reference grid is
!       ::                   empty and the running grid does not yet
!       ::                   have all its bins filled with min_point.
!       :: dim_status =  1 : Dimension exists with a non-empty 
!       ::                   reference grid and a running grid with all
!       ::                   bins filled with more than min_points entries.
!    
!     DS_set_grid_mode(grid_name,grid_mode)
!       :: Sets the kind of reference grid which is currently active.
!       :: grid_mode = 'default' : This means that the reference grid holds 
!       ::   the same kind of weights than the running grid. When the reference
!       ::   grid will be updated, the running grid will be *combined* with
!       ::   the reference grid, and not overwritten by it.
!       :: grid_mode = 'init' : This means that the reference grid is used for
!       ::   initialisation, and its weights do not compare with those put
!       ::   in the running grid. When updated, the reference grid will 
!       ::   therefore be *overwritten* by the running grid.
!
!
!     DS_get_damping_for_grid(grid_name, small_contrib, damping_power)
!       :: Returns the current value stored in the run grid for
!       :: dimension grid_name of what are the parameter damping the
!       :: bin with small contributions and whose Jacobian can
!       :: potentially be very large. See the definition of these 
!       :: parameters for a description of the procedure.
!
!     DS_set_damping_for_grid(grid_name, small_contrib, damping_power)
!       :: Sets the value for both the ref and running grid of the
!       :: dimension grid_name of what are the parameter damping the
!       :: bin with small contributions and whose Jacobian can
!       :: potentially be very large. See the definition of these 
!       :: parameters for a description of the procedure.
!
      module DiscreteSampler

      use StringCast

!     Global options for the module
      logical    DS_verbose
      save DS_verbose
      logical    DS_tolerate_zero_norm
      save DS_tolerate_zero_norm
!     An allocatable to mark initialization
      logical, dimension(:), allocatable  :: DS_isInitialized

!     This parameter sets how large must be the sampling bar when
!     displaying information about a dimension
      integer samplingBarWidth
      parameter (samplingBarWidth=80)

!     Attributes identifying a bin
!     For now just an integer
      type binID
        integer id
      endtype
!     And an easy way to create a binIDs
      interface assignment (=)
        module procedure  binID_from_binID
        module procedure  binID_from_integer
      end interface assignment (=)
!     Define and easy way of comparing binIDs 
      interface operator (==)
        module procedure  equal_binID
      end interface operator (==)

!     Information relevant to a bin
      type bin
        real*8 weight
!       Sum of the squared weights, to compute the variance
        real*8 weight_sqr
!       Sum of the absolute value of the weights put in this bin,
!       necessary when in presence of negative weights.
        real*8 abs_weight
        integer n_entries
!       Practical to be able to identify a bin by its id
        type(binID) bid
      endtype

!     Define and easy way of adding Bins 
      interface operator (+)
        module procedure  DS_combine_two_bins
      end interface operator (+)

      type sampledDimension
!       The grid_mode can take the following values
!       grid_mode = 1 : This means that the reference grid holds the
!         same kind of weights than the running grid. When the reference
!         grid will be updated, the running grid will be *combined* with
!         the reference grid, and not overwrite it.
!       grid_mode = 2 : This means that the reference grid is used for
!         initialisation, and its weights do not compare with those put
!         in the running grid. When updated, the reference grid will 
!         therefore be *overwritten* by the running grid.
        integer                               :: grid_mode
!
!       Treat specially bin with a contribution (i.e. weight) worth less than 
!       'small_contrib_threshold' of the averaged contributionover all bins.
!       For those, we sample according to the square root (or the specified power 
!       'damping power' of the difference between the reference value corresponding 
!       to the chosen mode and the small_contrib_threshold.
!       In this way, we are less sensitive to possible large fluctuations 
!       of very suppressed contributions for which the Jacobian would be 
!       really big. However, the square-root is such that a really
!       suppressed contribution at the level of numerical precision
!       would still never be probed.
!       Notice that this procedure does *not* change the weight in the
!       bin, but only how it is used for bin picking.
        real*8                                :: small_contrib_threshold
        real*8                                :: damping_power
!       Minimum number of points to probe each bin with when the reference
!       grid is empty. Once each bin has been probed that many times, the
!       subroutine DS_get_point will use a uniform distribution
        integer                               :: min_bin_probing_points
!       Keep track of the norm (i.e. sum of all weights) and the total
!       number of points for ease and optimisation purpose
        real*8                                :: norm
!       The sum of the absolute value of the weight in each bin
        real*8                                :: abs_norm
!       The sum of the variance of the weight in each bin
        real*8                                :: variance_norm
!       The sum of the squared weights in each bin
        real*8                                :: norm_sqr    
        integer                               :: n_tot_entries
!       A handy way of referring to the dimension by its name rather than
!       an index.
        character, dimension(:), allocatable  :: dimension_name
!       Bins of the grid
        type(bin) , dimension(:), allocatable :: bins
      endtype sampledDimension

!     This stores the overall discrete reference grid
      type(sampledDimension), dimension(:), allocatable :: ref_grid

!       This is the running grid, whose weights are being updated for each point
!       but not yet used for the sampling. The user must call the 'update'
!       function for the running grid to be merged to the reference one.
      type(sampledDimension), dimension(:), allocatable :: run_grid

      interface DS_add_entry
        module procedure DS_add_entry_with_BinID
        module procedure DS_add_entry_with_BinIntID
      end interface DS_add_entry ! DS_add_entry

      interface DS_print_global_info
        module procedure DS_print_dim_global_info_from_name
        module procedure DS_print_dim_global_info_from_void
      end interface ! DS_print_dim_global_info

      interface DS_add_bin
        module procedure DS_add_bin_with_binID
        module procedure DS_add_bin_with_IntegerID
        module procedure DS_add_bin_with_void
      end interface ! DS_add_bin

      interface DS_remove_bin
        module procedure DS_remove_bin_withIntegerID
        module procedure DS_remove_bin_withBinID
      end interface ! DS_remove_bin

      interface DS_get_bin
        module procedure DS_get_bin_from_binID
        module procedure DS_get_bin_from_binID_and_dimName
      end interface ! DS_get_bin

      interface DS_update_grid
        module procedure DS_update_grid_with_dim_name
        module procedure DS_update_all_grids
        module procedure DS_update_grid_with_dim_index
      end interface ! DS_update_grid

      interface DS_write_grid
        module procedure DS_write_grid_with_filename
        module procedure DS_write_grid_with_streamID
      end interface ! DS_write_grid

      interface DS_load_grid
        module procedure DS_load_grid_with_filename
        module procedure DS_load_grid_with_streamID
      end interface ! DS_load_grid

      interface DS_dim_index
        module procedure DS_dim_index_default
        module procedure DS_dim_index_with_force
        module procedure DS_dim_index_default_with_chararray
        module procedure DS_dim_index_with_force_with_chararray
      end interface ! DS_dim_index

      interface DS_bin_index
        module procedure DS_bin_index_default
        module procedure DS_bin_index_with_force
      end interface ! DS_bin_index

      interface DS_get_point
        module procedure DS_get_point_with_integerBinID
        module procedure DS_get_point_with_BinID
      end interface ! DS_get_point

      contains

!       ---------------------------------------------------------------
!       This subroutine is simply the logger of this module
!       ---------------------------------------------------------------
        subroutine DS_initialize(tolerate_zero_norm, verbose)
        implicit none
!         
!         Subroutine arguments
!         
          logical, optional, intent(in)            :: tolerate_zero_norm
          logical, optional, intent(in)            :: verbose 

          if (allocated(DS_isInitialized)) then
            write(*,*) "DiscreteSampler:: Error: The DiscreteSampler"//
     &        " module can only be initialized once."
            stop 1
          else
            allocate(DS_isInitialized(1))
          endif
          if (present(verbose)) then
            DS_verbose = verbose
          else
            DS_verbose = .False.
          endif

          if (present(tolerate_zero_norm)) then
            DS_tolerate_zero_norm = tolerate_zero_norm
          else
            DS_tolerate_zero_norm = .True.
          endif

!         Re-instore the if statement below if too annoying
!          if(DS_verbose) then
            write(*,*) ''
            write(*,*) '********************************************'
            write(*,*) '* You are using the DiscreteSampler module *'
            write(*,*) '*      part of the MG5_aMC framework       *'
            write(*,*) '*         Author: Valentin Hirschi         *'
            write(*,*) '********************************************'
            write(*,*) ''
!          endif

        end subroutine DS_initialize

!       ---------------------------------------------------------------
!       This subroutine is simply the logger of this module
!       ---------------------------------------------------------------

        subroutine DS_Logger(msg)
        implicit none
!         
!         Subroutine arguments
!         
          character(len=*), intent(in)        :: msg

          if (DS_verbose) write(*,*) msg

        end subroutine DS_Logger

!       ---------------------------------------------------------------
!       This subroutine clears the module and reinitialize all data 
!       ---------------------------------------------------------------
        subroutine DS_clear()
          call DS_deallocate_grid(ref_grid)
          call DS_deallocate_grid(run_grid)
        end subroutine DS_clear

        subroutine DS_deallocate_grid(grid)
          integer i
          type(sampledDimension), dimension(:), allocatable,
     &                                            intent(inout) :: grid
          if (allocated(grid)) then
            do i = 1,size(grid)
              if (allocated(grid(i)%bins)) then
                deallocate(grid(i)%bins)
              endif
              if (allocated(grid(i)%dimension_name)) then
                deallocate(grid(i)%dimension_name)
              endif
            enddo
            deallocate(grid)            
          endif
        end subroutine DS_deallocate_grid

!       ---------------------------------------------------------------
!       This subroutine takes care of registering a new dimension in
!       the DSampler module by characterizin it by its name and number
!       of bins.
!       ---------------------------------------------------------------
        subroutine DS_register_dimension(dim_name,n_bins,all_grids)
        implicit none
!         
!         Subroutine arguments
!        
          integer , intent(in)                :: n_bins
          character(len=*), intent(in)        :: dim_name
          logical , optional                  :: all_grids
!
!         Local variables
!
          logical                             :: do_all_grids
!
!         Begin code
!
!         Make sure the module is initialized
          if (.not.allocated(DS_isInitialized)) then
              call DS_initialize()
          endif
          if (present(all_grids)) then
            do_all_grids = all_grids
          else
            do_all_grids = .True.
          endif
          if (do_all_grids) then
            call DS_add_dimension_to_grid(ref_grid, dim_name, n_bins)
          endif
          call DS_add_dimension_to_grid(run_grid, dim_name, n_bins)

          call DS_Logger("DiscreteSampler:: Successfully registered "//
     $    "dimension '"//dim_name//"' ("//TRIM(toStr(n_bins))//' bins)')

        end subroutine DS_register_dimension

!       ---------------------------------------------------------------
!       This subroutine registers a dimension to a given grid 
!       ---------------------------------------------------------------
        subroutine DS_add_dimension_to_grid(grid, dim_name, n_bins)
        implicit none
!         
!         Subroutine arguments
!
          type(sampledDimension), dimension(:), allocatable,
     &      intent(inout)                          :: grid
          integer , intent(in)                     :: n_bins
          character(len=*), intent(in)             :: dim_name
!
!         Local variables
!
          integer                                           :: dim_index
          type(sampledDimension), dimension(:), allocatable :: tmp
          integer i
!
!         Begin code
!
!         Make sure the module is initialized
          if (.not.allocated(DS_isInitialized)) then
              call DS_initialize()
          endif
          if(allocated(grid)) then
            dim_index = DS_dim_index(grid,dim_name,.True.)
            if (dim_index.ne.-1) then
               write(*,*) 'DiscreteSampler:: Error, the dimension'//
     $              " with name '"//dim_name//"' is already registered."
               stop 1
            endif
          endif

!         Either allocate the discrete grids or append a dimension 
          if (allocated(grid)) then
            allocate(tmp(size(grid)))
            do i=1, size(grid)
              call DS_copy_dimension(grid(i), tmp(i))
            enddo
            call DS_deallocate_grid(grid)
            allocate(grid(size(tmp)+1))
            do i=1, size(tmp)
              call DS_copy_dimension(tmp(i), grid(i))
            enddo
            call DS_deallocate_grid(tmp)
          else
            allocate(grid(1))
          endif
!         Now we can fill in the appended element with the
!         characteristics of the dimension which must be added
          allocate(grid(size(grid))%bins(n_bins))
          allocate(grid(size(grid))%dimension_name(len(dim_name)))
!         Initialize the values of the grid with default
          call DS_initialize_dimension(grid(size(grid)))
!         For the string assignation, I have to it character by
!         character.
          do i=1, len(dim_name)
            grid(size(grid))%dimension_name(i) = dim_name(i:i)
          enddo

        end subroutine DS_add_dimension_to_grid

!       ----------------------------------------------------------------------
!       Copy a dimension from source to target, making sure to allocate  
!       ----------------------------------------------------------------------
        subroutine DS_copy_dimension(source, trget)
          type(sampledDimension), intent(out)   :: trget
          type(sampledDimension), intent(in)    :: source
          integer i

          if (allocated(trget%bins)) then
            deallocate(trget%bins)
          endif
          allocate(trget%bins(size(source%bins)))
          do i=1,size(source%bins)
            call DS_copy_bin(source%bins(i),trget%bins(i))
          enddo
          if (allocated(trget%dimension_name)) then
            deallocate(trget%dimension_name)
          endif
          allocate(trget%dimension_name(size(source%dimension_name)))
          do i=1, size(source%dimension_name)
            trget%dimension_name(i) = source%dimension_name(i)
          enddo
          trget%norm                    = source%norm
          trget%abs_norm                = source%abs_norm
          trget%variance_norm           = source%variance_norm
          trget%norm_sqr                = source%norm_sqr
          trget%n_tot_entries           = source%n_tot_entries 
          trget%min_bin_probing_points  = source%min_bin_probing_points
          trget%grid_mode               = source%grid_mode
          trget%damping_power           = source%damping_power
          trget%small_contrib_threshold = source%small_contrib_threshold
        end subroutine DS_copy_dimension

!       ----------------------------------------------------------------------
!       This subroutine removes a dimension at index dim_index from a given grid 
!       ----------------------------------------------------------------------
        subroutine DS_remove_dimension(dim_name)
        implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in) :: dim_name
!
!         Local variables
!
          integer         :: ref_dim_index, run_dim_index
!
!         Begin code
!
          ref_dim_index = DS_dim_index(ref_grid, dim_name)
          run_dim_index = DS_dim_index(run_grid, dim_name)
          call DS_remove_dimension_from_grid(ref_grid, ref_dim_index)
          call DS_remove_dimension_from_grid(run_grid, run_dim_index)
        end subroutine DS_remove_dimension
!

!       ----------------------------------------------------------------------
!       This subroutine removes a dimension at index dim_index from a given grid 
!       ----------------------------------------------------------------------
        subroutine DS_remove_dimension_from_grid(grid, dim_index)
        implicit none
!         
!         Subroutine arguments
!
          type(sampledDimension), dimension(:), allocatable,
     &      intent(inout)                          :: grid
          integer, intent(in)                      :: dim_index
!
!         Local variables
!
          type(sampledDimension), dimension(:), allocatable :: tmp
          integer i
!
!         Begin code
!

          allocate(tmp(size(grid)-1))
          do i=1,dim_index-1
            call DS_copy_dimension(grid(i), tmp(i))
          enddo
          do i=dim_index+1,size(grid)
            call DS_copy_dimension(grid(i), tmp(i-1))
          enddo
          call DS_deallocate_grid(grid)
          allocate(grid(size(tmp)))
          do i=1,size(tmp)
            call DS_copy_dimension(tmp(i), grid(i))
          enddo
          call DS_deallocate_grid(tmp)
        end subroutine DS_remove_dimension_from_grid

!       ---------------------------------------------------------------
!       This subroutine takes care of reinitializing a given dimension
!       with default values
!       ---------------------------------------------------------------
        subroutine DS_reinitialize_dimension(d_dim)
        implicit none
!         
!         Subroutine arguments
!
          type(sampledDimension), intent(inout) :: d_dim 
!
!         Local variables
!

          integer i
!
!         Begin code
!
          do i=1, size(d_dim%bins)
            call DS_reinitialize_bin(d_dim%bins(i))
          enddo
          d_dim%norm_sqr        = 0.0d0
          d_dim%abs_norm        = 0.0d0
          d_dim%variance_norm   = 0.0d0
          d_dim%norm            = 0.0d0
          d_dim%n_tot_entries   = 0

        end subroutine DS_reinitialize_dimension

!       ---------------------------------------------------------------
!       This subroutine takes care of initializing a given dimension
!       with default values
!       ---------------------------------------------------------------
        subroutine DS_initialize_dimension(d_dim)
        implicit none
!         
!         Subroutine arguments
!
          type(sampledDimension), intent(inout) :: d_dim 
!
!         Local variables
!

          integer i
!
!         Begin code
!
          call DS_reinitialize_dimension(d_dim)
          do i=1, size(d_dim%bins)
            call DS_initialize_bin(d_dim%bins(i))
          enddo
          do i= 1, len(d_dim%dimension_name)
            d_dim%dimension_name(i:i) = ' '
          enddo
!         By default require each bin to be probed by 10 points
!         before a uniform distribution is used when the reference grid
!         is empty
          d_dim%min_bin_probing_points  = 10
          d_dim%grid_mode               = 1
!         Turn off the damping of small contributions by default
          d_dim%small_contrib_threshold = 0.0d0
          d_dim%damping_power           = 0.5d0
!         By default give sequential ids to the bins
          do i=1, size(d_dim%bins)
            d_dim%bins(i)%bid = i
          enddo
        end subroutine DS_initialize_dimension

!       ---------------------------------------------------------------
!       This subroutine takes care of reinitializing a given bin 
!       ---------------------------------------------------------------
        subroutine DS_initialize_bin(d_bin)
        implicit none
!         
!         Subroutine arguments
!
          type(bin), intent(inout) :: d_bin
!
!         Begin code
!
          call DS_reinitialize_bin(d_bin)
          d_bin%bid         = 0
        end subroutine DS_initialize_bin

!       ---------------------------------------------------------------
!       This subroutine takes care of initializing a given bin 
!       ---------------------------------------------------------------
        subroutine DS_reinitialize_bin(d_bin)
        implicit none
!         
!         Subroutine arguments
!
          type(bin), intent(inout) :: d_bin
!
!         Begin code
!
          d_bin%weight_sqr = 0.0d0
          d_bin%abs_weight = 0.0d0          
          d_bin%weight     = 0.0d0
          d_bin%n_entries  = 0
        end subroutine DS_reinitialize_bin

!       ---------------------------------------------------------------
!       Set the minimum number of point for which the bins must be
!       probed before a uniform distribution is used when the reference
!       grid is empty
!       ---------------------------------------------------------------
        subroutine DS_set_min_points(min_points, dim_name)
        implicit none
!         
!         Subroutine arguments
!
!     
          integer, intent(in)                      :: min_points
          character(len=*), intent(in), optional   :: dim_name
!
!         Local variables
!
          integer i
!
!         Begin Code
!
          if(present(dim_name)) then
            ref_grid(DS_dim_index(ref_grid,dim_name))%
     &                               min_bin_probing_points = min_points
            run_grid(DS_dim_index(ref_grid,dim_name))% 
     &                               min_bin_probing_points = min_points
          else
            do i=1,size(ref_grid)
              ref_grid(i)%min_bin_probing_points = min_points
              run_grid(i)%min_bin_probing_points = min_points
            enddo
          endif
        end subroutine DS_set_min_points

!       ---------------------------------------------------------------
!       Returns an integer that specifies the status of a given dimension
!       dim_status = -1 : Dimension is not registered.
!       dim_status =  0 : Dimension exists but the reference grid is
!                         empty and the running grid does not yet
!                         have all its bins filled with min_point.
!       dim_status =  1 : Dimension exists with a non-empty 
!                         reference grid and a running grid with all
!                         bins filled with more than min_points entries.
!       ---------------------------------------------------------------
        function DS_get_dim_status(grid_name)
        implicit none
!         
!         Function arguments
!
          character(len=*), intent(in)     :: grid_name
          integer                          :: DS_get_dim_status
!
!         Local variables
!
          integer                           :: ref_grid_index
          integer                           :: run_grid_index
          integer                           :: int_grid_mode
          type(Bin)                         :: mRunBin
          integer                           :: i
!         
!         Begin code
!
          ref_grid_index = DS_dim_index(ref_grid, grid_name, .True.)
          run_grid_index = DS_dim_index(run_grid, grid_name, .True.)
          if (ref_grid_index.eq.-1.or.run_grid_index.eq.-1) then
            DS_get_dim_status = -1
            return
          endif
          if (ref_grid(ref_grid_index)%n_tot_entries.ne.0) then
            DS_get_dim_status = 1
            return
          endif    
         
!         If the running grid has zero entries, then consider the grid
!         uninitialized
          if(size(run_grid(run_grid_index)%bins).eq.0) then
            DS_get_dim_status = 0
            return
          endif

          do i=1,size(ref_grid(ref_grid_index)%bins)
            mRunBin = DS_get_bin(run_grid(run_grid_index)%bins,
     &                             ref_grid(ref_grid_index)%bins(i)%bid)
            if (mRunBin%n_entries.lt.ref_grid(ref_grid_index)
     &                                    %min_bin_probing_points) then
              DS_get_dim_status = 0
              return
            endif
          enddo

          DS_get_dim_status = 1
          return
        end function DS_get_dim_status

!       ---------------------------------------------------------------
!       Access function to modify the damping parameters of small
!       contributions
!       ---------------------------------------------------------------
        subroutine DS_set_damping_for_grid(grid_name, in_small_contrib,
     &                                                 in_damping_power)
        implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)     :: grid_name          
          real*8, intent(in)               :: in_small_contrib
          real*8, intent(in)               :: in_damping_power
!
!         Local variables
!
          integer                          :: ref_grid_index
          integer                          :: run_grid_index
!         
!         Begin code
!
          ref_grid_index = DS_dim_index(ref_grid, grid_name, .True.)
          if (ref_grid_index.eq.-1) then
            write(*,*) "DiscreteSampler:: Error in 'DS_set_damping_"//
     &        "for_grid', dimension '"//grid_name//"' could not be"//
     &        " found in the reference grid."
              stop 1
          endif
          run_grid_index = DS_dim_index(run_grid, grid_name, .True.)
          if (run_grid_index.eq.-1) then
            write(*,*) "DiscreteSampler:: Error in 'DS_set_damping_"//
     &        "for_grid', dimension '"//grid_name//"' could not be"//
     &        " found in the running grid."
              stop 1
          endif

!         Limit arbitrarily at 50% because anything above that really
!         breaks the assumption of a small grid deformation not 
!         significantly affecting the averaged contribution taked as
!         a threshold.
          if (in_small_contrib.lt.0.0d0.or.
     &                                  in_small_contrib.gt.0.5d0) then
            write(*,*) "The small relative contribution threshold "//
     &      toStr_real_with_ndig(in_small_contrib,3) 
     &      //") given in argument of the function 'DS_set_damping_"//
     &      "for_grid' must be >=0.0 and <= 0.5."
            stop 1
          endif

          if (in_damping_power.lt.0.0d0.or.
     &                                  in_damping_power.gt.1.0d0) then
            write(*,*) "The damping power ("//
     &      toStr_real_with_ndig(in_damping_power,3) 
     &      //") given in argument of the function 'DS_set_damping_"//
     &      "for_grid' must be >= 0.0 and <= 1.0."
            stop 1
          endif

          ref_grid(ref_grid_index)%small_contrib_threshold = 
     &                                                  in_small_contrib
          ref_grid(ref_grid_index)%damping_power = in_damping_power
          run_grid(run_grid_index)%small_contrib_threshold = 
     &                                                  in_small_contrib
          run_grid(run_grid_index)%damping_power = in_damping_power
        end subroutine DS_set_damping_for_grid

!       ---------------------------------------------------------------
!       Access function to access the damping parameters for small
!       contributions stored in the reference grid
!       ---------------------------------------------------------------
        subroutine DS_get_damping_for_grid(grid_name, out_small_contrib,
     &                                                out_damping_power)
        implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)      :: grid_name          
          real*8, intent(out)               :: out_small_contrib
          real*8, intent(out)               :: out_damping_power        
!
!         Local variables
!
          integer                           :: run_grid_index
!         
!         Begin code
!
          run_grid_index = DS_dim_index(run_grid, grid_name, .True.)
          if (run_grid_index.eq.-1) then
            write(*,*) "DiscreteSampler:: Error in 'DS_get_damping_"//
     &        "for_grid', dimension '"//grid_name//"' could not be"//
     &        " found in the running grid."
              stop 1
          endif

          out_small_contrib = run_grid(run_grid_index)%
     &                                           small_contrib_threshold
          out_damping_power = run_grid(run_grid_index)%damping_power

        end subroutine DS_get_damping_for_grid

!       ---------------------------------------------------------------
!       Access function to modify the mode of the reference grid:
!       grid_mode = 'default' : This means that the reference grid holds 
!         the same kind of weights than the running grid. When the reference
!         grid will be updated, the running grid will be *combined* with
!         the reference grid, and not overwritten by it.
!       grid_mode = 'init' : This means that the reference grid is used for
!         initialisation, and its weights do not compare with those put
!         in the running grid. When updated, the reference grid will 
!         therefore be *overwritten* by the running grid.
!       ---------------------------------------------------------------
        subroutine DS_set_grid_mode(grid_name, grid_mode)
        implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)     :: grid_mode
          character(len=*), intent(in)     :: grid_name          
!
!         Local variables
!
          integer                           :: ref_grid_index
          integer                           :: int_grid_mode
!         
!         Begin code
!
          ref_grid_index = DS_dim_index(ref_grid, grid_name, .True.)
          if (ref_grid_index.eq.-1) then
            write(*,*) 'DiscreteSampler:: Error in DS_set_grid_mode, '//
     &        "dimension '"//grid_name//"' could not be found in the "//
     &        "reference grid."
              stop 1
          endif
          if (grid_mode.eq.'init') then
            int_grid_mode=2
          elseif (grid_mode.eq.'default') then
            int_grid_mode=1
          else
            write(*,*) 'DiscreteSampler:: Error in DS_set_grid_mode, '//
     &        " grid_mode '"//grid_mode//"' not recognized. It must "//
     &        " be one of the following: 'default', 'init'."
              stop 1
          endif

!         Notice that we don't change the mode of the running_grid
!         because in this way, after any DS_update() is done, the
!         ref_grid will automatically turn its mode to 'default' because
!         it inherits the attribute of the running grid. 
!         However, if the running grid was loaded from a saved grid file
!         then it might be that the run_grid also has the grid_mode set
!         to 'initialization' which will then correctly be copied to the
!         ref_grid after the DS_update() performed at the end of
!         DS_load() which correctly reproduce the state of the
!         DiscreteSampler module at the time it wrote the grids.
          ref_grid(ref_grid_index)%grid_mode = int_grid_mode
        end subroutine DS_set_grid_mode

!       ---------------------------------------------------------------
!       Dictionary access-like subroutine to obtain a grid from its name
!       ---------------------------------------------------------------

        function DS_get_dimension(grid, dim_name)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in), allocatable
     &                                  :: grid
          character(len=*), intent(in)  :: dim_name
          type(sampledDimension)        :: DS_get_dimension
!
!         Begin code
!
          DS_get_dimension = grid(DS_dim_index(grid,dim_name))
        end function DS_get_dimension

!       ---------------------------------------------------------------
!       Returns the index of a bin with mBinID in the list bins
!       ---------------------------------------------------------------
        function DS_bin_index_default(bins, mBinID)
        implicit none
!         
!         Function arguments
!
          type(Bin), dimension(:), intent(in)  
     &                                  :: bins
          type(BinID)                   :: mBinID
          integer                       :: DS_bin_index_default
!
!         Begin code
!
          DS_bin_index_default = DS_bin_index_with_force(bins,mBinID,
     &                                                          .False.)
        end function DS_bin_index_default

        function DS_bin_index_with_force(bins, mBinID,force)
        implicit none
!         
!         Function arguments
!
          type(Bin), dimension(:), intent(in)  
     &                                  :: bins
          type(BinID)                   :: mBinID
          integer                       :: DS_bin_index_with_force
          logical                       :: force
!
!         Local variables
!
          integer i
!
!         Begin code
!
!         For efficiency first look at index mBinID%id
          if (size(bins).ge.mBinID%id) then
            if (bins(mBinID%id)%bid==mBinID) then
              DS_bin_index_with_force = mBinID%id
              return
            endif
          endif
          
          DS_bin_index_with_force = -1
          do i = 1, size(bins)
            if (bins(i)%bid==mBinID) then
              DS_bin_index_with_force = i
              return              
            endif
          enddo
          if (DS_bin_index_with_force.eq.-1.and.(.not.Force)) then
            write(*,*) 'DiscreteSampler:: Error in function bin_index'//
     &        "(), bin with BinID '"//trim(DS_toStr(mBinID))
     &        //"' not found."
            stop 1
          endif
        end function DS_bin_index_with_force

!       ---------------------------------------------------------------
!       Functions of the interface get_bin facilitating the access to a
!       given bin.
!       ---------------------------------------------------------------
        
        function DS_get_bin_from_binID(bins, mBinID)
        implicit none
!         
!         Function arguments
!
          type(Bin), dimension(:), intent(in)  
     &                                  :: bins
          type(BinID)                   :: mBinID
          type(Bin)                     :: DS_get_bin_from_binID
!
!         Local variables
!
          integer i
!
!         Begin code
!
          DS_get_bin_from_binID = bins(DS_bin_index(bins,mBinID))
        end function DS_get_bin_from_binID

        function DS_get_bin_from_binID_and_dimName(grid, dim_name,
     &                                                          mBinID)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in), allocatable
     &                                  :: grid
          character(len=*), intent(in)  :: dim_name
          type(BinID)                   :: mBinID
          type(Bin)             :: DS_get_bin_from_binID_and_dimName
!
!         Local variables
!
          integer i
          type(SampledDimension)        :: m_dim
!
!         Begin code
!
          m_dim = DS_get_dimension(grid,dim_name)
          DS_get_bin_from_binID_and_dimName = DS_get_bin_from_binID(
     &                  m_dim%bins,mBinID)
        end function DS_get_bin_from_binID_and_dimName


!       ---------------------------------------------------------------
!       Add a new weight to a certan bin (characterized by either its 
!       binID or index)
!       ---------------------------------------------------------------
        subroutine DS_add_entry_with_BinID(dim_name, mBinID, weight,
     &                                                            reset)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)  :: dim_name
          type(BinID)                   :: mBinID
          real*8                        :: weight
          logical, optional             :: reset
!
!         Local variables
!
          integer dim_index, bin_index
          type(Bin)                     :: newBin
          integer                       :: n_entries
          logical                       :: opt_reset          
!
!         Begin code
!
          if (present(reset)) then
            opt_reset = reset
          else
            opt_reset = .False.
          endif

          dim_index = DS_dim_index(run_grid, dim_name, .TRUE.)
          if (dim_index.eq.-1) then
              call DS_Logger('Dimension  '//dim_name//
     &        ' does not exist in the run grid. Creating it now.')
              call DS_register_dimension(dim_name,0)
              dim_index = DS_dim_index(run_grid, dim_name)
          endif

          bin_index = DS_bin_index(
     &                           run_grid(dim_index)%bins,mBinID,.TRUE.)
          if (bin_index.eq.-1) then
              call DS_Logger('Bin with binID '//trim(DS_toStr(mBinID))//
     &        ' does not exist in the run grid. Creating it now.')
              call DS_reinitialize_bin(newBin)
              newBin%bid = mBinID
              call DS_add_bin_to_bins(run_grid(dim_index)%bins,newBin)
              bin_index = DS_bin_index(run_grid(dim_index)%bins,mBinID)
          endif

!         First remove bin from global cumulative information in the grid
          run_grid(dim_index)%norm = run_grid(dim_index)%norm -
     &                   run_grid(dim_index)%bins(bin_index)%weight
          run_grid(dim_index)%norm_sqr = run_grid(dim_index)%norm_sqr -
     &                   run_grid(dim_index)%bins(bin_index)%weight_sqr
          run_grid(dim_index)%abs_norm = run_grid(dim_index)%abs_norm -
     &                   run_grid(dim_index)%bins(bin_index)%abs_weight
          run_grid(dim_index)%variance_norm = 
     &              run_grid(dim_index)%variance_norm -
     &              DS_bin_variance(run_grid(dim_index)%bins(bin_index))
          run_grid(dim_index)%n_tot_entries = 
     &              run_grid(dim_index)%n_tot_entries -
     &                     run_grid(dim_index)%bins(bin_index)%n_entries
!         Update the information directly stored in the bin
          if(.not.opt_reset) then
            n_entries = run_grid(dim_index)%bins(bin_index)%n_entries
            run_grid(dim_index)%bins(bin_index)%weight = 
     &        (run_grid(dim_index)%bins(bin_index)%weight*n_entries
     &                                           + weight)/(n_entries+1)
            run_grid(dim_index)%bins(bin_index)%weight_sqr = 
     &        (run_grid(dim_index)%bins(bin_index)%weight_sqr*n_entries
     &                                        + weight**2)/(n_entries+1)
            run_grid(dim_index)%bins(bin_index)%abs_weight = 
     &        (run_grid(dim_index)%bins(bin_index)%abs_weight*n_entries
     &                                      + abs(weight))/(n_entries+1)
            run_grid(dim_index)%bins(bin_index)%n_entries = n_entries+1
          else
            run_grid(dim_index)%bins(bin_index)%weight = weight
            run_grid(dim_index)%bins(bin_index)%weight_sqr = weight**2
            run_grid(dim_index)%bins(bin_index)%abs_weight = abs(weight)
            run_grid(dim_index)%bins(bin_index)%n_entries = 
     &                       run_grid(dim_index)%min_bin_probing_points
          endif
!         Now add the bin information back to the info in the grid
          run_grid(dim_index)%norm = run_grid(dim_index)%norm +
     &                   run_grid(dim_index)%bins(bin_index)%weight
          run_grid(dim_index)%norm_sqr = run_grid(dim_index)%norm_sqr +
     &                   run_grid(dim_index)%bins(bin_index)%weight_sqr
          run_grid(dim_index)%abs_norm = run_grid(dim_index)%abs_norm +
     &                   run_grid(dim_index)%bins(bin_index)%abs_weight
          run_grid(dim_index)%variance_norm = 
     &              run_grid(dim_index)%variance_norm +
     &              DS_bin_variance(run_grid(dim_index)%bins(bin_index))
          run_grid(dim_index)%n_tot_entries = 
     &              run_grid(dim_index)%n_tot_entries +
     &                     run_grid(dim_index)%bins(bin_index)%n_entries

        end subroutine DS_add_entry_with_BinID

        subroutine DS_add_entry_with_BinIntID(dim_name, BinIntID,
     &                                                weight, reset)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)  :: dim_name
          integer                       :: BinIntID
          real*8                        :: weight
          logical, optional             :: reset          
!
!         Begin code
!
          if (present(reset)) then
            call DS_add_entry_with_BinID(dim_name, DS_BinID(BinIntID),
     &                                                    weight, reset)
          else
            call DS_add_entry_with_BinID(dim_name, DS_BinID(BinIntID),
     &                                                           weight)
          endif
        end subroutine DS_add_entry_with_BinIntID

!       ---------------------------------------------------------------
!       Prints out all informations for dimension of index d_index, or
!       name d_name.
!       ---------------------------------------------------------------
        subroutine DS_print_dim_global_info_from_void()
          integer i
          if(allocated(ref_grid).and.allocated(run_grid)) then
            do i = 1, size(ref_grid)
              call DS_print_dim_global_info_from_name(
     &                          trim(toStr(ref_grid(i)%dimension_name)))
            enddo
          else
            write(*,*) 'DiscreteSampler:: No dimension setup yet.'
          endif
        end subroutine DS_print_dim_global_info_from_void

        subroutine DS_print_dim_global_info_from_name(d_name)
        implicit none

!         Function arguments
!
          character(len=*), intent(in) :: d_name
!
!         Local variables
!
          integer n_bins, ref_dim_index, run_dim_index
!
!         Begin code
!
!         The running grid and ref grid must have the same number of
!         bins at this stage

          if(.not.(allocated(ref_grid).and.allocated(run_grid))) then
            write(*,*) 'DiscreteSampler:: No dimension setup yet.'
            return
          endif

          ref_dim_index = DS_dim_index(ref_grid,d_name,.TRUE.)
          run_dim_index = DS_dim_index(run_grid,d_name,.TRUE.)

          if (ref_dim_index.ne.-1) then
            n_bins = size(ref_grid(DS_dim_index(ref_grid,d_name))%bins)
          elseif (run_dim_index.ne.-1) then
            n_bins = size(run_grid(DS_dim_index(run_grid,d_name))%bins)
          else
            write(*,*) 'DiscreteSampler:: No grid registered for name'//
     &        " '"//d_name//"'."
            return
          endif  

          write(*,*) "DiscreteSampler:: ========================"//
     &       "=========================="
          write(*,*) "DiscreteSampler:: Information for dimension '"//
     &                     d_name//"' ("//trim(toStr(n_bins))//" bins):"
          write(*,*) "DiscreteSampler:: -> Grids status ID : "//
     &                            trim(toStr(DS_get_dim_status(d_name)))
          if (ref_dim_index.ne.-1) then
            write(*,*) "DiscreteSampler:: || Reference grid "
            select case(ref_grid(ref_dim_index)%grid_mode)
              case(1)
                write(*,*) "DiscreteSampler::   -> Grid mode : default"
              case(2)
                write(*,*) "DiscreteSampler::   -> Grid mode : "//
     &            "initialization"
            end select
            call DS_print_dim_info(ref_grid(ref_dim_index))
          else
            write(*,*) "DiscreteSampler:: || No reference grid for "//
     &         "that dimension."
          endif
          if (run_dim_index.ne.-1) then
            write(*,*) "DiscreteSampler:: || Running grid "
            write(*,*) "DiscreteSampler::   -> Initialization "//
     &        "minimum points : "//trim(toStr(run_grid(
     &                           run_dim_index)%min_bin_probing_points))
            call DS_print_dim_info(run_grid(run_dim_index))
          else
            write(*,*) "DiscreteSampler:: || No running grid for "//
     &         "that dimension."
          endif
          write(*,*) "DiscreteSampler:: ========================"//
     &       "=========================="
        end subroutine DS_print_dim_global_info_from_name

!       ---------------------------------------------------------------
!       Print all informations related to a specific sampled dimension
!       in a given grid
!       ---------------------------------------------------------------
        subroutine DS_print_dim_info(d_dim)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), intent(in)  :: d_dim
!
!         Local variables
!
          integer i,j, curr_pos1, curr_pos2, curr_pos3
          integer n_bins, bin_width
!         Adding the minimum size for the separators '|' and binID assumed
!         of being of length 2 at most, so 10*2+11 and + 20 security :)

          character(samplingBarWidth+10*2+11+20)       :: samplingBar1
          character(samplingBarWidth+10*2+11+20)       :: samplingBar2
          character(samplingBarWidth+10*2+11+20)       :: samplingBar3
          real*8            :: tot_entries, tot_variance, tot_abs_weight
!
!         Begin code
!
!
!         Setup the sampling bars
!
          tot_entries = 0
          tot_variance = 0.0d0
          tot_abs_weight = 0.0d0
          do i=1,min(size(d_dim%bins),10)
            tot_entries = tot_entries + d_dim%bins(i)%n_entries
            tot_variance = tot_variance + DS_bin_variance(d_dim%bins(i))
            tot_abs_weight = tot_abs_weight + d_dim%bins(i)%abs_weight
          enddo
          if (d_dim%n_tot_entries.eq.0) then
            samplingBar1 = "| Empty grid |"
            samplingBar2 = "| Empty grid |"
            samplingBar3 = "| Empty grid |"
          else
            do i=1,len(samplingBar1)
              samplingBar1(i:i)=' '
            enddo
            do i=1,len(samplingBar2)
              samplingBar2(i:i)=' '
            enddo
            do i=1,len(samplingBar3)
              samplingBar3(i:i)=' '
            enddo
            samplingBar1(1:1) = '|'
            samplingBar2(1:1) = '|'
            samplingBar3(1:1) = '|'             
            curr_pos1 = 2
            curr_pos2 = 2
            curr_pos3 = 2 
            do i=1,min(10,size(d_dim%bins)) 
              samplingBar1(curr_pos1:curr_pos1+1) =
     &                             trim(DS_toStr(d_dim%bins(i)%bid))
              samplingBar2(curr_pos2:curr_pos2+1) = 
     &                             trim(DS_toStr(d_dim%bins(i)%bid))
              samplingBar3(curr_pos3:curr_pos3+1) = 
     &                             trim(DS_toStr(d_dim%bins(i)%bid))
              curr_pos1 = curr_pos1+2
              curr_pos2 = curr_pos2+2
              curr_pos3 = curr_pos3+2

              if (tot_abs_weight.ne.0.0d0) then
                bin_width = int((d_dim%bins(i)%abs_weight/
     &                                 tot_abs_weight)*samplingBarWidth)
                do j=1,bin_width
                  samplingBar1(curr_pos1+j:curr_pos1+j) = ' '
                enddo
                curr_pos1 = curr_pos1+bin_width+1
                samplingBar1(curr_pos1:curr_pos1) = '|'
                curr_pos1 = curr_pos1+1
              endif

              if (tot_entries.ne.0) then
                bin_width = int((float(d_dim%bins(i)%n_entries)/
     &                                    tot_entries)*samplingBarWidth)
                do j=1,bin_width
                  samplingBar2(curr_pos2+j:curr_pos2+j) = ' '
                enddo
                curr_pos2 = curr_pos2+bin_width+1
                samplingBar2(curr_pos2:curr_pos2) = '|'
                curr_pos2 = curr_pos2+1
              endif

              if (tot_variance.ne.0.0d0) then
                bin_width = int((DS_bin_variance(d_dim%bins(i))/
     &                                   tot_variance)*samplingBarWidth)
                do j=1,bin_width
                  samplingBar3(curr_pos3+j:curr_pos3+j) = ' '
                enddo
                curr_pos3 = curr_pos3+bin_width+1
                samplingBar3(curr_pos3:curr_pos3) = '|'
                curr_pos3 = curr_pos3+1
              endif
            enddo
            if (tot_abs_weight.eq.0.0d0) then
              samplingBar1 = "| All considered bins have zero weight |"
            endif
            if (tot_entries.eq.0) then
              samplingBar2 = "| All considered bins have no entries |"
            endif
            if (tot_variance.eq.0.0d0) then
              samplingBar3 = "| All variances are zeros in considered"//
     &        " bins. Maybe not enough entries (need at least one bin"//
     &        " with >=2 entries). |"
            endif
          endif
!
!         Write out info
!
          n_bins = size(d_dim%bins)
          
          write(*,*) "DiscreteSampler::   -> Total number of "//
     &         "entries : "//trim(toStr(d_dim%n_tot_entries))
          if (n_bins.gt.10) then
            write(*,*) "DiscreteSampler::   -> Sampled as"//
     &                                      " (first 10 bins):"
          else
            write(*,*) "DiscreteSampler::   -> Sampled as:"
          endif
          write(*,*) "DiscreteSampler::    "//trim(samplingBar2)
          write(*,*) "DiscreteSampler::   -> (norm_sqr , "//
     &      "abs_norm , norm     , variance ) :"
          write(*,*) "DiscreteSampler::      ("//
     &      trim(toStr(d_dim%norm_sqr,'Ew.3'))//", "//
     &      trim(toStr(d_dim%abs_norm,'Ew.3'))//", "//
     &      trim(toStr(d_dim%norm,'Ew.3'))//", "//
     &      trim(toStr(d_dim%variance_norm,'Ew.3'))//")"
          if (n_bins.gt.10) then
            write(*,*) "DiscreteSampler::   -> Abs weights sampled as"//
     &                                      " (first 10 bins):"
          else
            write(*,*) "DiscreteSampler::   -> Abs weights sampled as:"
          endif
          write(*,*) "DiscreteSampler::    "//trim(samplingBar1)
          if (n_bins.gt.10) then
            write(*,*) "DiscreteSampler::   -> Variance sampled as"//
     &                                      " (first 10 bins):"
          else
            write(*,*) "DiscreteSampler::   -> Variance sampled as:"
          endif
          write(*,*) "DiscreteSampler::    "//trim(samplingBar3)

        end subroutine DS_print_dim_info

!       ---------------------------------------------------------------
!         Functions to add a bin with different binID specifier
!       ---------------------------------------------------------------      
        subroutine DS_add_bin_with_IntegerID(dim_name,intID)
          implicit none
!         
!         Subroutine arguments
!
          integer, intent(in)      :: intID
          character(len=*)         :: dim_name
!
!         Begin code
!
          call DS_add_bin_with_binID(dim_name,DS_binID(intID))
        end subroutine DS_add_bin_with_IntegerID

        subroutine DS_add_bin_with_void(dim_name)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*)         :: dim_name
!
!         Local variables
!
          integer                  :: ref_size, run_size
!
!         Begin code
!
          ref_size=size(ref_grid(DS_dim_index(ref_grid,dim_name))%bins)
          run_size=size(run_grid(DS_dim_index(run_grid,dim_name))%bins)
          call DS_add_bin_with_binID(dim_name,DS_binID(
     &                                      max(ref_size, run_size)+1))
        end subroutine DS_add_bin_with_void

        subroutine DS_add_bin_with_binID(dim_name,mBinID)
          implicit none
!         
!         Subroutine arguments
!
          type(binID), intent(in)  :: mBinID
          character(len=*)         :: dim_name
!
!         Local variables
!
          type(Bin)                :: new_bin
!
!         Begin code
!
          call DS_reinitialize_bin(new_bin)
          new_bin%bid = mBinID
          call DS_add_bin_to_bins(ref_grid(DS_dim_index(ref_grid, 
     &                                          dim_name))%bins,new_bin)
          call DS_add_bin_to_bins(run_grid(DS_dim_index(run_grid, 
     &                                          dim_name))%bins,new_bin)
        end subroutine DS_add_bin_with_binID

        subroutine DS_add_bin_to_bins(bins,new_bin)
          implicit none
!         
!         Subroutine arguments
!
          type(Bin), dimension(:), allocatable, intent(inout)  
     &                             :: bins
          type(Bin)                :: new_bin
!
!         Local variables
!
          type(Bin), dimension(:), allocatable :: tmp
          integer                              :: i, bin_index
!
!         Begin code
!
          bin_index = DS_bin_index(bins,new_bin%bid,.True.)
          if (bin_index.ne.-1) then
             write(*,*)"DiscreteSampler:: Error, the bin with binID '"//
     &         trim(DS_toStr(new_bin%bid))//"' cannot be added "//
     &         "be added because it already exists."
               stop 1
          endif


          allocate(tmp(size(bins)+1))
          do i=1,size(bins)
            call DS_copy_bin(bins(i),tmp(i))
          enddo
          tmp(size(bins)+1) = new_bin
          deallocate(bins)
          allocate(bins(size(tmp)))
          do i=1,size(bins)
            call DS_copy_bin(tmp(i),bins(i))          
          enddo
          deallocate(tmp)
        end subroutine DS_add_bin_to_bins

        subroutine DS_copy_bin(source, trget)
            implicit none
            type(Bin), intent(out) :: trget
            type(Bin), intent(in)  :: source
            trget%weight     = source%weight
            trget%weight_sqr = source%weight_sqr
            trget%abs_weight = source%abs_weight
            trget%n_entries  = source%n_entries
            trget%bid        = DS_binID(source%bid%id)
        end subroutine DS_copy_bin

!       ---------------------------------------------------------------
!         Functions to remove a bin from a dimension
!       ---------------------------------------------------------------
        subroutine DS_remove_bin_withIndex(dim_name, binIndex)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)   :: dim_name
          integer, intent(in)            :: binIndex
!
!         Begin code
!

          call DS_remove_bin_from_grid(run_grid(
     &                       DS_dim_index(run_grid, dim_name)),binIndex)
        end subroutine DS_remove_bin_withIndex

        subroutine DS_remove_bin_withBinID(dim_name, mbinID)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)   :: dim_name
          type(binID), intent(in)        :: mbinID
!
!         Local variables
!
          integer                        :: ref_dim_index,run_dim_index
          integer                        :: ref_bin_index,run_bin_index
!
!         Begin code
!
          ref_dim_index = DS_dim_index(ref_grid, dim_name)
          ref_bin_index = DS_bin_index(ref_grid(ref_dim_index)%bins,
     &                                                          mbinID)
          call DS_remove_bin_from_grid(ref_grid(ref_dim_index),
     &                                                   ref_bin_index)
          run_dim_index = DS_dim_index(run_grid, dim_name)
          run_bin_index = DS_bin_index(run_grid(run_dim_index)%bins,
     &                                                          mbinID)
          call DS_remove_bin_from_grid(run_grid(run_dim_index),
     &                                                   run_bin_index)
        end subroutine DS_remove_bin_withBinID

        subroutine DS_remove_bin_withIntegerID(dim_name, mBinIntID)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)   :: dim_name
          integer, intent(in)            :: mBinIntID       
!
!         Begin code
!
          call DS_remove_bin_withBinID(dim_name,DS_binID(mBinIntID))
        end subroutine DS_remove_bin_withIntegerID

        subroutine DS_remove_bin_from_grid(grid, bin_index)
          implicit none
!         
!         Subroutine arguments
!
          type(SampledDimension), intent(inout)  :: grid
          integer, intent(in)                    :: bin_index
!
!         Local variables
!
          type(Bin), dimension(:), allocatable :: tmp
          integer                              :: i
!
!         Begin code
!

!         Update the norm, norm_sqr and the number of entries in
!         the corresponding dimension
          grid%norm = grid%norm - grid%bins(bin_index)%weight
          grid%norm_sqr = grid%norm_sqr - 
     &                                   grid%bins(bin_index)%weight_sqr
          grid%abs_norm = grid%abs_norm -
     &                                   grid%bins(bin_index)%abs_weight
          grid%variance_norm = grid%variance_norm
     &                           - DS_bin_variance(grid%bins(bin_index))
          grid%n_tot_entries = grid%n_tot_entries
     &                                  - grid%bins(bin_index)%n_entries
          allocate(tmp(size(grid%bins)-1))
          do i=1,bin_index-1
            tmp(i) = grid%bins(i)
          enddo
          do i=bin_index+1,size(grid%bins)
            tmp(i-1) = grid%bins(i)          
          enddo
          deallocate(grid%bins)
          allocate(grid%bins(size(tmp)))
          do i=1,size(tmp)
            grid%bins(i)=tmp(i)
          enddo
          deallocate(tmp)
        end subroutine DS_remove_bin_from_grid


!       ---------------------------------------------------------------
!       Function to update the reference grid with the running one
!       ---------------------------------------------------------------
        subroutine DS_update_all_grids(filterZeros)
        implicit none
!         
!         Subroutine arguments
!
          logical, optional :: filterZeros
!         
!         Local variables
!
          integer           :: i
          logical           :: do_filterZeros          
!
!         Begin code
!
          if (.not.allocated(run_grid)) then
            return
          endif
          if(present(filterZeros)) then
            do_filterZeros = filterZeros
          else
            do_filterZeros = .False.
          endif
          do i=1, size(run_grid)
            call DS_update_grid_with_dim_index(i,do_filterZeros)
          enddo
        end subroutine DS_update_all_grids

        subroutine DS_update_grid_with_dim_name(dim_name, filterZeros)
        implicit none
!         
!         Subroutine arguments
!
          character(len=*)                 :: dim_name
          logical, optional                :: filterZeros          
!         
!         Local variables
!
          integer           :: i
          logical           :: do_filterZeros
!
!         Begin code
!
          if(present(filterZeros)) then
            do_filterZeros = filterZeros
          else
            do_filterZeros = .False.
          endif
          call DS_update_grid_with_dim_index(
     &                   DS_dim_index(run_grid,dim_name),do_filterZeros)

        end subroutine DS_update_grid_with_dim_name

        subroutine DS_update_grid_with_dim_index(d_index,filterOutZeros)
        implicit none
!         
!         Subroutine arguments
!
          integer                               :: d_index
          logical                               :: filterOutZeros
!         
!         Local variables
!
          integer                               :: i, ref_d_index 
          integer                               :: ref_bin_index
          integer                               :: j, shift
          character, dimension(:), allocatable  :: dim_name
          type(BinID)                           :: mBinID
          type(Bin)                        :: new_bin, ref_bin, run_bin
          logical                          :: empty_ref_grid
!
!         Begin code
!
          allocate(dim_name(size(run_grid(d_index)%dimension_name)))
          dim_name = run_grid(d_index)%dimension_name
          call DS_Logger("Updating dimension '"//
     &                                      trim(toStr(dim_name))//"'.")

!         Start by making sure that the dimension exists in the
!         reference grid. If not, then create it.
          if (DS_dim_index(ref_grid,
     &         run_grid(d_index)%dimension_name,.True.).eq.-1) then
              call DS_Logger('Reference grid does not have dimension '//
     &                         trim(toStr(dim_name))//'. Adding it now')
              call DS_add_dimension_to_grid(ref_grid, 
     &                                        trim(toStr(dim_name)) , 0)
          endif
          ref_d_index = DS_dim_index(ref_grid, dim_name)

          empty_ref_grid = (ref_grid(ref_d_index)%n_tot_entries.eq.0)

          do i=1,size(run_grid(d_index)%bins)
            mBinID = run_grid(d_index)%bins(i)%bid
            ref_bin_index = DS_bin_index(
     &                        ref_grid(ref_d_index)%bins,mBinID,.True.) 
            if (ref_bin_index.eq.-1) then
              call DS_Logger('Bin with binID '//trim(DS_toStr(mBinID))//
     &              ' is missing in the reference grid. Adding it now.')
              call DS_reinitialize_bin(new_bin)
              new_bin%bid = mBinID
              call DS_add_bin_to_bins(ref_grid(ref_d_index)%bins,
     &                                                          new_bin)
              ref_bin_index = DS_bin_index(
     &                                ref_grid(ref_d_index)%bins,mBinID)
            endif
            
            run_bin = run_grid(d_index)%bins(i)
            if ((run_bin%n_entries.lt.ref_grid(ref_d_index)%
     &           min_bin_probing_points).and.empty_ref_grid) then
              write(*,*) "DiscreteSampler:: WARNING, the bin '"//
     &        trim(DS_toStr(run_bin%bid))//"' of dimension '"//
     &        trim(toStr(dim_name))//"' will be used for reference"//
     &        " even though it has been probed only "//
     &        trim(toStr(run_bin%n_entries))//" times (minimum "//
     &        "requested is "//trim(toStr(ref_grid(ref_d_index)%
     &        min_bin_probing_points))//" times)."
            endif

          ref_bin = ref_grid(ref_d_index)%bins(ref_bin_index)
          if (ref_grid(ref_d_index)%grid_mode.eq.2) then
!           This means that the reference grid is in 'initialization'
!           mode and should be overwritten by the running grid (instead
!           of being combined with it) when updated except for the
!           bins with not enough entries in the run_grid.
            if (run_bin%n_entries.ge.ref_grid(ref_d_index)%
     &                                    min_bin_probing_points) then
              call DS_reinitialize_bin(ref_bin)
            else
!             Then we combine the run_bin and the ref_bin by weighting
!             the ref_bin with the ratio of the corresponding norms
              ref_bin%weight = ref_bin%weight * (run_grid(
     &          d_index)%abs_norm / ref_grid(ref_d_index)%abs_norm)
              ref_bin%abs_weight = ref_bin%abs_weight * (run_grid(
     &          d_index)%abs_norm / ref_grid(ref_d_index)%abs_norm)
              ref_bin%weight_sqr = ref_bin%weight_sqr * (run_grid(
     &          d_index)%norm_sqr / ref_grid(ref_d_index)%norm_sqr)
            endif
          endif

          new_bin = ref_bin + run_bin

!         Now update the ref grid bin
          ref_grid(ref_d_index)%bins(ref_bin_index) = new_bin

          enddo
          call DS_synchronize_grid_with_bins(ref_grid(ref_d_index))

!         Now we set the global attribute of the reference_grid to be
!         the ones of the running grid.
          ref_grid(ref_d_index)%min_bin_probing_points =
     &       run_grid(d_index)%min_bin_probing_points
          ref_grid(ref_d_index)%grid_mode = run_grid(d_index)%grid_mode
          ref_grid(ref_d_index)%small_contrib_threshold = 
     &                        run_grid(d_index)%small_contrib_threshold
          ref_grid(ref_d_index)%damping_power = 
     &                                  run_grid(d_index)%damping_power

!         Now filter all bins in ref_grid that have 0.0 weight and
!         remove them! They will not be probed anyway.
          if (filterOutZeros) then
            shift = 0
            do j=1,size(ref_grid(ref_d_index)%bins)
              i = j - shift
              if ((ref_grid(ref_d_index)%bins(i)%weight.eq.0.0d0).and.
     &        (ref_grid(ref_d_index)%bins(i)%abs_weight.eq.0.0d0).and.
     &        (ref_grid(ref_d_index)%bins(i)%weight_sqr.eq.0.0d0)) then
                call DS_Logger('Bin with binID '//
     &            trim(DS_toStr(ref_grid(ref_d_index)%bins(i)%bid))//
     &            ' is zero and will be filtered out. Removing it now.')
                call DS_remove_bin_from_grid(ref_grid(ref_d_index),i)
                shift = shift + 1
              endif
            enddo
          endif

!         Clear the running grid now
          call DS_reinitialize_dimension(run_grid(d_index))

          deallocate(dim_name)

        end subroutine DS_update_grid_with_dim_index


        function DS_combine_two_bins(BinA, BinB) result(CombinedBin)
        implicit none
!         
!         Function arguments
!
          integer               :: d_index
          Type(Bin), intent(in) :: BinA, BinB
          Type(Bin)             :: CombinedBin
!         
!         Local variables
!
          call DS_reinitialize_bin(CombinedBin)
          if(.not.(BinA%bid==BinB%bid)) then
            write(*,*) 'DiscreteSampler:: Error in function '//
     &        'DS_combine_two_bins, cannot combine two bins '//
     &        ' with different bin IDs : '//trim(DS_toStr(BinA%bid))//
     &        ', '//trim(DS_toStr(BinB%bid))
            stop 1
          endif
          CombinedBin%bid = BinA%bid
          CombinedBin%n_entries = BinA%n_entries + BinB%n_entries
          if (CombinedBin%n_entries.eq.0) then
            CombinedBin%weight     = 0.0d0
            CombinedBin%abs_weight = 0.0d0
            CombinedBin%weight_sqr = 0.0d0
          else
            CombinedBin%weight     = (BinA%weight*BinA%n_entries + 
     &                 BinB%weight*BinB%n_entries)/CombinedBin%n_entries
            CombinedBin%abs_weight = (BinA%abs_weight*BinA%n_entries +
     &             BinB%abs_weight*BinB%n_entries)/CombinedBin%n_entries
            CombinedBin%weight_sqr = (BinA%weight_sqr*BinA%n_entries + 
     &             BinB%weight_sqr*BinB%n_entries)/CombinedBin%n_entries
          endif
        end function DS_combine_two_bins

!       ================================================
!       Main function to pick a point
!       ================================================
 
      subroutine DS_get_point_with_integerBinID(dim_name,
     &           random_variable, integerIDPicked, jacobian_weight,mode,
     &                                            convoluted_grid_names)
!
!       Subroutine arguments
!
        character(len=*), intent(in)             ::  dim_name
        real*8, intent(in)                       ::  random_variable
        integer, intent(out)                     ::  integerIDPicked
        real*8, intent(out)                      ::  jacobian_weight
        character(len=*), intent(in), optional   ::  mode
        character(len=*), dimension(:), intent(in), optional ::
     &                                             convoluted_grid_names
!
!       Local variables
!
        type(BinID)                             ::  mBinID
!
!       Begin code
!
        if (present(mode)) then
          if (present(convoluted_grid_names)) then
            call DS_get_point_with_BinID(dim_name,random_variable,
     &                      mBinID,jacobian_weight,mode=mode,
     &                      convoluted_grid_names=convoluted_grid_names)
          else
            call DS_get_point_with_BinID(dim_name,random_variable,
     &                                 mBinID,jacobian_weight,mode=mode)
          endif
        else
          if (present(convoluted_grid_names)) then
            call DS_get_point_with_BinID(dim_name,random_variable,
     &                      mBinID,jacobian_weight,
     &                      convoluted_grid_names=convoluted_grid_names)
          else
            call DS_get_point_with_BinID(dim_name,random_variable,
     &                                           mBinID,jacobian_weight)
          endif
        endif
        integerIDPicked = mBinID%id
      end subroutine DS_get_point_with_integerBinID

      subroutine DS_get_point_with_BinID(dim_name,
     &           random_variable, mBinID, jacobian_weight, mode,
     &                                            convoluted_grid_names)
!
!       Subroutine arguments
!
        character(len=*), intent(in)            ::  dim_name
        real*8, intent(in)                      ::  random_variable
        type(BinID), intent(out)                ::  mBinID
        real*8, intent(out)                     ::  jacobian_weight
        character(len=*), intent(in), optional  ::  mode
        character(len=*), dimension(:), intent(in), optional ::
     &                                             convoluted_grid_names
!
!       Local variables
!
!       chose_mode = 1 : Sampling accoridng to variance
!       chose_mode = 2 : Sampling according to norm
!       chose_mode = 3 : Uniform sampling
        integer                 :: chosen_mode
        type(SampledDimension)  :: mGrid, runGrid
        type(Bin)               :: mBin, mRunBin
        integer                 :: ref_grid_index, run_grid_index
        integer                 :: i,j
        real*8                  :: running_bound
        real*8                  :: normalized_bin_bound
        logical, dimension(:), allocatable   :: bin_indices_to_fill
        logical                 :: initialization_done
        real*8                  :: sampling_norm        
!       Local variables related to convolution
        real*8, dimension(:), allocatable :: convolution_factors
        integer                 :: conv_bin_index
        type(SampledDimension)  :: conv_dim
        logical                 :: one_norm_is_zero
        real*8                  :: small_contrib_thres
!
!       Begin code
!
        if (present(mode)) then
          if (mode.eq.'variance') then
            chosen_mode = 1
          elseif (mode.eq.'norm') then
            chosen_mode = 2
          elseif (mode.eq.'uniform') then
            chosen_mode = 3
          else
            write(*,*) "DiscreteSampler:: Error in subroutine"//
     &        " DS_get_point, mode '"//mode//"' is not recognized."
            stop 1
          endif
        else
          chosen_mode = 2
        endif  

        if (.not.allocated(ref_grid)) then
          write(*,*) "DiscreteSampler:: Error, dimensions"//
     &     " must first be registered with 'DS_register_dimension'"//
     &     " before the module can be used to pick a point."
          stop 1
        endif

        ref_grid_index = DS_dim_index(ref_grid, dim_name,.True.)
        if (ref_grid_index.eq.-1) then
          write(*,*) "DiscreteSampler:: Error in subroutine"//
     &     " DS_get_point, dimension '"//dim_name//"' not found."
          stop 1
        endif
        mGrid   = ref_grid(ref_grid_index)
        run_grid_index = DS_dim_index(run_grid, dim_name,.True.)
        if (run_grid_index.eq.-1) then
          write(*,*) "DiscreteSampler:: Error in subroutine"//
     &     " DS_get_point, dimension '"//dim_name//"' not found"//
     &     " in the running grid."
          stop 1
        endif
        runGrid = run_grid(run_grid_index)        

!       If the reference grid is empty, force the use of uniform
!       sampling
        if (mGrid%n_tot_entries.eq.0) then
          chosen_mode = 3
        endif

!       Pick the right norm for the chosen mode
        if (chosen_mode.eq.1) then
          sampling_norm           = mGrid%variance_norm
        elseif (chosen_mode.eq.2) then
          sampling_norm           = mGrid%abs_norm
        elseif (chosen_mode.eq.3) then
          sampling_norm           = float(size(mGrid%bins))
        endif

!       If the grid is empty we must first make sure that each bin was
!       probed with min_bin_probing_points before using a uniform grid
        allocate(bin_indices_to_fill(size(mGrid%bins)))
        initialization_done = .True.        
        if(mGrid%n_tot_entries.eq.0) then
          min_bin_index     = 1
          do i=1,size(mGrid%bins)
            mRunBin = DS_get_bin(runGrid%bins,mGrid%bins(i)%bid)
            if (mRunBin%n_entries.lt.mGrid%min_bin_probing_points) then
              bin_indices_to_fill(i) = .True.
              initialization_done    = .False.            
            else
              bin_indices_to_fill(i) = .False.
            endif
          enddo
          if(.not.initialization_done) then
!           In this case, we will only fill in bins which do not have 
!           have enough entries (and select them uniformly) and veto the 
!           others. The jacobian returned is still the one corresponding
!           to a uniform distributions over the whole set of bins.
!           Possible convolutions are ignored
            sampling_norm = 0.0d0
            do i=1,size(bin_indices_to_fill)
              if (bin_indices_to_fill(i)) then
                sampling_norm = sampling_norm + 1.0d0
              endif
            enddo
          endif
        endif

        if (initialization_done) then
          do i=1,size(mGrid%bins)
            bin_indices_to_fill(i) = .True.
          enddo
        endif

!       Pick the right reference bin value for the chosen mode. Note
!       that this reference value is stored in the %weight attribute
!       of the reference grid local copy mGrid
        do i=1,size(mGrid%bins)
          if (.not.bin_indices_to_fill(i)) then
            mGrid%bins(i)%weight    = 0.0d0
          elseif (chosen_mode.eq.1) then
            mGrid%bins(i)%weight    = DS_bin_variance(mGrid%bins(i))
          elseif (chosen_mode.eq.2) then
            mGrid%bins(i)%weight    = mGrid%bins(i)%abs_weight
          elseif (chosen_mode.eq.3) then
            mGrid%bins(i)%weight    = 1.0d0
          endif
        enddo

!
!       Treat specially contributions worth less than 5% of the
!       contribution averaged over all bins. For those, we sample
!       according to the square root (or the specified power 'pow'
!       of the reference value corresponding to the chosen mode. 
!       In this way, we are less sensitive to possible large fluctuations 
!       of very suppressed contributions for which the Jacobian would be 
!       really big. However, the square-root is such that a really
!       suppressed contribution at the level of numerical precision
!       would still never be probed.
!       
        average_contrib              = sampling_norm / size(mGrid%bins)
!       Ignore this if the average contribution is zero
        if (average_contrib.gt.0.0d0) then
          do i=1,size(mGrid%bins)
            mBin = mGrid%bins(i)    
            if ( (mBin%weight/average_contrib) .lt.
     &                               runGrid%small_contrib_threshold) then
              sampling_norm       = sampling_norm - mGrid%bins(i)%weight
              mGrid%bins(i)%weight = 
     &          ((mBin%weight/(runGrid%small_contrib_threshold
     &        *average_contrib))**runGrid%damping_power)*
     &        runGrid%small_contrib_threshold*average_contrib
              sampling_norm       = sampling_norm + mGrid%bins(i)%weight
            endif
          enddo
        endif
!
!       Now appropriately set the convolution factors
!
        allocate(convolution_factors(size(mGrid%bins)))
        if (present(convoluted_grid_names).and.initialization_done) then
!         Sanity check
          do j=1,size(convoluted_grid_names)
            if (DS_dim_index(run_grid,convoluted_grid_names(j),
     &                                               .True.).eq.-1) then
              write(*,*) "DiscreteSampler:: Error, dimension '"//
     &         convoluted_grid_names(j)//"' for convolut"//
     &         "ion could not be found in the running grid."
              stop 1
            endif
          enddo
          sampling_norm          = 0.0d0          
          do i=1,size(mGrid%bins)
            convolution_factors(i) = 1.0d0
            do j=1,size(convoluted_grid_names)
              conv_dim = DS_get_dimension(
     &                                run_grid,convoluted_grid_names(j))
              conv_bin_index = DS_bin_index(conv_dim%bins,
     &                                         mGrid%bins(i)%bid,.True.)
              if (conv_bin_index.eq.-1) then
                write(*,*) "DiscreteSampler:: Error, bin '"//
     &          trim(DS_toStr(mGrid%bins(i)%bid))//"' could not be fo"//
     &          "und in convoluted dimension '"//
     &                                    convoluted_grid_names(j)//"'."
                stop 1
              endif
              ! Notice that for the convolution we always use the
              ! absolute value of the weight because we assume the user
              ! has edited this grid by himself for with a single entry.
              convolution_factors(i) = convolution_factors(i)*
     &                          conv_dim%bins(conv_bin_index)%abs_weight
            enddo
            sampling_norm = sampling_norm + 
     &        convolution_factors(i)*mGrid%bins(i)%weight
          enddo
        else
          do i=1,size(mGrid%bins)
            convolution_factors(i)    = 1.0d0
          enddo
        endif

!       Now crash nicely on zero norm grid
        if (sampling_norm.eq.0d0.and..not.DS_tolerate_zero_norm) then
          one_norm_is_zero = .FALSE.
          write(*,*) 'DiscreteSampler:: Error, all bins'//
     &     " of sampled dimension '"//dim_name//"' or of the"//
     &     " following convoluted dimensions have zero weight:"
          if (chosen_mode.eq.2) then
            write(*,*) "DiscreteSampler:: Sampled dimension "//
     & "    : '"//trim(toStr(mGrid%dimension_name))//"' with norm "//
     &                     trim(toStr(mGrid%abs_norm,'ENw.3'))//"."
            one_norm_is_zero = (one_norm_is_zero.or.
     &                                          mGrid%abs_norm.eq.0.0d0)
          elseif (chosen_mode.eq.1) then
            write(*,*) "DiscreteSampler:: Sampled dimension "//
     & "    : '"//trim(toStr(mGrid%dimension_name))//"' with norm "//
     &                     trim(toStr(mGrid%variance_norm,'ENw.3'))//"."
            one_norm_is_zero = (one_norm_is_zero.or.
     &                                     mGrid%variance_norm.eq.0.0d0)
          elseif (chosen_mode.eq.3) then
            write(*,*) "DiscreteSampler:: Norm of sampled dimension '"//
     &       trim(toStr(mGrid%dimension_name))//"' irrelevant since"//
     &       " uniform sampling was selected."
          endif
          if(present(convoluted_grid_names).and.initialization_done)then
            do i=1,size(convoluted_grid_names)
              conv_dim = DS_get_dimension(run_grid,
     &                                         convoluted_grid_names(i))
              write(*,*) "DiscreteSampler:: Convoluted dimension "//
     &       trim(toStr(i))//": '"//convoluted_grid_names(i)//
     &       "' with norm "//trim(toStr(conv_dim%abs_norm,'ENw.3'))//"."
            one_norm_is_zero = (one_norm_is_zero.or.
     &                                       conv_dim%abs_norm.eq.0.0d0)
            enddo
          endif
          if(present(convoluted_grid_names).and.initialization_done
     &                                .and.(.not.one_norm_is_zero))then
            write(*,*) "DiscreteSampler:: None of the norm above"
     &        //" is zero, this means that the convolution (product)"
     &        //" of the grids yields zero for each bin, even though"
     &        //" they are not zero separately."
            write(*,*) "DiscreteSampler:: Use DS_print_global_info()"//
     &        " to investigate further."
          endif
          write(*,*) "DiscreteSampler:: One norm is zero, no sampling"//
     &     " can be done in these conditions. Set 'tolerate_zero_norm"//
     &     "' to .True. when initializating the module to proceed wi"//
     &     "th a uniform distribution for the grids of zero norm."
          stop 1
        endif

!       Or make it pure random if DS_tolerate_zero_norm is True.
        if (sampling_norm.eq.0d0) then
            do i=1,size(mGrid%bins)
               bin_indices_to_fill(i) = .True.
               if(chosen_mode.eq.2.and.mGrid%abs_norm.eq.0.0d0.or.
     &            chosen_mode.eq.1.and.mGrid%variance_norm.eq.0.0d0) then
                 mGrid%bins(i)%weight     = 1.0d0
               endif
               if (present(convoluted_grid_names).and.
     &             initialization_done.and.conv_dim%abs_norm.eq.0.0d0) then
                 conv_dim = DS_get_dimension(run_grid,
     &                                         convoluted_grid_names(i))
                 if (conv_dim%abs_norm.eq.0.0d0) then
                   convolution_factors(i) = 1.0d0
                 endif
               endif
               sampling_norm = sampling_norm + 
     &                       mGrid%bins(i)%weight*convolution_factors(i)
            enddo
!           If sampling_norm is again zero it means that the two grids
!           are "orthogonal" so that we have no choice but to randomize 
!           both.
            if (sampling_norm.eq.0.0d0) then
              do i=1,size(mGrid%bins)
                mGrid%bins(i)%weight = 1.0d0
                convolution_factors(i) = 1.0d0
                sampling_norm = sampling_norm + 1.0d0
              enddo
            endif
        endif

!
!       Now come the usual sampling method 
!
        running_bound = 0.0d0
        do i=1,size(mGrid%bins)
          if (.not.bin_indices_to_fill(i)) then
            cycle
          endif
          mBin = mGrid%bins(i)
          normalized_bin_bound = mBin%weight * 
     &                        ( convolution_factors(i) / sampling_norm )
          running_bound = running_bound + normalized_bin_bound
          if (random_variable.le.running_bound) then
            mBinID = mGrid%bins(i)%bid
            jacobian_weight = 1.0d0 / normalized_bin_bound
            deallocate(convolution_factors)
            deallocate(bin_indices_to_fill)
            return
          endif
        enddo
!       If no point was picked at this stage, there was a problem
        write(*,*) 'DiscreteSampler:: Error, no point could be '//
     &   'picked with random variable '//trim(toStr(random_variable))//
     &   ' using upper bound found of '//trim(toStr(running_bound))//'.'
        stop 1
      end subroutine DS_get_point_with_BinID

      function DS_bin_variance(mBin)
!
!       Function arguments
!
        type(Bin), intent(in)       :: mBin
        real*8                      :: DS_bin_variance
!
!       Begin code
!
        DS_bin_variance = ((mBin%weight_sqr - mBin%weight**2) *
     &                              (mBin%n_entries))/(mBin%n_entries+1)
      end function DS_bin_variance
!       ================================================
!       Grid I/O functions
!       ================================================

!       ---------------------------------------------------------------
!       This function writes the ref_grid to a file specified by its 
!       filename.
!       ---------------------------------------------------------------
        subroutine DS_write_grid_with_filename(filename, dim_name,
     &                                                        grid_type)
        implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)           :: filename
          character(len=*), intent(in), optional :: dim_name
          character(len=*), intent(in), optional :: grid_type          
!         
!         Local variables
!
          logical fileExist
!
!         Begin code
!
          inquire(file=filename, exist=fileExist)
          if (fileExist) then
            call DS_Logger('DiscreteSampler:: The file '
     &        //filename//' already exists, so beware that '//
     &                  ' the grid information will be appended to it.')
          endif
          open(123, file=filename, err=11, access='append',
     &                                                   action='write')
          goto 12
11        continue
          write(*,*) 'DiscreteSampler :: Error, file '//filename//
     &                               ' could not be opened for writing.'
          stop 1
12        continue
          if (present(dim_name)) then
            if (present(grid_type)) then
              call DS_write_grid_with_streamID(123, dim_name, grid_type)
            else
              call DS_write_grid_with_streamID(123, dim_name)
            endif
          else
            if (present(grid_type)) then              
              call DS_write_grid_with_streamID(123, grid_type=grid_type)
            else
              call DS_write_grid_with_streamID(123)
            endif                
          endif
          close(123)
        end subroutine DS_write_grid_with_filename

!       ---------------------------------------------------------------
!       This function writes the ref_grid or all grids to a file
!       specified by its stream ID.
!       ---------------------------------------------------------------
        subroutine DS_write_grid_with_streamID(streamID, dim_name, 
     &                                                       grid_type)
        implicit none
!         
!         Subroutine arguments
!
          integer, intent(in)                    :: streamID
          character(len=*), intent(in), optional :: dim_name
          character(len=*), intent(in), optional :: grid_type
!         
!         Local variables
!
          type(SampledDimension)                 :: grid
          integer                                :: i
          integer                                :: chosen_grid
!
!         Begin code
!
          if (present(grid_type)) then
            if (grid_type.eq.'ref') then
              chosen_grid = 1
            elseif (grid_type.eq.'run') then
              chosen_grid = 2
            elseif (grid_type.eq.'all') then
              chosen_grid = 3
            else
              write(*,*) 'DiscreteSampler:: Error in'//
     &          " subroutine 'DS_write_grid_with_streamID',"//
     &          " argument grid_type='"//grid_type//"' not"//
     &          " recognized."
              stop 1
            endif
          else
            chosen_grid = 1
          endif
          if ((chosen_grid.eq.1.or.chosen_grid.eq.3)
     &                        .and..not.allocated(ref_grid)) then
            return
          endif
          if ((chosen_grid.eq.2..or.chosen_grid.eq.3)
     &                        .and..not.allocated(run_grid)) then
            return
          endif
          if (present(dim_name)) then
            if (chosen_grid.eq.1.or.chosen_grid.eq.3) then            
              grid = ref_grid(DS_dim_index(ref_grid, dim_name))
              call DS_write_grid_from_grid(grid, streamID,'ref')
            endif
            if (chosen_grid.eq.2.or.chosen_grid.eq.3) then
              grid = run_grid(DS_dim_index(run_grid, dim_name))
              call DS_write_grid_from_grid(grid, streamID,'run')
            endif
          else
            if (chosen_grid.eq.1.or.chosen_grid.eq.3) then
              do i=1,size(ref_grid)
                grid = ref_grid(i)
                call DS_write_grid_from_grid(grid, streamID,'ref')
              enddo
            endif
            if (chosen_grid.eq.2.or.chosen_grid.eq.3) then
              do i=1,size(run_grid)
                grid = run_grid(i)
                call DS_write_grid_from_grid(grid, streamID,'run')
              enddo
            endif
          endif
        end subroutine DS_write_grid_with_streamID

!       ---------------------------------------------------------------
!       This function writes a given grid to a file.
!       ---------------------------------------------------------------
        subroutine DS_write_grid_from_grid(grid, streamID, grid_type)
        implicit none
!         
!         Subroutine arguments
!
          integer, intent(in)                    :: streamID
          type(SampledDimension), intent(in)     :: grid
          character(len=*), intent(in)           :: grid_type          
!         
!         Local variables
!
          integer                                :: i
!
!         Begin code
!

          write(streamID,*) ' <DiscreteSampler_grid>'
          write(streamID,*) ' '//trim(toStr(grid%dimension_name))
          if (grid_type.eq.'ref') then
            write(streamID,*) ' '//trim(toStr(1))
     &      //" # 1 for a reference and 2 for a running grid."
          elseif (grid_type.eq.'run') then
            write(streamID,*) ' '//trim(toStr(2))
     &      //" # 1 for a reference and 2 for a running grid."
          else
            write(*,*) "DiscreteSampler:: Error, grid_type'"//
     &       grid_type//"' not recognized."
            stop 1
          endif  
          write(streamID,*) ' '//trim(toStr(grid%min_bin_probing_points
     &      ))//" # Attribute 'min_bin_probing_points' of the grid."
          write(streamID,*) ' '//trim(toStr(grid%grid_mode
     &      ))//" # Attribute 'grid_mode' of the grid. 1=='default',"
     &      //"2=='initialization'"
          write(streamID,*) ' '//trim(toStr(grid%small_contrib_threshold
     &      ))//" # Attribute 'small_contrib_threshold' of the grid."
          write(streamID,*) ' '//trim(toStr(grid%damping_power
     &      ))//" # Attribute 'damping_power' of the grid."
          write(streamID,*) '# binID   n_entries weight   weight_sqr'//
     &      '   abs_weight'
          do i=1,size(grid%bins)
            write(streamID,*) 
     &          '   '//trim(DS_toStr(grid%bins(i)%bid))//
     &          '   '//trim(toStr(grid%bins(i)%n_entries))//
     &          '   '//trim(toStr(grid%bins(i)%weight,'ESw.15E3'))//
     &          '   '//trim(toStr(grid%bins(i)%weight_sqr,'ESw.15E3'))//
     &          '   '//trim(toStr(grid%bins(i)%abs_weight,'ESw.15E3'))
          enddo
          write(streamID,*) ' </DiscreteSampler_grid>'

        end subroutine DS_write_grid_from_grid

!       ---------------------------------------------------------------
!       This function loads the grid specified in a file specified by its
!       stream ID into the run_grid.
!       ---------------------------------------------------------------
        subroutine DS_load_grid_with_filename(filename, dim_name)
        implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)           :: filename
          character(len=*), intent(in), optional :: dim_name
!         
!         Local variables
!
          logical fileExist        
!
!         Begin code
!
!         Make sure the module is initialized
          if (.not.allocated(DS_isInitialized)) then
              call DS_initialize()
          endif
          inquire(file=filename, exist=fileExist)
          if (.not.fileExist) then
            write(*,*) 'DiscreteSampler:: Error, the file '//filename//
     &                                           ' could not be found.'
            stop 1
          endif
          open(124, file=filename, err=13, action='read')
          goto 14
13        continue
          write(*,*) 'DiscreteSampler :: Error, file '//filename//
     &                                ' exists but could not be read.'
14        continue
          if (present(dim_name)) then
            call DS_load_grid_with_streamID(124, dim_name)
          else
            call DS_load_grid_with_streamID(124)
          endif
          close(124)
        end subroutine DS_load_grid_with_filename

!       ---------------------------------------------------------------
!       This function loads the grid specified in a file specified by its 
!       stream ID into the run_grid.
!       ---------------------------------------------------------------
        subroutine DS_load_grid_with_streamID(streamID, dim_name)
        implicit none
!         
!         Subroutine arguments
!
          integer, intent(in)                    :: streamID
          character(len=*), intent(in), optional :: dim_name
!         
!         Local variables
!
          integer                                :: i
          character(512)                         :: buff
          character(2)                           :: TwoBuff
          character(3)                           :: ThreeBuff
          logical                                :: startedGrid
          real*8                       :: weight, abs_weight, weight_sqr
          integer                      :: n_entries, bid
          type(Bin)                    :: new_bin
          integer                      :: char_size
          integer                      :: read_position
          integer                      :: run_dim_index
          integer                      :: grid_mode
          real*8                       :: small_contrib_threshold
          real*8                       :: damping_power
!
!         Begin code
!
!         Make sure the module is initialized
          if (.not.allocated(DS_isInitialized)) then
              call DS_initialize()
          endif
!         Now start reading the file
          startedGrid   = .False.
          read_position = 0
          do
998         continue          
            read(streamID, "(A)", size=char_size, eor=998,
     &                                    end=999, advance='no') TwoBuff

      
            if (char_size.le.1) then
              cycle
            endif
            if (TwoBuff(1:1).eq.'#'.or.TwoBuff(2:2).eq.'#') then
!             Advance the stream
              read(streamID,*,end=990) buff
              cycle
            endif
            if (startedGrid) then
              read(streamID, "(A)", size=char_size,
     &                                    end=999, advance='no') TwoBuff
              if (TwoBuff(1:2).eq.'</') then
!             Advance the stream
                read(streamID,*,end=990) buff
                startedGrid   = .False.
                read_position = 0
                cycle
              endif
              read(streamID,*,end=990) bid, n_entries, weight, 
     &                                            weight_sqr, abs_weight
              new_bin%bid           = bid
              new_bin%n_entries     = n_entries
              new_bin%weight        = weight
              new_bin%weight_sqr    = weight_sqr
              new_bin%abs_weight    = abs_weight
              call DS_add_bin_to_bins(run_grid(size(run_grid))%bins,
     &                                                          new_bin)
            else
!             Advance the stream
              if (read_position.eq.0) then
                read(streamID,*,end=990) buff
                if (buff(1:22).eq.'<DiscreteSampler_grid>') then
                  read_position = read_position + 1
                endif
              else
                select case(read_position)
                  case(1)
                    read(streamID,*,end=990) buff
                    run_dim_index = DS_dim_index(run_grid,
     &                                               trim(buff),.True.)
                    if (run_dim_index.ne.-1) then
                      call DS_remove_dimension_from_grid(run_grid, 
     &                                                    run_dim_index)
                    endif
                    call DS_register_dimension(trim(buff),0,.False.)
                  case(2)
                    read(streamID,*,end=990) grid_mode
                    if (grid_mode.ne.1) then
                      write(*,*) 'DiscreteSampler:: Warning, the '//
     &                  "grid read is not of type 'reference'."//
     &                  "  It will be skipped."
                      call DS_remove_dimension_from_grid(run_grid, 
     &                                                    run_dim_index)
                      read_position = 0
                      startedGrid = .False.
                      goto 998
                    endif
                  case(3)
                    read(streamID,*,end=990) 
     &                run_grid(size(run_grid))%min_bin_probing_points
                  case(4)
                    read(streamID,*,end=990) 
     &                run_grid(size(run_grid))%grid_mode
                  case(5)
                    read(streamID,*,end=990) small_contrib_threshold
                    if (small_contrib_threshold.lt.0.0d0.or.
     &                              small_contrib_threshold.gt.0.5d0) then
                      write(*,*) 'DiscreteSampler:: The '//
     &                  'small_contrib_threshold must be >= 0.0 and '//
     &                  '< 0.5 to be meaningful.'
                      stop 1
                    endif
                    run_grid(size(run_grid))%small_contrib_threshold
     &                                         = small_contrib_threshold
                  case(6)
                    read(streamID,*,end=990) damping_power
                    if (damping_power.lt.0.0d0.or.
     &                                         damping_power.gt.1.0d0) then
                      write(*,*) 'DiscreteSampler:: The damping power'//
     &                  ' must be >= 0.0 and <= 1.0.'
                      stop 1
                    endif
                    run_grid(size(run_grid))%damping_power
     &                                                   = damping_power
!                   Make sure that the last info read before reading the
!                   bin content (here the info with read_position=6)
!                   sets startedGrid to .True. to start the bin readout 
                    startedGrid   = .True.
                  case default
                    write(*,*) 'DiscreteSampler:: Number of entries'//
     &                ' before reaching bin lists exceeded.'
                    goto 990 
                end select
                read_position = read_position + 1
              endif
            endif
          enddo
          goto 999
990       continue
          write(*,*) 'DiscreteSampler:: Error, when loading grids'//
     &      ' from file.'
          stop 1
999       continue

!         Now update the running grid into the reference one
          call DS_update_grid()
        end subroutine DS_load_grid_with_streamID


!       ---------------------------------------------------------------
!       Synchronizes the cumulative information in a given grid from
!       its bins.
!       --------------------------------------------------------------- 
        subroutine DS_synchronize_grid_with_bins(grid)
        implicit none
!
!         Subroutine argument
!
          type(sampledDimension), intent(inout) :: grid
!         
!         Local variables
!
          real*8           :: norm, abs_norm, norm_sqr, variance_norm
          integer          :: i, n_tot_entries
!
!         Begin Code
!
          norm              = 0.0d0
          abs_norm          = 0.0d0
          norm_sqr          = 0.0d0
          variance_norm     = 0.0d0
          n_tot_entries     = 0
          do i=1,size(grid%bins)
            n_tot_entries   = n_tot_entries  + grid%bins(i)%n_entries
            norm_sqr        = norm_sqr       + grid%bins(i)%weight_sqr
            abs_norm        = abs_norm       + grid%bins(i)%abs_weight
            norm            = norm           + grid%bins(i)%weight
            variance_norm   = variance_norm  + 
     &                                     DS_bin_variance(grid%bins(i))
          enddo
          grid%n_tot_entries = n_tot_entries
          grid%norm_sqr      = norm_sqr
          grid%abs_norm      = abs_norm
          grid%norm          = norm
          grid%variance_norm = variance_norm
        end subroutine DS_synchronize_grid_with_bins

!       ================================================
!       Functions and subroutine handling derived types
!       ================================================

!       ---------------------------------------------------------------
!       Specify how bin idea should be compared
!       ---------------------------------------------------------------
        function equal_binID(binID1,binID2)
        implicit none
!         
!         Function arguments
!
          type(binID), intent(in)  :: binID1, binID2
          logical                  :: equal_binID
!
!         Begin code
!
          if(binID1%id.ne.binID2%id) then
            equal_binID = .False.
            return
          endif
          equal_binID = .True.
          return
        end function equal_binID

!       ---------------------------------------------------------------
!       BinIDs constructors
!       ---------------------------------------------------------------
        pure elemental subroutine binID_from_binID(binID1,binID2)
        implicit none
!         
!         Function arguments
!
          type(binID), intent(out)  :: binID1
          type(binID), intent(in)  :: binID2
!
!         Begin code
!
          binID1%id = binID2%id
        end subroutine binID_from_binID

        pure elemental subroutine binID_from_integer(binID1,binIDInt)
        implicit none
!         
!         Function arguments
!
          type(binID), intent(out)  :: binID1
          integer,     intent(in)   :: binIDInt
!
!         Begin code
!
          binID1%id = binIDInt
        end subroutine binID_from_integer

!       Provide a constructor-like way of creating a binID
        function DS_binID(binIDInt)
        implicit none
!         
!         Function arguments
!
          type(binID)              :: DS_binID
          integer,     intent(in)  :: binIDInt
!
!         Begin code
!
          DS_binID = binIDInt
        end function DS_binID
!       ---------------------------------------------------------------
!       String representation of a binID
!       ---------------------------------------------------------------
        function DS_toStr(mBinID)
        implicit none
!         
!         Function arguments
!
          type(binID), intent(in)  :: mBinID
          character(100)           :: DS_toStr
!
!         Begin code
!
          DS_toStr = trim(toStr(mBinID%id))
        end function DS_toStr


!       ================================================
!        Access routines emulating a dictionary
!       ================================================

!       ---------------------------------------------------------------
!       Returns the index of the discrete dimension with name dim_name
!       ---------------------------------------------------------------
        function DS_dim_index_default(grid, dim_name)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in), allocatable
     &                                  :: grid
          character(len=*), intent(in)  :: dim_name
          integer                       :: DS_dim_index_default
!
!         Begin code
!  
          DS_dim_index_default =
     &               DS_dim_index_with_force(grid, dim_name, .False.)
        end function DS_dim_index_default

        function DS_dim_index_with_force(grid, dim_name, force)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in), allocatable
     &                                  :: grid
          character(len=*), intent(in)  :: dim_name
          integer                       :: DS_dim_index_with_force
          logical                       :: force
!
!         Local variables
!

          integer i,j
!
!         Begin code
!
          DS_dim_index_with_force = -1
          if (.not.allocated(grid)) then
            return
          endif
          do i = 1, size(grid)
            if (len(dim_name).ne.size(grid(i)%dimension_name)) cycle
            do j =1, len(dim_name)
              if(grid(i)%dimension_name(j).ne.dim_name(j:j)) then
                goto 1
              endif
            enddo
            DS_dim_index_with_force = i
            return
1           continue
          enddo
          if (DS_dim_index_with_force.eq.-1.and.(.not.force)) then
            write(*,*) 'DiscreteSampler:: Error in function dim_index'//
     &        "(), dimension name '"//dim_name//"' not found."
            stop 1
          endif
        end function DS_dim_index_with_force

        function DS_dim_index_default_with_chararray(grid, dim_name)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in), allocatable 
     &                                         :: grid
          character, dimension(:), intent(in)  :: dim_name
          integer                 :: DS_dim_index_default_with_chararray
!
!         Begin code
!  
          DS_dim_index_default_with_chararray = 
     &                DS_dim_index_with_force_with_chararray(
     &                                          grid, dim_name, .False.)
        end function DS_dim_index_default_with_chararray

        function DS_dim_index_with_force_with_chararray(
     &                                            grid, dim_name, force)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in), allocatable
     &                                        :: grid
          character, dimension(:), intent(in) :: dim_name
          integer              :: DS_dim_index_with_force_with_chararray
          logical                             :: force
!
!         Local variables
!

          integer i,j
!
!         Begin code
!
          DS_dim_index_with_force_with_chararray = -1
          if (.not.allocated(grid)) then
            return
          endif
          do i = 1, size(grid)
            if (size(dim_name).ne.size(grid(i)%dimension_name)) cycle
            do j =1, size(dim_name)
              if(grid(i)%dimension_name(j).ne.dim_name(j)) then
                goto 1
              endif
            enddo
            DS_dim_index_with_force_with_chararray = i
            return
1           continue
          enddo
          if (DS_dim_index_with_force_with_chararray.eq.-1.and.
     &                                                (.not.force)) then
            write(*,*) 'DiscreteSampler:: Error in function dim_index'//
     &        "(), dimension name '"//dim_name//"' not found."
            stop 1
          endif
        end function DS_dim_index_with_force_with_chararray

!       End module
        end module DiscreteSampler
