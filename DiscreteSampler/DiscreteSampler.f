!
!     Module      : DiscreteSampler
!     Author      : Valentin Hirschi
!     Date        : 29.10.2014
!     Destriction : 
!              A relatively simple and flexible module to do
!              sampling of discrete dimensions for Monte-Carlo
!              purposes.
!
!     List of public subroutines and usage :
!
!
!     DS_register_dimension(name, n_bins)
!       ::  Register a new dimension with its name and number of bins
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
!     DS_add_entry(dimension_name, (binID|integerID), weight)
!       ::  Add a new weight to a certan bin (characterized by either
!       ::  its binID or the integer of the binID)
!
!       --- DONE UP TO HERE ---
! 
!     DS_get_point(dim_name, random_variable, 
!                       (binIDPicked|integerIDPicked), jacobian_weight)
!       :: From a given random variable in [0.0,1.0] and a dimension
!       :: name, this subroutine returns the picked bin or index and
!       :: the Jacobian weight associated.
!       :: Jacobian =
!       ::  n_total_bins*(wgt_normalise_bin_selectione) === absolute_wgt_bin_selected / average_wgt
!
!     DS_update_grid((dim_name|void))
!       ::  Update the reference grid of the dimension dim_name or 
!       ::  update all of them at the same time without argument.
!
!     DS_load_grid((file_name|stream_id), (dim_name|void))
!       :: Reset the running_grid for dimension dim_name and loads in
!       :: the data obtained from file 'file_name' for this dimension
!       :: or all
!
!     DS_write_grid((file_name|stream_id), (dim_name|void))
!       :: Append to file 'file_name' or a given stream the data for 
!       :: the current reference grid for dimension dim_name or
!       :: or all

      module DiscreteSampler

      use StringCast

      logical    DS_verbose
      parameter (DS_verbose=.False.)

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
        integer n_entries
!       Practical to be able to identify a bin by its id
        type(binID) bid
      endtype

      type sampledDimension
!       These are the reference weights, for the grid currently used
!       and controlling the sampling
        type(bin) , dimension(:), allocatable    :: bins
!       Keep track of the norm (i.e. sum of all weights) and the total
!       number of points for ease and optimisation purpose
        real*8                                   :: norm
        integer                                  :: n_tot_entries
!       A handy way of referring to the dimension by its name rather than
!       an index.
        character, dimension(:), allocatable     :: dimension_name
      endtype sampledDimension

!     This stores the overall discrete reference grid
      type(sampledDimension), dimension(:), allocatable :: ref_grid

!       This is the running grid, whose weights are being updated for each point
!       but not yet used for the sampling. The user must call the 'update'
!       function for the running grid to be merged to the reference one.
      type(sampledDimension), dimension(:), allocatable :: running_grid

      interface DS_add_entry
        module procedure DS_add_entry_with_BinID
        module procedure DS_add_entry_with_BinIntID
      end interface DS_add_entry

      interface DS_print_global_info
        module procedure DS_print_dim_global_info_from_name
        module procedure DS_print_dim_global_info_from_index
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

      contains

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
          deallocate(ref_grid)
          deallocate(running_grid)
        end subroutine DS_clear

!       ---------------------------------------------------------------
!       This subroutine takes care of registering a new dimension in
!       the DSampler module by characterizin it by its name and number
!       of bins.
!       ---------------------------------------------------------------
        subroutine DS_register_dimension(dim_name,n_bins)
        implicit none
!         
!         Subroutine arguments
!        
          integer , intent(in)                :: n_bins
          character(len=*), intent(in)        :: dim_name
!
!         Begin code
!
          call DS_add_dimension_to_grid(ref_grid, dim_name, n_bins)
          call DS_add_dimension_to_grid(running_grid, dim_name, 
     &                                                          n_bins)

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
          type(sampledDimension), dimension(:), allocatable :: tmp
          integer i
!
!         Begin code
!
!         Either allocate the discrete grids or append a dimension 
          if (allocated(grid)) then
            allocate(tmp(size(grid)+1))
            tmp(1:size(grid)) = grid
            deallocate(grid)
            allocate(grid(size(tmp)))
            grid = tmp
            deallocate(tmp)
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
          integer dim_index
!
!         Begin code
!
          dim_index = DS_dim_index(ref_grid, dim_name)
          call DS_remove_dimension_from_grid(ref_grid, dim_index)
          call DS_remove_dimension_from_grid(running_grid, dim_index)
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
          if (dim_index.eq.1) then
            tmp = grid(2:size(grid))
          elseif (dim_index.eq.size(grid)) then
            tmp = grid(1:size(grid)-1)
          else
            tmp(1:dim_index-1)       = grid(1:dim_index-1)
            tmp(dim_index:size(tmp)) = grid(dim_index+1:size(grid))
          endif
          deallocate(grid)
          allocate(grid(size(tmp)))
          grid = tmp
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
          d_dim%norm      = 0.0d0
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
          do i=1, size(d_dim%bins)
            call DS_initialize_bin(d_dim%bins(i))
          enddo
          do i= 1, len(d_dim%dimension_name)
            d_dim%dimension_name(i:i) = ' '
          enddo
          d_dim%norm      = 0.0d0
          d_dim%n_tot_entries   = 0

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
          d_bin%weight    = 0.0d0
          d_bin%n_entries = 0
          d_bin%bid       = 0
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
          d_bin%weight    = 0.0d0
          d_bin%n_entries = 0
        end subroutine DS_reinitialize_bin

!       ---------------------------------------------------------------
!       Returns the index of the discrete dimension with name dim_name
!       ---------------------------------------------------------------
        function DS_dim_index(grid, dim_name)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in)
     &                                  :: grid
          character(len=*), intent(in)  :: dim_name
          integer                       :: DS_dim_index
!
!         Local variables
!

          integer i,j
!
!         Begin code
!
          DS_dim_index = -1
          do i = 1, size(grid)
            if (len(dim_name).ne.size(grid(i)%dimension_name)) cycle
            do j =1, len(dim_name)
              if(grid(i)%dimension_name(j).ne.dim_name(j:j)) then
                goto 1
              endif
            enddo
            DS_dim_index = i
            return
1           continue
          enddo
          if (DS_dim_index.eq.-1) then
            write(*,*) 'DiscreteSampler:: Error in function dim_index'//
     &        "(), dimension name '"//dim_name//"' not found."
            stop 1
          endif
        end function DS_dim_index

        function DS_get_dimension(grid, dim_name)
        implicit none
!         
!         Function arguments
!
          type(sampledDimension), dimension(:), intent(in)
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
        function DS_bin_index(bins, mBinID)
        implicit none
!         
!         Function arguments
!
          type(Bin), dimension(:), intent(in)  
     &                                  :: bins
          type(BinID)                   :: mBinID
          integer                       :: DS_bin_index
!
!         Local variables
!
          integer i
!
!         Begin code
!
!         For efficiency first look at index mBinID%id
          if (bins(mBinID%id)%bid==mBinID) then
              DS_bin_index = mBinID%id
              return
          endif
          
          DS_bin_index = -1
          do i = 1, size(bins)
            if (bins(i)%bid==mBinID) then
              DS_bin_index = i
              return              
            endif
          enddo
          if (DS_bin_index.eq.-1) then
            write(*,*) 'DiscreteSampler:: Error in function bin_index'//
     &        "(), bin with BinID '"//trim(DS_toStr(mBinID))
     &        //"' not found."
            stop 1
          endif
        end function DS_bin_index

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
          type(sampledDimension), dimension(:), intent(in)
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
        subroutine DS_add_entry_with_BinID(dim_name, mBinID,weight)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)  :: dim_name
          type(BinID)                   :: mBinID
          real*8                        :: weight
!
!         Local variables
!
          integer dim_index, bin_index
!
!         Begin code
!
          dim_index = DS_dim_index(running_grid, dim_name)
          bin_index = DS_bin_index(running_grid(dim_index)%bins,mBinID)
          running_grid(dim_index)%norm = 
     &                   running_grid(dim_index)%norm + weight
          running_grid(dim_index)%n_tot_entries =  
     &                   running_grid(dim_index)%n_tot_entries + 1
          running_grid(dim_index)%bins(bin_index)%weight = 
     &           running_grid(dim_index)%bins(bin_index)%weight + weight
          running_grid(dim_index)%bins(bin_index)%n_entries =
     &           running_grid(dim_index)%bins(bin_index)%n_entries + 1
        end subroutine DS_add_entry_with_BinID

        subroutine DS_add_entry_with_BinIntID(dim_name, BinIntID,
     &                                                       weight)
          implicit none
!         
!         Subroutine arguments
!
          character(len=*), intent(in)  :: dim_name
          integer                       :: BinIntID
          real*8                        :: weight 
!
!         Begin code
!
          call DS_add_entry_with_BinID(dim_name, DS_BinID(BinIntID),
     &                                                          weight)
        end subroutine DS_add_entry_with_BinIntID

!       ---------------------------------------------------------------
!       Prints out all informations for dimension of index d_index, or
!       name d_name.
!       ---------------------------------------------------------------
        subroutine DS_print_dim_global_info_from_void()
          integer i
          do i = 1, size(ref_grid)
            call DS_print_dim_global_info_from_index(i)
          enddo
        end subroutine DS_print_dim_global_info_from_void

        subroutine DS_print_dim_global_info_from_name(d_name)
          character(len=*), intent(in) :: d_name
!         We assume the index to be the same for all grids, which very
!         reasonable, so we pick ref_grid to get the index here
          call DS_print_dim_global_info_from_index(
     &                                  DS_dim_index(ref_grid,d_name))
        end subroutine DS_print_dim_global_info_from_name

        subroutine DS_print_dim_global_info_from_index(d_index)
        implicit none
!         
!         Function arguments
!
          integer, intent(in) :: d_index
!
!         Local variables
!
          integer n_bins
!
!         Begin code
!
          n_bins = size(ref_grid(d_index)%bins) 
          write(*,*) "DiscreteSampler:: Information for dimension '"//
     & trim(toStr_char_array(ref_grid(d_index)%dimension_name))//"' ("//
     &                                    trim(toStr(n_bins))//" bins):"

          write(*,*) "DiscreteSampler:: || Reference grid "
          call DS_print_dim_info(ref_grid(d_index))
          write(*,*) "DiscreteSampler:: || Running grid "
          call DS_print_dim_info(running_grid(d_index))

        end subroutine DS_print_dim_global_info_from_index

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
          integer i,j, curr_pos1, curr_pos2
          integer n_bins, bin_width
!         Adding the minimum size for the separators '|' and binID assumed
!         of being of length 2 at most, so 10*2+11 and + 20 security :)

          character(samplingBarWidth+10*2+11+20)       :: samplingBar1
          character(samplingBarWidth+10*2+11+20)       :: samplingBar2
!
!         Begin code
!
!
!         Setup the sampling bars
!
          if (d_dim%norm.eq.0.0d0) then
            samplingBar1 = "| Empty grid |"
            samplingBar2 = "| Empty grid |"
          else
            do i=1,len(samplingBar1)
              samplingBar1(i:i)=' '
              samplingBar2(i:i)=' '
            enddo
            samplingBar1(1:1) = '|'
            samplingBar2(1:1) = '|' 
            curr_pos1 = 2
            curr_pos2 = 2
            do i=1,min(10,size(d_dim%bins)) 
              samplingBar1(curr_pos1:curr_pos1+1) =
     &                             trim(DS_toStr(d_dim%bins(i)%bid))
              samplingBar2(curr_pos2:curr_pos2+1) = 
     &                             trim(DS_toStr(d_dim%bins(i)%bid))
              curr_pos1 = curr_pos1+2
              curr_pos2 = curr_pos2+2

              bin_width = int((d_dim%bins(i)%weight/d_dim%norm)*
     &                                             samplingBarWidth)
              do j=1,bin_width
                samplingBar1(curr_pos1+j:curr_pos1+j) = ' '
              enddo
              curr_pos1 = curr_pos1+bin_width+1
              samplingBar1(curr_pos1:curr_pos1) = '|'
              curr_pos1 = curr_pos1+1

              bin_width = int((float(d_dim%bins(i)%n_entries)/
     &                            d_dim%n_tot_entries)*samplingBarWidth)
              do j=1,bin_width
                samplingBar2(curr_pos2+j:curr_pos2+j) = ' '
              enddo
              curr_pos2 = curr_pos2+bin_width+1
              samplingBar2(curr_pos2:curr_pos2) = '|'
              curr_pos2 = curr_pos2+1
            enddo
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

          write(*,*) "DiscreteSampler::   -> norm : "//
     &                           trim(toStr(d_dim%norm,'Ew.2'))
          if (n_bins.gt.10) then
            write(*,*) "DiscreteSampler::   -> Sampled as"//
     &                                      " (first 10 bins):"
          else
            write(*,*) "DiscreteSampler::   -> Sampled as:"
          endif
          write(*,*) "DiscreteSampler::    "//trim(samplingBar1)

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
          integer                  :: dim_index
!
!         Begin code
!
          dim_index = DS_dim_index(ref_grid, dim_name)
          call DS_add_bin_with_binID( dim_name,DS_binID(
     &                    size(ref_grid(dim_index)%bins)+1 ))
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
          integer dim_index
          type(Bin)                :: new_bin
!
!         Begin code
!
          call DS_reinitialize_bin(new_bin)
          new_bin%bid = mBinID
          dim_index = DS_dim_index(ref_grid, dim_name)
          call DS_add_bin_to_bins(ref_grid(dim_index)%bins,new_bin)
          call DS_add_bin_to_bins(running_grid(dim_index)%bins,new_bin)
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
!
!         Begin code
!
          allocate(tmp(size(bins)+1))
          tmp(1:size(bins)) = bins
          tmp(size(bins)+1) = new_bin
          deallocate(bins)
          allocate(bins(size(tmp)))
          bins = tmp
        end subroutine DS_add_bin_to_bins

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
!         Local variables
!
          integer                        :: dim_index
!
!         Begin code
!
          dim_index = DS_dim_index(ref_grid, dim_name)
          call DS_remove_bin_from_bins(ref_grid(dim_index)%bins,
     &                                                        binIndex)
          call DS_remove_bin_from_bins(running_grid(dim_index)%bins,
     &                                                        binIndex)
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
          integer                        :: dim_index
          integer                        :: bin_index          
!
!         Begin code
!
          dim_index = DS_dim_index(ref_grid, dim_name)
          bin_index = DS_bin_index(ref_grid(dim_index)%bins,mbinID)
          call DS_remove_bin_withIndex(dim_name, bin_index)
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

        subroutine DS_remove_bin_from_bins(bins,bin_index)
          implicit none
!         
!         Subroutine arguments
!
          type(Bin), dimension(:), allocatable, intent(inout)  
     &                             :: bins
          integer, intent(in)      :: bin_index
!
!         Local variables
!
          type(Bin), dimension(:), allocatable :: tmp
!
!         Begin code
!
          allocate(tmp(size(bins)-1))
          if (bin_index.eq.1) then
            tmp = bins(2:size(bins))
          elseif (bin_index.eq.size(bins)) then
            tmp = bins(1:size(bins)-1)
          else
            tmp(1:bin_index-1)       = bins(1:bin_index-1)
            tmp(bin_index:size(tmp)) = bins(bin_index+1:size(bins))
          endif
          deallocate(bins)
          allocate(bins(size(tmp)))
          bins = tmp
        end subroutine DS_remove_bin_from_bins


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

!       End module
        end module DiscreteSampler

