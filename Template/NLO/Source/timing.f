      subroutine timing_reset()
      implicit none
      include 'timing_variables.inc'
      integer(kind=8) clock_reads(timing_ncategories)
      common/timing_read_counts/clock_reads
      double precision elapsed(timing_ncategories),last_clock
      integer stack(0:timing_stack_size),depth,sample_position
      integer(kind=8) points,samples,sampler_state
      common/timing_clocks/elapsed,last_clock
      common/timing_stack/stack,depth,sample_position
      common/timing_counts/points,samples,sampler_state
      clock_reads=0_8
      elapsed=0d0
      last_clock=0d0
      stack=timing_weights
      depth=0
      points=0_8
      samples=0_8
      sampler_state=104729_8
      sample_position=-1
      timing_sampled=.false.
      end

      subroutine timing_begin_point()
c One independently chosen position per block of 100 MINT points avoids
c aliasing with repeated sector/history loops. This private Park-Miller
c sequence never consumes a physics random number. Its product fits int64.
      implicit none
      include 'timing_variables.inc'
      double precision elapsed(timing_ncategories),last_clock
      integer stack(0:timing_stack_size),depth,sample_position,position
      integer(kind=8) points,samples,sampler_state
      common/timing_clocks/elapsed,last_clock
      common/timing_stack/stack,depth,sample_position
      common/timing_counts/points,samples,sampler_state
      if(timing_sampled)error stop 'Unclosed timing point'
      position=int(mod(points,100_8))
      if(position.eq.0)then
         sampler_state=mod(48271_8*sampler_state,2147483647_8)
         sample_position=int(mod(sampler_state,100_8))
      endif
      points=points+1_8
      timing_sampled=position.eq.sample_position
      if(.not.timing_sampled)return
      samples=samples+1_8
      depth=0
      stack(0)=timing_weights
      call cpu_time(last_clock)
      end

      subroutine timing_end_point()
      implicit none
      include 'timing_variables.inc'
      integer(kind=8) clock_reads(timing_ncategories)
      common/timing_read_counts/clock_reads
      double precision elapsed(timing_ncategories),last_clock,now
      integer stack(0:timing_stack_size),depth,sample_position
      common/timing_clocks/elapsed,last_clock
      common/timing_stack/stack,depth,sample_position
      if(.not.timing_sampled)return
      if(depth.ne.0)error stop 'Unclosed timing category'
      clock_reads(timing_weights)=clock_reads(timing_weights)+1_8
      call cpu_time(now)
      elapsed(timing_weights)=elapsed(timing_weights)+now-last_clock
      timing_sampled=.false.
      end

      subroutine timing_enter(category)
c Charge the previous category before switching. Reentrant use of the
c same category only pushes the stack; it needs no additional clock read.
      implicit none
      include 'timing_variables.inc'
      integer(kind=8) clock_reads(timing_ncategories)
      common/timing_read_counts/clock_reads
      integer category,previous
      double precision elapsed(timing_ncategories),last_clock,now
      integer stack(0:timing_stack_size),depth,sample_position
      common/timing_clocks/elapsed,last_clock
      common/timing_stack/stack,depth,sample_position
      if(.not.timing_sampled)return
      if(category.lt.1.or.category.gt.timing_ncategories)
     $     error stop 'Invalid timing category'
      if(depth.eq.timing_stack_size)error stop 'Timing stack overflow'
      previous=stack(depth)
      depth=depth+1
      stack(depth)=category
      if(previous.eq.category)return
      clock_reads(previous)=clock_reads(previous)+1_8
      call cpu_time(now)
      elapsed(previous)=elapsed(previous)+now-last_clock
      last_clock=now
      end

      subroutine timing_leave(category)
      implicit none
      include 'timing_variables.inc'
      integer(kind=8) clock_reads(timing_ncategories)
      common/timing_read_counts/clock_reads
      integer category
      double precision elapsed(timing_ncategories),last_clock,now
      integer stack(0:timing_stack_size),depth,sample_position
      common/timing_clocks/elapsed,last_clock
      common/timing_stack/stack,depth,sample_position
      if(.not.timing_sampled)return
      if(depth.eq.0)error stop 'Timing stack underflow'
      if(stack(depth).ne.category)
     $     error stop 'Mismatched timing category'
      depth=depth-1
      if(stack(depth).eq.category)return
      clock_reads(category)=clock_reads(category)+1_8
      call cpu_time(now)
      elapsed(category)=elapsed(category)+now-last_clock
      last_clock=now
      end

      subroutine timing_report(total)
      implicit none
      include 'timing_variables.inc'
      integer(kind=8) clock_reads(timing_ncategories)
      common/timing_read_counts/clock_reads
      double precision total,elapsed(timing_ncategories),last_clock
      integer(kind=8) points,samples,sampler_state
      common/timing_clocks/elapsed,last_clock
      common/timing_counts/points,samples,sampler_state
      integer category,i
      double precision read_cost,calibration_begin,calibration_end
      character*32 names(timing_ncategories)
      parameter(names=[character*32::
     $     'Phase_space','Born_matrix_elements','Real_matrix_elements',
     $     'Virtual_matrix_elements','FKS_local_subtraction',
     $     'FKS_integrated_subtraction','MC_subtraction',
     $     'Clustering_and_merging',
     $     'Cuts_and_jets','Scales_and_couplings','PDF_luminosities',
     $     'Scale_and_PDF_reweighting','EW_Sudakov','Analysis',
     $     'Weights_and_bookkeeping'])
      if(timing_sampled)
     $     error stop 'Reporting an unfinished timing point'
      write(*,*)'Detailed CPU timing: ',samples,' sampled of ',points,
     $     ' integration points (one per 100 points)'
      if(samples.gt.0_8)then
c Sampling removes clock overhead from most points, but multiplying the
c sampled intervals by 100 would magnify that overhead in the estimates.
c Calibrate it once, after the job, and subtract one read per interval.
         call cpu_time(calibration_begin)
         do i=1,256
            call cpu_time(calibration_end)
         enddo
         read_cost=max(0d0,calibration_end-calibration_begin)/256d0
         write(*,*)'Category times are exclusive estimates'
     $        //' (sample x100; clock cost subtracted).'
         write(*,*)'They cover integrand evaluations; setup,'
     $        //' integration control and event-file output are'
     $        //' included only in Total.'
         do category=1,timing_ncategories
c Every point has inclusion probability 1/100, including a partial last
c block. Do not renormalize estimates to the exact full-job CPU time.
            write(*,*)'Time spent in ',trim(names(category)),' : ',
     $           100d0*max(0d0,elapsed(category)
     $           -read_cost*dble(clock_reads(category)))
         enddo
         write(*,*)'Estimated sampled clock overhead : ',
     $        read_cost*dble(sum(clock_reads)+samples)
      else
         write(*,*)'No sampled integration points;'
     $        //' no detailed estimate.'
      endif
c Keep this exact field for job accounting and the log statistics parser.
      write(*,*)'Time spent in Total : ',total
      end

      block data timing_initial_state
c Diagnostic programs can call instrumented routines without a driver.
c They remain untimed until a MINT point explicitly enables sampling.
      implicit none
      include 'timing_variables.inc'
      integer(kind=8) clock_reads(timing_ncategories)
      common/timing_read_counts/clock_reads
      double precision elapsed(timing_ncategories),last_clock
      integer stack(0:timing_stack_size),depth,sample_position
      integer(kind=8) points,samples,sampler_state
      integer stack_entries
      parameter(stack_entries=timing_stack_size+1)
      common/timing_clocks/elapsed,last_clock
      common/timing_stack/stack,depth,sample_position
      common/timing_counts/points,samples,sampler_state
      data clock_reads/timing_ncategories*0_8/
      data elapsed/timing_ncategories*0d0/,last_clock/0d0/
      data stack/stack_entries*timing_weights/
      data depth/0/,sample_position/-1/
      data points/0_8/,samples/0_8/,sampler_state/104729_8/
      data timing_sampled/.false./
      end
