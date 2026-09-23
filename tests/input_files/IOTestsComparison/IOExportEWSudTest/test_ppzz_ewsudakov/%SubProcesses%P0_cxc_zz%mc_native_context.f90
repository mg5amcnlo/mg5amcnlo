module mc_native_context
use mc_born_support
use mc_born_types
implicit none
integer,parameter::local_context=4,local_provider=4
integer,parameter::native_context_ids(1)=[4]
integer,parameter::native_sector_ids(1)=[1]
integer,parameter::native_provider_ids(1)=[4]
integer,parameter::context_providers(10)=[1,2,3,4,1,2,1,2,3,4]
integer,save::active_context=0,active_sector=1,native_epoch=0,active_history=0
logical,save::native_mapping=.false.
integer,save::native_order_map(2,10)=0,native_amplitude_map(2,10)=0
type(BornMetadata),target,save::contexts(10)
type(BornMetadata),pointer,save::native_metadata=>null()
type(BornResult),save::native_result
integer,parameter::history_count=0
integer,save::history_flavours(1,1)=0,history_permutations(5,1)=0
logical,save::history_initialized=.false.
logical,save::shared_real_active=.false.
integer,save::shared_real_epoch=0
real(8),save::shared_real_point(0:3,5)
contains
subroutine ensure_native_context()
if(active_context.eq.0)call activate_native_context(1)
end subroutine
subroutine activate_native_context(sector)
integer,intent(in)::sector
integer context,status
if(sector.lt.1.or.sector.gt.size(native_context_ids))stop "Invalid native sector"
context=native_context_ids(sector)
active_sector=native_sector_ids(sector)
if(context.eq.active_context)return
if(contexts(context)%%context.eq.0)then
call born_query(context_providers(context),context,contexts(context),status)
if(status.ne.0)stop "Native Born metadata query failed"
endif
select case(context)
case(4)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:2,context)=[1,2]
end select
active_context=context
native_metadata=>contexts(context)
native_epoch=native_epoch+1
call mc_sync_native_tables()
end subroutine
subroutine set_native_history(history)
integer,intent(in)::history
if(history.lt.0.or.history.gt.history_count)stop "Invalid native history"
active_history=history
if(history_initialized)return
history_initialized=.true.
end subroutine
subroutine mc_begin_real_point(p)
real(8),intent(in)::p(0:3,5)
shared_real_point=p
shared_real_active=p(0,1).gt.0d0
shared_real_epoch=shared_real_epoch+1
end subroutine
subroutine mc_end_real_point()
shared_real_active=.false.
end subroutine
end module
