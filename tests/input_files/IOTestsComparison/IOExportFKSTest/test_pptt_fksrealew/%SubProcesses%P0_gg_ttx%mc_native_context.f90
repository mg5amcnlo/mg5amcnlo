module mc_native_context
use mc_born_support
use mc_born_types
implicit none
integer,parameter::local_context=5,local_provider=5
integer,parameter::native_context_ids(18)=[5,5,5,5,5,5,5,5,5,5,2,2,3,3,6,6,7,7]
integer,parameter::native_sector_ids(18)=[1,2,3,4,5,6,7,8,9,10,10,12,10,12,10,12,10,12]
integer,parameter::native_provider_ids(18)=[5,5,5,5,5,5,5,5,5,5,2,2,3,3,6,6,7,7]
integer,parameter::context_providers(7)=[1,2,3,4,5,6,7]
integer,save::active_context=0,active_sector=1,native_epoch=0,active_history=0
logical,save::native_mapping=.false.
integer,save::native_order_map(2,7)=0,native_amplitude_map(6,7)=0
type(BornMetadata),target,save::contexts(7)
type(BornMetadata),pointer,save::native_metadata=>null()
type(BornResult),save::native_result
integer,parameter::history_count=20
integer,save::history_flavours(2,20)=0,history_permutations(5,20)=0
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
case(2)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:6,context)=[1,0,0,3,2,4]
case(3)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:6,context)=[1,0,0,3,2,4]
case(5)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:4,context)=[1,2,3,4]
case(6)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:6,context)=[1,0,0,3,2,4]
case(7)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:6,context)=[1,0,0,3,2,4]
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
history_flavours(1:2,1)=[1,2]
history_permutations(:,1)=[1,2,3,4,5]
history_flavours(1:2,2)=[1,2]
history_permutations(:,2)=[1,2,3,4,5]
history_flavours(1:2,3)=[1,2]
history_permutations(:,3)=[1,2,3,4,5]
history_flavours(1:2,4)=[1,2]
history_permutations(:,4)=[1,2,3,4,5]
history_flavours(1:2,5)=[1,2]
history_permutations(:,5)=[1,2,3,4,5]
history_flavours(1:2,6)=[1,2]
history_permutations(:,6)=[1,2,3,4,5]
history_flavours(1:2,7)=[1,2]
history_permutations(:,7)=[1,2,3,4,5]
history_flavours(1:2,8)=[1,2]
history_permutations(:,8)=[1,2,3,4,5]
history_flavours(1:2,9)=[1,2]
history_permutations(:,9)=[1,2,3,4,5]
history_flavours(1:2,10)=[1,2]
history_permutations(:,10)=[1,2,3,4,5]
history_flavours(1:2,11)=[1,2]
history_permutations(:,11)=[1,2,3,4,5]
history_flavours(1:2,12)=[1,2]
history_permutations(:,12)=[1,2,3,4,5]
history_flavours(1:2,13)=[1,2]
history_permutations(:,13)=[1,2,3,4,5]
history_flavours(1:2,14)=[1,2]
history_permutations(:,14)=[1,2,3,4,5]
history_flavours(1:2,15)=[1,2]
history_permutations(:,15)=[1,2,3,4,5]
history_flavours(1:2,16)=[1,2]
history_permutations(:,16)=[1,2,3,4,5]
history_flavours(1:1,17)=[1]
history_permutations(:,17)=[1,2,3,4,5]
history_flavours(1:1,18)=[1]
history_permutations(:,18)=[1,2,3,4,5]
history_flavours(1:1,19)=[1]
history_permutations(:,19)=[1,2,3,4,5]
history_flavours(1:1,20)=[1]
history_permutations(:,20)=[1,2,3,4,5]
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
