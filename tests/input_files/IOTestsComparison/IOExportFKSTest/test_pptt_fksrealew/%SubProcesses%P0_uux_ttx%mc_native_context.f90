module mc_native_context
use mc_born_support
use mc_born_types
implicit none
integer,parameter::local_context=6,local_provider=6
integer,parameter::native_context_ids(16)=[6,6,6,6,6,6,6,6,6,6,6,6,1,4,5,5]
integer,parameter::native_sector_ids(16)=[1,2,3,4,5,6,7,8,9,10,11,12,6,7,4,7]
integer,parameter::native_provider_ids(16)=[6,6,6,6,6,6,6,6,6,6,6,6,1,4,5,5]
integer,parameter::context_providers(7)=[1,2,3,4,5,6,7]
integer,save::active_context=0,active_sector=1,native_epoch=0,active_history=0
logical,save::native_mapping=.false.
integer,save::native_order_map(2,7)=0,native_amplitude_map(6,7)=0
type(BornMetadata),target,save::contexts(7)
type(BornMetadata),pointer,save::native_metadata=>null()
type(BornResult),save::native_result
integer,parameter::history_count=40
integer,save::history_flavours(2,40)=0,history_permutations(5,40)=0
logical,save::history_initialized=.false.
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
case(1)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:2,context)=[2,4]
case(4)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:2,context)=[2,4]
case(5)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:4,context)=[1,5,4,6]
case(6)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:6,context)=[1,2,3,4,5,6]
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
history_flavours(1:2,17)=[1,2]
history_permutations(:,17)=[1,2,3,4,5]
history_flavours(1:2,18)=[1,2]
history_permutations(:,18)=[1,2,3,4,5]
history_flavours(1:2,19)=[1,2]
history_permutations(:,19)=[1,2,3,4,5]
history_flavours(1:2,20)=[1,2]
history_permutations(:,20)=[1,2,3,4,5]
history_flavours(1:2,21)=[1,2]
history_permutations(:,21)=[1,2,3,4,5]
history_flavours(1:2,22)=[1,2]
history_permutations(:,22)=[1,2,3,4,5]
history_flavours(1:2,23)=[1,2]
history_permutations(:,23)=[1,2,3,4,5]
history_flavours(1:2,24)=[1,2]
history_permutations(:,24)=[1,2,3,4,5]
history_flavours(1:2,25)=[1,2]
history_permutations(:,25)=[1,2,3,4,5]
history_flavours(1:2,26)=[1,2]
history_permutations(:,26)=[1,2,3,4,5]
history_flavours(1:2,27)=[1,2]
history_permutations(:,27)=[1,2,3,4,5]
history_flavours(1:2,28)=[1,2]
history_permutations(:,28)=[1,2,3,4,5]
history_flavours(1:2,29)=[1,2]
history_permutations(:,29)=[1,2,3,4,5]
history_flavours(1:2,30)=[1,2]
history_permutations(:,30)=[1,2,3,4,5]
history_flavours(1:2,31)=[1,2]
history_permutations(:,31)=[1,2,3,4,5]
history_flavours(1:2,32)=[1,2]
history_permutations(:,32)=[1,2,3,4,5]
history_flavours(1:2,33)=[1,2]
history_permutations(:,33)=[1,2,3,4,5]
history_flavours(1:2,34)=[1,2]
history_permutations(:,34)=[1,2,3,4,5]
history_flavours(1:2,35)=[1,2]
history_permutations(:,35)=[1,2,3,4,5]
history_flavours(1:2,36)=[1,2]
history_permutations(:,36)=[1,2,3,4,5]
history_flavours(1:2,37)=[1,2]
history_permutations(:,37)=[1,2,3,4,5]
history_flavours(1:2,38)=[1,2]
history_permutations(:,38)=[1,2,3,4,5]
history_flavours(1:2,39)=[1,2]
history_permutations(:,39)=[1,2,3,4,5]
history_flavours(1:2,40)=[1,2]
history_permutations(:,40)=[1,2,3,4,5]
history_initialized=.true.
end subroutine
end module
