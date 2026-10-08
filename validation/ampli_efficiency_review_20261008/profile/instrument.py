"""Add diagnostic CPU timers to an isolated exported process, never repository sources."""
import difflib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
paths = json.loads((HERE / 'paths.json').read_text())
SP = Path(paths['profile']) / 'SubProcesses'

def change(name, replacements):
    path = SP / name
    before = (Path(paths['original']) / 'SubProcesses' / name).read_text()
    after = before
    for old, new in replacements:
        assert after.count(old) == 1, (name, old, after.count(old))
        after = after.replace(old, new)
    path.write_text(after)
    (HERE / (name + '.patch')).write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile=name, tofile=name)))

change('ampli_mint_adapter.f90', [
    ('  integer(kind=8),save :: production_points=0_8',
     '  real(kind=8),save :: p_sample=0d0,p_map=0d0,p_fun=0d0,p_virt=0d0,p_consider=0d0,p_finish=0d0\n'
     '  integer(kind=8),save :: p_nfun=0_8,p_nvirt=0_8\n'
     '  integer(kind=8),save :: production_points=0_8'),
    ('    integer :: folds(ndimmax),iret,ifl,ipoint\n    call sampler%sample(tmp,unused,base)',
     '    integer :: folds(ndimmax),iret,ifl,ipoint\n'
     '    real(kind=8) :: pt0,pt1,pt2,pt3,fun_elapsed\n'
     '    character(len=4) :: prof_abrv\n'
     '    common /to_abrv/ prof_abrv\n'
     '    call cpu_time(pt0)\n'
     '    call sampler%sample(tmp,unused,base)\n'
     '    call cpu_time(pt1)\n'
     '    p_sample=p_sample+pt1-pt0\n'
     '    fun_elapsed=0d0'),
    ('       call sampler%map_fold(base,folds(1:ndim),ifold(1:ndim),x(1:ndim),vol)',
     '       call cpu_time(pt0)\n'
     '       call sampler%map_fold(base,folds(1:ndim),ifold(1:ndim),x(1:ndim),vol)'),
    ('       dummy=fun(x,vol,ifl,values)',
     '       call cpu_time(pt1)\n'
     '       p_map=p_map+pt1-pt0\n'
     '       dummy=fun(x,vol,ifl,values)\n'
     '       call cpu_time(pt2)\n'
     '       fun_elapsed=fun_elapsed+pt2-pt1'),
    ('    dummy=fun(x,vol,2,values)\n  end subroutine ampli_evaluate',
     '    call cpu_time(pt1)\n'
     '    dummy=fun(x,vol,2,values)\n'
     '    call cpu_time(pt2)\n'
     '    fun_elapsed=fun_elapsed+pt2-pt1\n'
     '    p_fun=p_fun+fun_elapsed\n'
     '    p_nfun=p_nfun+1_8\n'
     "    if (prof_abrv.eq.'virt') then\n"
     '       p_virt=p_virt+fun_elapsed\n'
     '       p_nvirt=p_nvirt+1_8\n'
     '    endif\n'
     '  end subroutine ampli_evaluate'),
    ('    logical :: iteration_done\n    character(len=4) :: abrv',
     '    logical :: iteration_done\n    real(kind=8) :: pt0,pt1\n    character(len=4) :: abrv'),
    ('       call sampler%finish_native_iteration(done)',
     '       call cpu_time(pt0)\n       call sampler%finish_native_iteration(done)\n'
     '       call cpu_time(pt1)\n       p_finish=p_finish+pt1-pt0'),
    ('    call sampler%native_consider(point,x(1:ndim),to_write,iteration_done)',
     '    call cpu_time(pt0)\n    call sampler%native_consider(point,x(1:ndim),to_write,iteration_done)\n'
     '    call cpu_time(pt1)\n    p_consider=p_consider+pt1-pt0'),
    ("    write(*,*) 'AmpliCol generation trials, candidates, requested events:', &",
     "    write(*,*) 'PROFILE sample,map,fun,virt,consider,finish:', &\n"
     '         p_sample,p_map,p_fun,p_virt,p_consider,p_finish\n'
     "    write(*,*) 'PROFILE integrand trials,virtual trials:',p_nfun,p_nvirt\n"
     "    write(*,*) 'AmpliCol generation trials, candidates, requested events:', &"),
])

text = (Path(paths['original']) / 'SubProcesses/simple_integrator.f90').read_text()
start = text.index('  subroutine finish_native_iteration(this,done)')
end = text.index('  end subroutine finish_native_iteration', start)
body = text[start:end]
# Timers only at iteration boundaries, outside the candidate/Jacobian loops.
decl = '    class(staged_integrator),intent(inout) :: this'
assert decl in body
new = body.replace(decl, decl + '\n    real(kind=8) :: pt0,pt1\n'
    '    real(kind=8),save :: p_envelope=0d0,p_select=0d0,p_updates=0d0,p_cutoff=0d0')
for statement, accumulator in [
    ('call this%native_update_envelopes()', 'p_envelope'),
    ('call this%native_select()', 'p_select'),
    ('call this%native_update_maps()', 'p_updates'),
    ('cutoff=this%native_next_cutoff()', 'p_cutoff'),
]:
    assert new.count(statement) == 1
    new = new.replace(statement, 'call cpu_time(pt0)\n    ' + statement +
        '\n    call cpu_time(pt1)\n    ' + accumulator + '=' + accumulator + '+pt1-pt0')
statement = '    if (done) then'
assert statement in new
new = new.replace(statement, statement +
    "\n       write(*,*) 'PROFILE envelope,select,map_updates,next_cutoff:', &\n"
    '            p_envelope,p_select,p_updates,p_cutoff')
change('simple_integrator.f90', [(body, new)])
