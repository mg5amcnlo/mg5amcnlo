"""Export native MC histories into the runtime's FKS metadata index space.

The first FKS_INTEGRATED entries are the ordinary integration channels. Extra
entries are native evaluation contexts, never new integration channels. Born
implementations stay in the output-wide library; only immutable tables and
small real/PDF dispatchers are installed in each subprocess.
"""

import re

from madgraph import MadGraph5Error
from madgraph.iolibs import born_support as born
from madgraph.iolibs import file_writers


def write(path, lines):
    if path.is_symlink():
        path.unlink()
    with file_writers.FortranWriter(str(path)) as stream:
        stream.writelines(lines)


def parameter(lines, name):
    return int(next(re.search(r'\b'+name+r'\s*=\s*(\d+)', line, re.I)[1]
                    for line in lines if re.search(r'\b'+name+r'\s*=\s*\d+', line, re.I)))


def sector_data(lines, old, new, orders=None):
    """Select one native row from the generated DATA statements."""
    result = []
    for line in lines:
        match = re.match(r'\s*data\s+(\w+)\s*/(.*)/\s*$', line, re.I)
        if match:
            values = match[2].split(',')
            value = values[old-1].strip()
            if orders and match[1].lower() in ('isplitorder_born_d','isplitorder_cnt_d'):
                value = str(orders[int(value)-1]) if int(value) else '0'
            result.append('data %s(%d) /%s/' % (match[1],new,value))
            continue
        if not re.match(r'\s*data\b', line, re.I):
            continue
        match = re.search(r'(\w+)\s*\(\s*(\d+)\s*,', line)
        if not match or int(match[2]) != old:
            continue
        line = line[:match.start(2)]+str(new)+line[match.end(2):]
        if orders and match[1].lower() == 'split_type_d':
            head, values, tail = line.split('/')
            values = values.split(',')
            reordered = ['.false.']*len(orders)
            for value, target in zip(values,orders):
                reordered[target-1] = value.strip()
            line = head+'/'+','.join(reordered)+'/'+tail
        result.append(line)
    return result


def dat_sector(text, old, new):
    """Select a sector, retaining the daughter records of its forest nodes."""
    result, daughters = [], 0
    for line in text.splitlines():
        if not line or line[0] == '#':
            continue
        if line[0] == 'D':
            if daughters:
                result.append(line)
                daughters -= 1
            continue
        parts = line.split()
        if int(parts[1]) != old:
            continue
        parts[1] = str(new)
        result.append(' '.join(parts))
        if parts[0] == 'F':
            daughters = int(parts[4])
    return result


def export(root, records, histories):
    # Snapshot before altering any subprocess: a later directory may need the
    # original metadata, PDF routine or mapping properties of an earlier one.
    sources = {}
    for r in records:
        directory = root/'SubProcesses'/r['directory']
        names = ['fks_info.inc','fks_symmetry.inc','leshouche_decl.inc',
                 'leshouche_info.dat','configs_and_props_decl.inc',
                 'configs_and_props_info.dat','born_props.inc','born_nhel.inc',
                 'genps.inc','pmass.inc','real_from_born_configs.inc','born_conf.inc']
        names += [p.name for p in directory.glob('parton_lum_*.f')]
        sources[r['context']] = {name:(directory/name).read_text() for name in names}
    for outer in records:
        write_context(root, outer, records, histories[outer['directory']], sources)


def write_context(root, outer, records, tables, sources):
    directory = root/'SubProcesses'/outer['directory']
    by_id = {r['context']:r for r in records}
    local = outer['context']
    native_keys = {(h['context'],h['sector']) for t in tables for h in t['histories']}
    original = [(local,s['sector']) for s in outer['sectors']] or [(local,1)]
    keys = original+sorted(native_keys-set(original))
    aliases = {key:i for i,key in enumerate(keys,1)}
    reachable = sorted({c for c,s in keys})
    count, integrated = len(keys), len(original)
    nexternal = outer['nexternal']
    maxproc = max(parameter(born.statements(sources[c]['genps.inc']),'maxproc') for c in reachable)
    maxflow = max(parameter(born.statements(sources[c]['leshouche_decl.inc']),'maxflow_used') for c in reachable)
    maxbcol = max(by_id[c]['ncolor'] for c in reachable)
    # Born tables also use these buffers. In particular LO-only contexts have
    # no real configurations, but still need space for their Born topologies.
    maxconfigs = max(max(
        parameter(born.statements(sources[c]['configs_and_props_decl.inc']),'lmaxconfigs_used'),
        parameter(born.statements(sources[c]['born_conf.inc']),'lmaxconfigsb_used'))
        for c in reachable)
    maxbranch = max(parameter(born.statements(sources[c]['configs_and_props_decl.inc']),'max_branch_used') for c in reachable)
    write(directory/'nFKSconfigs.inc', ['integer FKS_CONFIGS,FKS_INTEGRATED',
        'parameter(FKS_CONFIGS=%d,FKS_INTEGRATED=%d)' % (count,integrated)])
    genps = list(born.statements(sources[local]['genps.inc']))
    genps = [re.sub(r'(?i)\b(maxproc|maxflow)\s*=\s*\d+',
              lambda m:m[1]+'='+str(maxproc if m[1].lower() == 'maxproc' else maxflow),line) for line in genps]
    write(directory/'genps.inc',genps)
    # All real topology buffers must accommodate the native tables. The local
    # Born numerical COMMON extents (NGRAPHS, NCOLOR) are deliberately unchanged.
    write(directory/'maxconfigs.inc',['integer LMAXCONFIGS','parameter(LMAXCONFIGS=%d)' % maxconfigs])
    nhel = [re.sub(r'(?i)max_bcol\s*=\s*\d+','MAX_BCOL=%d' % maxbcol,line)
            for line in born.statements(sources[local]['born_nhel.inc'])]
    write(directory/'born_nhel.inc',nhel)
    fks = ['integer ipos,jpos',
           'integer fks_i_d(%d),fks_j_d(%d),extra_cnt_d(%d)' % (count,count,count),
           'integer isplitorder_born_d(%d),isplitorder_cnt_d(%d)' % (count,count),
           'integer fks_j_from_i_d(%d,nexternal,0:nexternal)' % count,
           'integer particle_type_d(%d,nexternal),pdg_type_d(%d,nexternal)' % (count,count),
           'logical particle_tag_d(%d,nexternal)' % count,
           'double precision particle_charge_d(%d,nexternal)' % count,
           'logical split_type_d(%d,%d)' % (count,len(outer['orders'])),
           'logical need_color_links_d(%d),need_charge_links_d(%d)' % (count,count)]
    symmetry = ['integer FKS_FAC_I_D(%d),FKS_FAC_J_D(%d)' % (count,count),
                'double precision FKS_IDEN_BORN_D(%d),FKS_IDEN_REAL_D(%d)' % (count,count)]
    leshouche, configs = [], []
    for alias,(c,s) in enumerate(keys,1):
        ordermap = [outer['orders'].index(o)+1 for o in by_id[c]['orders']]
        fks += sector_data(born.statements(sources[c]['fks_info.inc']),s,alias,ordermap)
        symmetry += sector_data(born.statements(sources[c]['fks_symmetry.inc']),s,alias)
        leshouche += dat_sector(sources[c]['leshouche_info.dat'],s,alias)
        configs += dat_sector(sources[c]['configs_and_props_info.dat'],s,alias)
    write(directory/'fks_info.inc',fks)
    write(directory/'fks_symmetry.inc',symmetry)
    (directory/'leshouche_info.dat').write_text('\n'.join(leshouche)+'\n')
    (directory/'configs_and_props_info.dat').write_text('\n'.join(configs)+'\n')
    for name, scalar_changes in [('leshouche_decl.inc',dict(maxproc_used=maxproc,maxflow_used=maxflow)),
                                  ('configs_and_props_decl.inc',dict(lmaxconfigs_used=maxconfigs,max_branch_used=maxbranch))]:
        lines = []
        for line in born.statements(sources[local][name]):
            for scalar,value in scalar_changes.items():
                line = re.sub(r'(?i)\b'+scalar+r'\s*=\s*\d+',scalar+'='+str(value),line)
            # Each *_D array has the sector as its first dimension.
            line = re.sub(r'(?i)(\b\w+_d\s*\()\s*\d+',r'\g<1>'+str(count),line)
            lines.append(line)
        write(directory/name,lines)
    mappings = ['integer irfbc',
                'integer real_from_born_conf(lmaxconfigs,%d)' % count]
    for alias,(c,sector) in enumerate(keys,1):
        for line in born.statements(sources[c]['real_from_born_configs.inc']):
            match = re.search(r'(?i)real_from_born_conf\(irfbc,(\d+)\)',line)
            if not line.lower().lstrip().startswith('data') or not match or int(match[1]) != sector:
                continue
            mappings.append(line[:match.start(1)]+str(alias)+line[match.end(1):])
    write(directory/'real_from_born_configs.inc',mappings)
    rows = write_histories(directory,outer,tables,aliases)
    write_module(directory,outer,by_id,keys,reachable,rows,maxproc)
    write_born_tables(directory)
    write_dispatchers(directory,outer,by_id,keys,sources)
    props = ['subroutine mc_native_props(pmass,pwidth,pow)',
             'use mc_native_context, only: active_context,ensure_native_context',
             'implicit none', "include 'nexternal.inc'", "include 'maxconfigs.inc'",
             "include 'coupl.inc'",'double precision zero','parameter(zero=0d0)',
             'double precision pmass(-nexternal:0,lmaxconfigs),pwidth(-nexternal:0,lmaxconfigs)',
             'integer pow(-nexternal:0,lmaxconfigs)',
             'call ensure_native_context()', 'pmass=0d0','pwidth=0d0','pow=0','select case(active_context)']
    for c in reachable:
        props += ['case(%d)' % c]+list(born.statements(sources[c]['born_props.inc']))
    props += ['end select','end']
    write(directory/'mc_native_props.f',props)
    write(directory/'born_props.inc',['call mc_native_props(pmass,pwidth,pow)'])
    write(directory/'pmass.inc',['call mc_native_masses(pmass)'])


def write_histories(directory, outer, tables, aliases):
    rows,first,last,own,complete = [],[],[],[],[]
    for table in tables:
        first.append(len(rows)+1)
        own.append(0)
        for h in table['histories']:
            rows.append(h)
            if (h['context'] == outer['context'] and h['sector'] == table['sector'] and
                    list(h['permutation']) == list(range(1,outer['nexternal']+1))):
                own[-1] = len(rows)
        last.append(len(rows))
        complete.append(not(table['unresolved'] or table['ambiguous']))
        if not own[-1]:
            raise MadGraph5Error('Missing native owner in global history table')
    if not tables:
        first,last,own,complete = [1],[0],[0],[True]
    nrows, nsectors = max(1,len(rows)),len(aliases)
    lines = ['integer MC_HIST_COUNT','parameter(MC_HIST_COUNT=%d)' % len(rows),
             'integer MC_HIST_FIRST(%d),MC_HIST_LAST(%d),MC_HIST_OWN(%d)' % (nsectors,nsectors,nsectors),
             'logical MC_HIST_COMPLETE(%d)' % nsectors,
             'integer MC_HIST_NATIVE(%d),MC_HIST_I(%d),MC_HIST_J(%d)' % (nrows,nrows,nrows),
             'integer MC_HIST_PERM(nexternal,%d)' % nrows,'integer mc_hist_k']
    for i,(a,b,c,d) in enumerate(zip(first,last,own,complete),1):
        lines += ['data MC_HIST_FIRST(%d),MC_HIST_LAST(%d),MC_HIST_OWN(%d) /%d,%d,%d/' % (i,i,i,a,b,c),
                  'data MC_HIST_COMPLETE(%d) /%s/' % (i,'.true.' if d else '.false.')]
    for i,h in enumerate(rows,1):
        lines += ['data MC_HIST_NATIVE(%d),MC_HIST_I(%d),MC_HIST_J(%d) /%d,%d,%d/' %
                  (i,i,i,aliases[h['context'],h['sector']],h['i'],h['j']),
                  'data (MC_HIST_PERM(mc_hist_k,%d),mc_hist_k=1,nexternal) /%s/' %
                  (i,','.join(map(str,h['permutation'])))]
    write(directory/'mc_histories.inc',lines)
    return rows


def write_module(directory,outer,by_id,keys,reachable,rows,maxproc):
    nctx = max(by_id)
    def array(name,values):
        return 'integer,parameter::%s(%d)=[%s]' % (name,len(values),','.join(map(str,values)))
    lines = ['module mc_native_context','use mc_born_support','use mc_born_types','implicit none',
             'integer,parameter::local_context=%d,local_provider=%d' % (outer['context'],outer['provider']),
             array('native_context_ids',[c for c,s in keys]),
             array('native_sector_ids',[s for c,s in keys]),
             array('native_provider_ids',[by_id[c]['provider'] for c,s in keys]),
             array('context_providers',[by_id[c]['provider'] for c in range(1,nctx+1)]),
             'integer,save::active_context=0,active_sector=1,native_epoch=0,active_history=0',
             'logical,save::native_mapping=.false.',
             'integer,save::native_order_map(%d,%d)=0,native_amplitude_map(%d,%d)=0' %
             (len(outer['orders']),nctx,max(len(r['amp_orders']) for r in by_id.values()),nctx),
             'type(BornMetadata),target,save::contexts(%d)' % nctx,
             'type(BornMetadata),pointer,save::native_metadata=>null()',
             'type(BornResult),save::native_result',
             'integer,parameter::history_count=%d' % len(rows),
             'integer,save::history_flavours(%d,%d)=0,history_permutations(%d,%d)=0' %
             (maxproc,max(1,len(rows)),outer['nexternal'],max(1,len(rows))),
             'logical,save::history_initialized=.false.',
             'contains','subroutine ensure_native_context()',
             'if(active_context.eq.0)call activate_native_context(1)','end subroutine',
             'subroutine activate_native_context(sector)', 'integer,intent(in)::sector', 'integer context,status',
             'if(sector.lt.1.or.sector.gt.size(native_context_ids))stop "Invalid native sector"',
             'context=native_context_ids(sector)','active_sector=native_sector_ids(sector)',
             'if(context.eq.active_context)return',
             'if(contexts(context)%context.eq.0)then',
             'call born_query(context_providers(context),context,contexts(context),status)',
             'if(status.ne.0)stop "Native Born metadata query failed"','endif',
             'select case(context)']
    for c in reachable:
        r = by_id[c]
        order_map = [outer['orders'].index(o)+1 for o in r['orders']]
        amp_map = []
        for powers in r['amp_orders']:
            powers = [powers[r['orders'].index(o)] for o in outer['orders']]
            amp_map.append(next((i for i,p in enumerate(outer['amp_orders'],1) if list(p) == powers),0))
        lines += ['case(%d)' % c,'native_order_map(:,context)=[%s]' % ','.join(map(str,order_map)),
                  'native_amplitude_map(1:%d,context)=[%s]' % (len(amp_map),','.join(map(str,amp_map)))]
    lines += ['end select','active_context=context','native_metadata=>contexts(context)',
             'native_epoch=native_epoch+1','call mc_sync_native_tables()','end subroutine',
             'subroutine set_native_history(history)','integer,intent(in)::history',
             'if(history.lt.0.or.history.gt.history_count)stop "Invalid native history"',
             'active_history=history','if(history_initialized)return']
    for i,h in enumerate(rows,1):
        lines += ['history_flavours(1:%d,%d)=[%s]' % (len(h['flavours']),i,','.join(map(str,h['flavours']))),
                  'history_permutations(:,%d)=[%s]' % (i,','.join(map(str,h['permutation'])))]
    lines += ['history_initialized=.true.','end subroutine','end module']
    (directory/'mc_native_context.f90').write_text('\n'.join(lines)+'\n')


def write_born_tables(directory):
    write(directory/'born_leshouche.inc',['integer iproc_born','common/mc_native_born_lh/idup,mothup,icolup',
        'common/mc_native_born_count/iproc_born'])
    write(directory/'born_conf.inc',[
        'integer lmaxconfigsb_used,max_branchb_used',
        'parameter(lmaxconfigsb_used=lmaxconfigs,max_branchb_used=max_branch)',
        'integer mapconfig(0:lmaxconfigs),iforest(2,-max_branch:-1,lmaxconfigs)',
        'integer sprop(-max_branch:-1,lmaxconfigs),tprid(-max_branch:-1,lmaxconfigs)',
        'logical gforcebw(-max_branch:-1,lmaxconfigs)',
        'common/mc_native_born_topology/mapconfig,iforest,sprop,tprid,gforcebw'])
    write(directory/'born_coloramps.inc',['logical icolamp(max_bcol,lmaxconfigs,1)',
        'common/mc_native_born_colours/icolamp'])


def write_dispatchers(directory,outer,by_id,keys,sources):
    real = ['subroutine smatrix_real(p,wgt)','implicit none',"include 'nexternal.inc'",
            'double precision p(0:3,nexternal),q(0:3,nexternal),wgt','integer nfksprocess',
            'common/c_nfksprocess/nfksprocess','select case(nfksprocess)']
    lumi = ['double precision function dlum()','implicit none','integer nfksprocess',
            'common/c_nfksprocess/nfksprocess','select case(nfksprocess)']
    copied = set()
    for alias,(c,s) in enumerate(keys,1):
        native = by_id[c]
        sector = native['sectors'][s-1] if native['sectors'] else dict(real=0)
        lumi += ['case(%d)' % alias]
        if c == outer['context']:
            real += ['case(%d)' % alias, 'call smatrix%d(p,wgt)' % sector['real']] if native['sectors'] else []
            lumi += ['call dlum_%d(dlum)' % sector['real']]
            continue
        # A native real amplitude is already present in the owner directory.
        # Resolve its labelled input order explicitly, independently of PDF
        # list ordering. No second real implementation is emitted.
        match = next(((os,perm) for os in outer['sectors'] for np in sector['processes']
                      for op in os['processes'] for perm in born.labelled_maps(born.freeze(np),born.freeze(op),outer['nincoming'],anchors=())),None)
        if match is None:
            raise MadGraph5Error('No local real evaluator for native history %s' % ((c,s),))
        os,perm = match
        real += ['case(%d)' % alias]+['q(:,%d)=p(:,%d)' % (target,i) for i,target in enumerate(perm,1)]
        real += ['call smatrix%d(q,wgt)' % os['real']]
        name = 'dlum_native_%d_%d' % (c,sector['real'])
        lumi += ['call %s(dlum)' % name]
        if name not in copied:
            copied.add(name)
            lines = [born.rename(line,{'dlum_%d' % sector['real']:name})
                     for line in born.statements(sources[c]['parton_lum_%d.f' % sector['real']])]
            write(directory/('parton_lum_native_%d_%d.f' % (c,sector['real'])),lines)
    real += ['case default','stop "Invalid native real sector"','end select','end']
    lumi += ['case default','stop "Invalid native PDF sector"','end select','call mc_filter_native_lum(dlum)','end']
    write(directory/'real_me_chooser.f',real)
    write(directory/'parton_lum_chooser.f',lumi)
