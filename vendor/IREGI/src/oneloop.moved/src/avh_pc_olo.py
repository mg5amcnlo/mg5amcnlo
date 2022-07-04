import re
import avh_pc


def xprec(srcdir,rctype,prec,xkind,xkindmod):
    lines = []
    if rctype == '':
        return
    elif rctype == 'intrinsic':
        if xkind == '':
            return
        else:
            lines = open(srcdir+'avh_olo_kinds.f90').readlines()
            if xkindmod != '':
                avh_pc.case('XKIND','yes',lines)
                avh_pc.subs('XKINDMOD',xkindmod,lines)
                avh_pc.subs('kindr2=>KINDPAR','kindr2=>'+xkind,lines)
            else:
                avh_pc.case('XKIND','no',lines)
                avh_pc.subs('kindr2=KINDPAR','kindr2='+xkind,lines)
    lines.extend(open(srcdir+'avh_olo_arrays.f90'    ).readlines())
    lines.extend(open(srcdir+'avh_olo_'+rctype+'.f90').readlines())
    lines.extend(open(srcdir+'avh_olo_print.f90'     ).readlines())
    lines.extend(open(srcdir+'avh_olo_auxfun.f90'    ).readlines())
    lines.extend(open(srcdir+'avh_olo_olog.f90'      ).readlines())
    lines.extend(open(srcdir+'avh_olo_dilog.f90'     ).readlines())
    lines.extend(open(srcdir+'avh_olo_bnlog.f90'     ).readlines())
    lines.extend(open(srcdir+'avh_olo_qmplx.f90'     ).readlines())
    lines.extend(open(srcdir+'avh_olo_bub.f90'       ).readlines())
    lines.extend(open(srcdir+'avh_olo_tri.f90'       ).readlines())
    lines.extend(open(srcdir+'avh_olo_box.f90'       ).readlines())
    lines.extend(open(srcdir+'avh_olo_boxc.f90'      ).readlines())
    lines.extend(open(srcdir+'avh_olo_main.f90'      ).readlines())
    avh_pc.incl(srcdir,lines)
    avh_pc.subs('avh_olo','avh_olo_'+prec,lines)
    avh_pc.subs('avh_olo_'+prec+'_version','avh_olo_version',lines)
    avh_pc.subs('avh_olo_'+prec+'_units','avh_olo_units',lines)
    if rctype == 'intrinsic':
        avh_pc.case('RCTYPE','intrinsic',lines)
    else:
        avh_pc.case('RCTYPE',prec+'type',lines)
    avh_pc.case('RCPROG',rctype,lines)
    return lines


def full( srcdir_in ,dpkind,qpkind,kindmod ,ddtype,qdtype,mptype ,tlevel,cppintf ):
    srcdir = re.sub(r'\/*$','/',srcdir_in)
    callingme = 'no'
    protected = 'no'
    filename = 'avh_olo.f90'

    if dpkind+qpkind+ddtype+qdtype+mptype == '':
        avh_pc.prnt('avh_pc_olo.py: no kind/type provided, putting -dpkind="kind(1d0)"')
        dpkind = 'kind(1d0)'
    
    lines = open(srcdir+'avh_olo_version.f90').readlines()
    lines.extend(open(srcdir+'avh_olo_units.f90').readlines())
    comblines = open(srcdir+'avh_olo_comb.f90').readlines()
    
    if dpkind != '':
        lines.extend( xprec(srcdir,'intrinsic','dp',dpkind,kindmod) )
        avh_pc.case('dp','yes',comblines)
    else:
        avh_pc.case('dp','no',comblines)
    
    if qpkind != '':
        lines.extend( xprec(srcdir,'intrinsic','qp',qpkind,kindmod) )
        avh_pc.case('qp','yes',comblines)
    else:
        avh_pc.case('qp','no',comblines)
    
    if ddtype != '':
        lines.extend( xprec(srcdir,ddtype,'dd','','') )
        avh_pc.case('dd','yes',comblines)
    else:
        avh_pc.case('dd','no',comblines)
    
    if qdtype != '':
        lines.extend( xprec(srcdir,qdtype,'qd','','') )
        avh_pc.case('qd','yes',comblines)
    else:
        avh_pc.case('qd','no',comblines)
    
    if mptype != '':
        lines.extend( xprec(srcdir,mptype,'mp','','') )
        avh_pc.case('mp','yes',comblines)
    else:
        avh_pc.case('mp','no',comblines)
    
    lines.extend(comblines)
    
    if tlevel == 'yes':
        lines.extend(open(srcdir+'avh_olo_wrp01.f90').readlines())
    
    avh_pc.case('cppINTERFACE',cppintf,lines)
    avh_pc.dich('CALLINGME',callingme,lines)
    avh_pc.dich('PROTECTED',protected,lines)
    avh_pc.clean(lines)
    
    iofile = open(filename,'w')
    iofile.writelines(lines)
    iofile.close()

    return filename


