class Matrix_0_uux_uux(object):

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        self.jamp = []

    def smatrix(self, p, model, flavor=None):
        #  
        #  MadGraph5_aMC@NLO v. %(version)s, %(date)s
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        # Process: u u~ > u u~
        # Process: c c~ > c c~
        #  
        # Clean additional output
        #
        self.clean()
        #  
        # CONSTANTS
        #  
        nexternal = 4
        ndiags = 4
        ncomb = 16
        #  
        # LOCAL VARIABLES 
        #  
        helicities = [ \
        [1,-1,-1,1],
        [1,-1,-1,-1],
        [1,-1,1,1],
        [1,-1,1,-1],
        [1,1,-1,1],
        [1,1,-1,-1],
        [1,1,1,1],
        [1,1,1,-1],
        [-1,-1,-1,1],
        [-1,-1,-1,-1],
        [-1,-1,1,1],
        [-1,-1,1,-1],
        [-1,1,-1,1],
        [-1,1,-1,-1],
        [-1,1,1,1],
        [-1,1,1,-1]]
        denominator = 36
        # ----------
        # BEGIN CODE
        # ----------
        self.amp2 = [0.] * ndiags
        self.helEvals = []
        ans = 0.
        for hel in helicities:
            t = self.matrix(p, hel, model, flavor)
            ans = ans + t
            self.helEvals.append([hel, t.real / denominator ])
        # Apply flavor-dependent symmetry factor (broken_sym) for merged
        # processes, following the same decay-aware component/block logic as
        # Fortran/C++ templates.
        if flavor is not None:
            _comp_beg = list([1])
            _comp_end = list([2])
            _comp_old = list([1])
            _pid_list = list([2, -2])
            _block_start = list([3, 4])
            _block_len = list([1, 1])
            _pid_work = list(_pid_list)
            _total_factor = 1
            for _icomp in range(1):
                _old_factor = _comp_old[_icomp]
                if _old_factor > 1:
                    for _i in range(_comp_beg[_icomp] - 1, _comp_end[_icomp]):
                        if _pid_work[_i] == 0:
                            continue
                        _n_tot = 1
                        for _j in range(_i + 1, _comp_end[_icomp]):
                            if _pid_work[_i] != _pid_work[_j]:
                                continue
                            if _block_len[_i] != _block_len[_j]:
                                continue
                            _same_block = True
                            for _k in range(_block_len[_i]):
                                if flavor[_block_start[_i] - 1 + _k] != flavor[_block_start[_j] - 1 + _k]:
                                    _same_block = False
                                    break
                            if _same_block:
                                _pid_work[_j] = 0
                                _n_tot += 1
                                _old_factor = _old_factor // _n_tot
                _total_factor *= _old_factor
            ans = ans * _total_factor / denominator
        else:
            ans = ans / denominator
        return ans.real

    def matrix(self, p, hel, model, flavor=None):
        #  
        #  MadGraph5_aMC@NLO v. %(version)s, %(date)s
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        # Process: u u~ > u u~
        # Process: c c~ > c c~
        #  
        #  
        # Process parameters
        #  
        ngraphs = 4
        nexternal = 4
        nwavefuncs = 5
        ncolor = 2
        ZERO = 0.
        #  
        # Color matrix
        #  
        denom = [1,1];
        cf = [[9,3],
        [3,9]];
        #
        # Model parameters
        #
        MZ = model.get('parameter_dict')["MZ"]
        WZ = model.get('parameter_dict')["WZ"]
        GC_10 = model.get('coupling_dict')["GC_10"]
        GC_35 = model.get('coupling_dict')["GC_35"]
        GC_47 = model.get('coupling_dict')["GC_47"]
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        w[0] = ixxxxx(p[0],ZERO,hel[0],+1, flavor[0] if flavor is not None else -1)
        w[1] = oxxxxx(p[1],ZERO,hel[1],-1, flavor[1] if flavor is not None else -1)
        w[2] = oxxxxx(p[2],ZERO,hel[2],+1, flavor[2] if flavor is not None else -1)
        w[3] = ixxxxx(p[3],ZERO,hel[3],-1, flavor[3] if flavor is not None else -1)
        w[4]= FFV1_3(w[0],w[1],GC_10,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp[0]= FFV1_0(w[3],w[2],w[4],GC_10)
        w[4]= FFV2_5_3(w[0],w[1],GC_35,GC_47,MZ,WZ)
        # Amplitude(s) for diagram number 2
        amp[1]= FFV2_5_0(w[3],w[2],w[4],GC_35,GC_47)
        w[4]= FFV1_3(w[0],w[2],GC_10,ZERO,ZERO)
        # Amplitude(s) for diagram number 3
        amp[2]= FFV1_0(w[3],w[1],w[4],GC_10)
        w[4]= FFV2_5_3(w[0],w[2],GC_35,GC_47,MZ,WZ)
        # Amplitude(s) for diagram number 4
        amp[3]= FFV2_5_0(w[3],w[1],w[4],GC_35,GC_47)

        jamp = [None] * ncolor

        jamp[0] = +1./6.*amp[0]-amp[1]+1./2.*amp[2]
        jamp[1] = -1./2.*amp[0]-1./6.*amp[2]+amp[3]

        self.amp2[0]+=abs(amp[0]*amp[0].conjugate())
        self.amp2[1]+=abs(amp[1]*amp[1].conjugate())
        self.amp2[2]+=abs(amp[2]*amp[2].conjugate())
        self.amp2[3]+=abs(amp[3]*amp[3].conjugate())
        matrix = 0.
        for i in range(ncolor):
            ztemp = 0
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   
        self.jamp.append(jamp)

        return matrix
