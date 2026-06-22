class Matrix_%(process_string)s(object):

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        self.jamp = []

    def smatrix(self, p, model, flavor=None):
        #  
        %(info_lines)s
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        %(process_lines)s
        #  
        # Clean additional output
        #
        self.clean()
        #  
        # CONSTANTS
        #  
        nexternal = %(nexternal)d
        ndiags = %(ndiags)d
        ncomb = %(ncomb)d
        #  
        # LOCAL VARIABLES 
        #  
        %(helicity_lines)s
        %(den_factor_line)s
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
            _comp_beg = list(%(broken_sym_component_starts)s)
            _comp_end = list(%(broken_sym_component_ends)s)
            _comp_old = list(%(broken_sym_component_old_factors)s)
            _pid_list = list(%(broken_sym_pid_list)s)
            _block_start = list(%(broken_sym_block_starts)s)
            _block_len = list(%(broken_sym_block_lengths)s)
            _pid_work = list(_pid_list)
            _total_factor = 1
            for _icomp in range(%(broken_sym_ncomponents)d):
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
        %(info_lines)s
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        %(process_lines)s
        #  
        #  
        # Process parameters
        #  
        ngraphs = %(ngraphs)d
        nexternal = %(nexternal)d
        nwavefuncs = %(nwavefuncs)d
        ncolor = %(ncolor)d
        ZERO = 0.
        #  
        # Color matrix
        #  
        %(color_matrix_lines)s
        #
        # Model parameters
        #
        %(model_parameters)s
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        %(helas_calls)s

        jamp = [None] * ncolor

        %(jamp_lines)s

        %(amp2_lines)s
        matrix = 0.
        for i in range(ncolor):
            ztemp = 0
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   
        self.jamp.append(jamp)

        return matrix
