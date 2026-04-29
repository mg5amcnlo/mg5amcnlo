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
        # processes.  For a merged matrix element the denominator includes the
        # identical-particle factorial of the *base* process.  When the actual
        # runtime flavor has a different set of identical final-state particles,
        # a correction factor must be applied, exactly as in the Fortran/C++
        # BROKEN_SYM function:  ans = ans * broken_sym / denominator.
        if flavor is not None:
            _sym_pid = list(%(broken_sym_pid)s)
            _sym_factor = %(broken_sym_old_factor)d
            _ninitial = %(ninitial)d
            for _i in range(len(_sym_pid)):
                if _sym_pid[_i] == 0:
                    continue
                _n_tot = 1
                for _j in range(_i + 1, len(_sym_pid)):
                    if _sym_pid[_i] == _sym_pid[_j] and flavor[_ninitial + _i] == flavor[_ninitial + _j]:
                        _sym_pid[_j] = 0
                        _n_tot += 1
                        _sym_factor = _sym_factor // _n_tot
            ans = ans * _sym_factor / denominator
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
