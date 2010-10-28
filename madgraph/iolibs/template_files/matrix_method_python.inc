class Matrix(object):

    def smatrix(self,p, model):
        #  
        %(info_lines)s
        # 
        # MadGraph StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        %(process_lines)s
        #  
        #  
        # CONSTANTS
        #  
        nexternal = %(nexternal)d
        ncomb = %(ncomb)d
        #  
        # LOCAL VARIABLES 
        #  
        %(helicity_lines)s
        %(den_factor_line)s
        # ----------
        # BEGIN CODE
        # ----------
        self.amp2 = [0] * %(ngraphs)d
        ans = 0.
        for hel in helicities:
            t = self.matrix(p, hel, model)
            ans = ans + t
        ans = ans / denominator
        return ans.real

    def matrix(self, p, hel, model):
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

        self.amp2 = [self.amp2[i] + abs(amp[i]*amp[i].conjugate()) for i in \
                     range(ngraphs)]

        matrix = 0.
        for i in range(ncolor):
            ztemp = 0
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   

        return matrix
