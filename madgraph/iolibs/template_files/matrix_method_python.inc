def smatrix_%(process_string)s(p, model):
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
    ans = 0.
    for hel in nhel:
        t = matrix(p, hel, model)
        ans = ans + t

    ans = ans / denominator

    return ans
       
def matrix_%(process_string)s(p, hel, model):
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
    amp = []
    w = []
    %(helas_calls)s

    jamp = []
    %(jamp_lines)s
    
    matrix = 0.
    for i in range(ncolor):
        ztemp = 0
        for j in range(ncolor):
            ztemp = ztemp + cf[i][j]*jamp[j]
        matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   

    return matrix
