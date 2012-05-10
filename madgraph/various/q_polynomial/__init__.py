def get_number_of_coefs_for_rank(r):
    """ Returns the number of independent coefficients there is in a
    fully symmetric tensor of rank r """
    return sum([((3+ri)*(2+ri)*(1+ri))/6 for ri in range(0,r+1)])