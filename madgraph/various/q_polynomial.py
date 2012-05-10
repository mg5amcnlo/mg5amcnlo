import array
import copy

class PolynomialError(Exception): pass

def get_number_of_coefs_for_rank(r):
    """ Returns the number of independent coefficients there is in a
    fully symmetric tensor of rank r """
    return sum([((3+ri)*(2+ri)*(1+ri))/6 for ri in range(0,r+1)])

class polynomial(object):
    """ A class to represent a polynomial in the loop momentum (4-vector) q"""
    
    def __init__(self, rank):
        if rank<0:
            raise PolynomialError, \
                            "The rank of a q-polynomial should be 0 or positive"
        self.rank=rank
        self.init_coef_list()
        
    def init_coef_list(self):
        """ Creates a list whose elements are arrays being the coefficient
        indices sorted in growing order and the value is their position in a 
        one-dimensional vector. For example the position of the coefficient
        C_01032 will be placed in the list under array.array('i',(0,0,1,3,2)). 
        """
        self.coef_list=[]
        self.coef_list.append(array.array('i',()))
        
        if self.rank==0:
            return
        
        tmp_coef_list=[array.array('i',(0,)),array.array('i',(1,)),
                   array.array('i',(2,)),array.array('i',(3,))]
        self.coef_list.extend(tmp_coef_list)

        for i in range(1,self.rank):
            new_tmp_coef_list=[]
            for coef in tmp_coef_list:
                for val in range(coef[-1],4):
                    new_coef=copy.copy(coef)
                    new_coef.append(val)
                    new_tmp_coef_list.append(new_coef)
            tmp_coef_list=new_tmp_coef_list
            self.coef_list.extend(tmp_coef_list)
    
    def get_coef_position(self, indices_list):
        """ Returns the canonical position for a coefficient characterized 
        by the value of the indices of the loop momentum q it multiplies,
        that is for example C_01032 multiplying q_0*q_1*q_0*q_3*q_2 """

        new_indices_list=copy.copy(indices_list)
        new_indices_list.sort()
        try:
            return self.coef_list.index(array.array('i',new_indices_list))
        except ValueError:
            raise PolynomialError,\
                "The index %s looked for could not be found"%str(indices_list)   
