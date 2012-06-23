import array
import copy

class PolynomialError(Exception): pass

def get_number_of_coefs_for_rank(r):
    """ Returns the number of independent coefficients there is in a
    fully symmetric tensor of rank r """
    return sum([((3+ri)*(2+ri)*(1+ri))/6 for ri in range(0,r+1)])

class Polynomial(object):
    """ A class to represent a polynomial in the loop momentum (4-vector) q"""
    
    def __init__(self, rank):
        
        assert rank > -1, "The rank of a q-polynomial should be 0 or positive"
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

    def get_coef_at_position(self, pos):
        """ Returns the coefficient at position pos in the one dimensional
        vector """
        return list(self.coef_list[pos])

class PolynomialRoutines(object):
    """ The mother class to output the polynomial subroutines """
    
    def __init__(self, max_rank, coef_format='complex*16', sub_prefix='' 
                                                                ,line_split=30):
        self.coef_format=coef_format
        self.sub_prefix=sub_prefix
        if coef_format=='complex*16':
            self.rzero='0.0d0'
            self.czero='(0.0d0,0.0d0)'
        elif coef_format=='complex*32':
            self.rzero='0.0e0_16'
            self.czero='CMPLX(0.0e0_16,0.0e0_16,KIND=16)'            
        else:
            self.rzero='0.0e0'
            self.czero='(0.0e0,0.0e0)'            
        self.line_split=line_split
        if max_rank<0:
            raise PolynomialError, \
                            "The rank of a q-polynomial should be 0 or positive"
        self.max_rank=max_rank
        self.pq=Polynomial(max_rank)

class FortranPolynomialRoutines(PolynomialRoutines):
    """ A daughter class to output the subroutine in the fortran format"""
    
    def write_wl_updater(self,r_1,r_2):
        """ Give out the subroutine to update a polynomial of rank r_1 with
        one of rank r_2 """
        
        # The update is basically given by 
        # OUT(j,coef,i) = A(k,*,i) x B(j,*,k)
        # with k a summed index and the 'x' operation is equivalent to 
        # putting together two regular polynomial in q with scalar coefficients
        # The complexity of this subroutine is therefore 
        # MAXLWFSIZE**3 * NCoef(r_1) * NCoef(r_2)
        # Which is for example 22'400 when updating a rank 4 loop wavefunction
        # with a rank 1 updater.
        
        lines=[]
        
        # Start by writing out the header:
        lines.append(
          """SUBROUTINE %(sub_prefix)sUPDATE_WL_%(r_1)d_%(r_2)d(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
                        include 'coef_specs.inc'
                        INTEGER I,J,K
                        %(coef_format)s A(MAXLWFSIZE,0:LOOP_MAXCOEFS-1,MAXLWFSIZE)
                        %(coef_format)s B(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
                        %(coef_format)s OUT(MAXLWFSIZE,0:LOOP_MAXCOEFS-1,MAXLWFSIZE)
                        INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE
                        """%{'sub_prefix':self.sub_prefix,'r_1':r_1,'r_2':r_2,
                                                'coef_format':self.coef_format})
        
        # Start the loop on the elements i,j of the vector OUT(i,coef,j)
        lines.append("DO I=1,LCUT_SIZE")
        lines.append("  DO J=1,OUT_SIZE")
        lines.append("    DO K=0,%d"%(get_number_of_coefs_for_rank(r_2+r_1)-1))
        lines.append("      OUT(J,K,I)=%s"%self.czero)
        lines.append("    ENDDO")
        lines.append("    DO K=1,IN_SIZE")
        
        # Now we write the lines defining the coefs of OUT(j,*,i) from those
        # of A(k,*,i) and B(j,*,k)
        # The dictionary below stores the position of the new coefficient 
        # derived as keys and the list of the buidling blocks expressing
        # them as values
        coef_expressions={}
        for coef_a in range(0,get_number_of_coefs_for_rank(r_1)):
            for coef_b in range(0,get_number_of_coefs_for_rank(r_2)):
                ind_list=self.pq.get_coef_at_position(coef_a)+\
                         self.pq.get_coef_at_position(coef_b)
                new_term="A(K,%d,I)*B(J,%d,K)"%(coef_a,coef_b)
                new_position=self.pq.get_coef_position(ind_list)
                try:
                    coef_expressions[new_position].append(new_term)
                except KeyError:
                    coef_expressions[new_position]=[new_term,]
        for coef, value in coef_expressions.items():
            split=0
            while split<len(value):
                lines.append("OUT(J,%d,I)=OUT(J,%d,I)+"%(coef,coef)+\
                             '+'.join(value[split:split+self.line_split]))
                split=split+self.line_split
                
        # And now we simply close the enddo.
        lines.append("    ENDDO")
        lines.append("  ENDDO")
        lines.append("ENDDO")
        lines.append("END")

        # return the subroutine
        return '\n'.join(lines)
        
    def write_polynomial_evaluator(self):
        """ Give out the subroutine to evaluate a polynomial of a rank up to
        the maximal one specified when initializing the FortranPolynomialRoutines
        object. """
        lines=[]
        
        # Start by writing out the header:
        lines.append("""SUBROUTINE %(sub_prefix)sEVAL_POLY(C,R,Q,OUT)
                        include 'coef_specs.inc'
                        %(coef_format)s C(0:LOOP_MAXCOEFS-1)
                        INTEGER R
                        %(coef_format)s Q(0:3)
                        %(coef_format)s OUT                                                 
                        """%{'sub_prefix':self.sub_prefix,
                             'coef_format':self.coef_format})
        
        # Start by the trivial coefficient of order 0.
        lines.append("OUT=C(0)")
        # Now scan them all progressively
        for r in range(1,self.max_rank+1):
            lines.append("IF (R.GE.%d) then"%r)
            terms=[]
            for coef_num in range(get_number_of_coefs_for_rank(r-1)
                                              ,get_number_of_coefs_for_rank(r)):
                coef_inds=self.pq.get_coef_at_position(coef_num)
                terms.append('*'.join(['C(%d)'%coef_num,]+
                                            ['Q(%d)'%ind for ind in coef_inds]))
            split=0
            while split<len(terms):
                lines.append("OUT=OUT+"+\
                                   '+'.join(terms[split:split+self.line_split]))
                split=split+self.line_split            
            lines.append("ENDIF")
        lines.append("END")
        
        return '\n'.join(lines)

    def write_wl_merger(self):
        """ Give out the subroutine to merge the components of a final loop
        wavefunction of a loop to create the coefficients of the polynomial
        representing the numerator, while multiplying each of them by 'const'."""
        lines=[]
        
        # Start by writing out the header:
        lines.append("""SUBROUTINE %(sub_prefix)sMERGE_WL(WL,R,LCUT_SIZE,CONST,OUT)
                        include 'coef_specs.inc'
                        INTEGER I,J
                        %(coef_format)s WL(MAXLWFSIZE,0:LOOP_MAXCOEFS-1,MAXLWFSIZE)
                        INTEGER R,LCUT_SIZE
                        %(coef_format)s CONST
                        %(coef_format)s OUT(0:LOOP_MAXCOEFS-1)
                        """%{'sub_prefix':self.sub_prefix,
                             'coef_format':self.coef_format})

        # Add an array specifying how many coefs there are for given ranks
        lines.append("""INTEGER NCOEF_R(0:%(max_rank)d)
                        DATA NCOEF_R/%(ranks)s/
                        """%{'max_rank':self.max_rank,'ranks':','.join([
                            str(get_number_of_coefs_for_rank(r)) for r in
                                                    range(0,self.max_rank+1)])})                     
     
        # Now scan them all progressively
        lines.append("DO I=1,LCUT_SIZE")
        lines.append("  DO J=0,NCOEF_R(R)-1")
        lines.append("    OUT(J)=OUT(J)+WL(I,J,I)*CONST")               
        lines.append("  ENDDO")
        lines.append("ENDDO")
        lines.append("END")
        
        return '\n'.join(lines)       
             
    def write_add_coefs(self):
        """ Give out the subroutine to simply add together the coefficients
        of two loop polynomials of rank R1 and R2 storing the result in the
        first polynomial given in the arguments."""
        lines=[]
        
        # Start by writing out the header:
        lines.append("""SUBROUTINE %(sub_prefix)sADD_COEFS(A,RA,B,RB)
                        include 'coef_specs.inc'
                        INTEGER I
                        %(coef_format)s A(0:LOOP_MAXCOEFS-1),B(0:LOOP_MAXCOEFS-1)
                        INTEGER RA,RB
                        """%{'sub_prefix':self.sub_prefix,
                             'coef_format':self.coef_format})

        # Add an array specifying how many coefs there are for given ranks
        lines.append("""INTEGER NCOEF_R(0:%(max_rank)d)
                        DATA NCOEF_R/%(ranks)s/
                        """%{'max_rank':self.max_rank,'ranks':','.join([
                            str(get_number_of_coefs_for_rank(r)) for r in
                                                    range(0,self.max_rank+1)])})                     
     
        # Now scan them all progressively
        lines.append("DO I=0,NCOEF_R(RB)-1")
        lines.append("  A(I)=A(I)+B(I)")               
        lines.append("ENDDO")
        lines.append("END")
        
        return '\n'.join(lines)