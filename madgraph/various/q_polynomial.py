import array
import copy
import math

class PolynomialError(Exception): pass

def get_number_of_coefs_for_rank(r):
    """ Returns the number of independent coefficients there is in a
    fully symmetric tensor of rank r """
    return sum([((3+ri)*(2+ri)*(1+ri))/6 for ri in range(0,r+1)])

class Polynomial(object):
    """ A class to represent a polynomial in the loop momentum (4-vector) q 
    and how the symmetrized coefficients are ordered. The ordering rule 
    correspond to what is presented in Eq. C.15 of arxiv:hep-ph/1405.0301"""
        
    def __init__(self, rank):
        
        assert rank > -1, "The rank of a q-polynomial should be 0 or positive"
        self.rank=rank
        self.init_coef_list()
        
    def init_coef_list(self):
        """ Creates a list whose elements are arrays being the coefficient
        indices. We order this list according to the algorithm in  
        get_coef_position. This coef_list can then be used for the function
        get_coef_at_position()
        """
        
        self.coef_list=[None,]*get_number_of_coefs_for_rank(self.rank)
        
        PNO = Polynomial_naive_ordering(self.rank)
        
        for coef in PNO.coef_list:            
            self.coef_list[self.get_coef_position(list(coef))]=coef
        
    def get_coef_position(self, indices_list):
        """ Returns the canonical position for a coefficient characterized 
        by the value of the indices of the loop momentum q it multiplies,
        that is for example C_01032 multiplying q_0*q_1*q_0*q_3*q_2.
        We assume that the explicit construction of the position below is
        faster than a lookup in a table"""

        fact = math.factorial

        if len(indices_list)==0:
            return 0
        
        res = get_number_of_coefs_for_rank(len(indices_list)-1)

        new_indices_list = copy.copy(indices_list)
        new_indices_list.sort()

        for i, ind in enumerate(new_indices_list):
            if ind>0:
                res = res + (fact(ind+i)/(fact(i+1)*fact(ind - 1)))
                
        return res

    def get_coef_at_position(self, pos):
        """ Returns the coefficient at position pos in the one dimensional
        vector """
        return list(self.coef_list[pos])

class Polynomial_naive_ordering(object):
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
    
    def __init__(self, max_rank, updater_max_rank=None, 
                        coef_format='complex*16', sub_prefix='',
                        proc_prefix='',mp_prefix='',
                        line_split=30):

        self.coef_format=coef_format
        self.sub_prefix=sub_prefix
        self.proc_prefix=proc_prefix
        self.mp_prefix=mp_prefix
        if updater_max_rank is None:
            self.updater_max_rank = max_rank
        else:
            if updater_max_rank > max_rank:
                raise PolynomialError, "The updater max rank must be at most"+\
                                                " equal to the overall max rank"
            else:
                self.updater_max_rank = updater_max_rank            
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
        
        # A useful replacement dictionary
        self.rep_dict = {'sub_prefix':self.sub_prefix,
                         'proc_prefix':self.proc_prefix,
                         'mp_prefix':self.mp_prefix,
                         'coef_format':self.coef_format}

class FortranPolynomialRoutines(PolynomialRoutines):
    """ A daughter class to output the subroutine in the fortran format"""
    
    def write_polynomial_constant_module(self):
        """ Writes a fortran90 module that defined polynomial constants objects."""
        
        # Start with the polynomial constants module header
        polynomial_constant_lines = []
        polynomial_constant_lines.append(
"""MODULE %sPOLYNOMIAL_CONSTANTS
implicit none
include 'coef_specs.inc'
include 'loop_max_coefs.inc'
"""%self.sub_prefix)   
        # Add the N coef for rank
        polynomial_constant_lines.append(
'C Map associating a rank to each coefficient position')
        polynomial_constant_lines.append(
                                     'INTEGER COEFTORANK_MAP(0:LOOPMAXCOEFS-1)')
        for rank in range(self.max_rank+1):
            start = get_number_of_coefs_for_rank(rank-1)
            end   = get_number_of_coefs_for_rank(rank)-1
            polynomial_constant_lines.append(
'DATA COEFTORANK_MAP(%(start)d:%(end)d)/%(n_entries)d*%(rank)d/'%
{'start': start,'end': end,'n_entries': end-start+1,'rank': rank})
        
        polynomial_constant_lines.append(
'\nC Map defining the number of coefficients for a symmetric tensor of a given rank')
        polynomial_constant_lines.append(
"""INTEGER NCOEF_R(0:%(max_rank)d)
DATA NCOEF_R/%(ranks)s/"""%{'max_rank':self.max_rank,'ranks':','.join([
      str(get_number_of_coefs_for_rank(r)) for r in range(0,self.max_rank+1)])})
        polynomial_constant_lines.append(
'\nC Map defining the coef position resulting from the multiplication of two lower rank coefs.')
        mult_matrix = [[
          self.pq.get_coef_position(self.pq.get_coef_at_position(coef_a)+
                                    self.pq.get_coef_at_position(coef_b))
            for coef_b in range(0,get_number_of_coefs_for_rank(self.updater_max_rank))]
              for coef_a in range(0,get_number_of_coefs_for_rank(self.max_rank))]

        polynomial_constant_lines.append(
'INTEGER COMB_COEF_POS(0:LOOPMAXCOEFS-1,0:%(max_updater_rank)d)'\
%{'max_updater_rank':(get_number_of_coefs_for_rank(self.updater_max_rank)-1)})

        for j, line in enumerate(mult_matrix):
            chunk_size = 20
            for k in xrange(0, len(line), chunk_size):
                polynomial_constant_lines.append(
                "DATA COMB_COEF_POS(%3r,%3r:%3r) /%s/" % \
                (j, k, min(k + chunk_size, len(line))-1,
                    ','.join(["%3r" % i for i in line[k:k + chunk_size]])))

        polynomial_constant_lines.append(
                            "\nEND MODULE %sPOLYNOMIAL_CONSTANTS\n"%self.sub_prefix)
        
        return '\n'.join(polynomial_constant_lines)
    
    
    def write_pjfry_mapping(self):
        """ Returns a fortran subroutine which fills in the array of integral reduction 
        coefficients following MadLoop standards using pjfry++ coefficients."""
        
        # THE OUTPUT OF COEFS FROM PJFRY++ IS
        # RANK=0: (,)
        # RANK=1: (0,),(1,),(2,),(3,)
        # RANK=2: (0,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)
        # ...
        # THE OUTPUT OF COEFS FROM MADLOOP IS
        # RANK=0: (,)
        # RANK=1: (0,),(1,),(2,),(3,)
        # RANK=2: (0,0),(0,1),(0,2),(0,3),(1,1),(2,1),(3,1),(2,2),(2,3),(3,3)
        # ...
        
        
        # Helper function
        def format_power(pow):
            b, e = pow

            if e == 1:
                return str(b)
            else:
                return "%s^%d" % (b, e)
            
            
        def get_coef_position(indices_list):
            new_indices_list=copy.copy(indices_list)
            new_indices_list.sort()
            r=len(new_indices_list)
            if r == 0:
                pos=0
            else:
                pos=get_number_of_coefs_for_rank(r-1)
            for i,mu in enumerate(new_indices_list):
                num = mu
                den = 1
                if mu > 0 and i > 0:
                    for j in range(2,i+2):
                        num *= (mu+j-1)
                        den *= j
                pos += num/den
            return pos
        
        lines = []
        lines.append(
                """SUBROUTINE %(sub_prefix)sCONVERT_PJFRY_COEFFS(RANK,PJCOEFS,TIRCOEFS)
C      GLOABLE VARIABLES
                include 'coef_specs.inc'
                include 'loop_max_coefs.inc'
C      ARGUMENTS
                INTEGER RANK
                %(coef_format)s PJCOEFS(0:LOOPMAXCOEFS-1,3)
                %(coef_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)"""
                %{'sub_prefix':self.sub_prefix,'coef_format':self.coef_format})
        
        for R in range(self.max_rank+1):
            Ncoeff=((3+R)*(2+R)*(1+R))/6
            if R == 0:
                offset=0
            else:
                offset=get_number_of_coefs_for_rank(R-1)
            for i in range(offset,Ncoeff+offset):
                indices_list=self.pq.get_coef_at_position(i)
                sindices = map(lambda i: "q(%d)" % i, indices_list)
                coeff_list = []
                for j in range(4):
                    qvalue = "q(%d)"%j
                    qpow = sindices.count(qvalue)
                    if qpow > 0:
                        coeff_list.append(format_power([qvalue,qpow]))
                                        
                if not coeff_list:
                    coeff_str = "1"
                else:
                    coeff_str = "*".join(coeff_list)

                pjpos = get_coef_position(indices_list)
                lines.append("c Reduction Coefficient %s"%coeff_str)
                lines.append('TIRCOEFS(%d,1:3)=PJCOEFS(%d,1:3)'%(i,pjpos))
            lines.append('IF(RANK.LE.%d)RETURN'%R)

        lines.append('end')
        
        return '\n'.join(lines)
    
    def write_iregi_mapping(self):
        """ Returns a fortran subroutine which fills in the array of integral reduction 
        coefficients following MadLoop standards using IREGI coefficients."""
        
        # THE OUTPUT OF COEFS FROM IREGI IS
        # RANK=0: (,)
        # RANK=1: (0,),(1,),(2,),(3,)
        # RANK=2: (0,0),(0,1),(0,2),(0,3),(1,1),(2,1),(3,1),(2,2),(2,3),(3,3)
        # ...
                
        # Helper function
        def format_power(pow):
            b, e = pow

            if e == 1:
                return str(b)
            else:
                return "%s^%d" % (b, e)
            
        lines = []
        lines.append(
                """SUBROUTINE %(sub_prefix)sCONVERT_IREGI_COEFFS(RANK,IREGICOEFS,TIRCOEFS)
C        GLOABLE VARIABLES
                include 'coef_specs.inc'
                include 'loop_max_coefs.inc'
C        ARGUMENTS
                INTEGER RANK
                %(coef_format)s IREGICOEFS(0:LOOPMAXCOEFS-1,3)
                %(coef_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)"""
                %{'sub_prefix':self.sub_prefix,'coef_format':self.coef_format})
        
        iregi_gen = FromIREGIFortranCodeGenerator(self.max_rank)
        for R in range(self.max_rank+1):
            Ncoeff=((3+R)*(2+R)*(1+R))/6
            if R == 0:
                offset=0
            else:
                offset=get_number_of_coefs_for_rank(R-1)
            for i in range(offset,Ncoeff+offset):
                indices_list=self.pq.get_coef_at_position(i)
                sindices = map(lambda i: "q(%d)" % i, indices_list)
                coeff_list = []
                for j in range(4):
                    qvalue = "q(%d)"%j
                    qpow = sindices.count(qvalue)
                    if qpow > 0:
                        coeff_list.append(format_power([qvalue,qpow]))
                                        
                if not coeff_list:
                    coeff_str = "1"
                else:
                    coeff_str = "*".join(coeff_list)

                iregipos = iregi_gen.get_coef_position(indices_list)
                lines.append("c Reduction Coefficient %s"%coeff_str)
                lines.append('TIRCOEFS(%d,1:3)=IREGICOEFS(%d,1:3)'%(i,iregipos))
            lines.append('IF(RANK.LE.%d)RETURN'%R)
        lines.append('end')
        
        return '\n'.join(lines)
    
    def write_golem95_mapping(self):
        """ Returns a fortran subroutine which fills in the array of tensorial
        coefficients following golem95 standards using MadLoop coefficients."""

        subroutines = []
        
        # Set number of space-time dimensions to 4 here
        d = 4
        golem_max_rank = 6

        # First generate the block_info which contains information about the
        # about the block structure of the system
        block_info = {}
        for R in range(1,self.max_rank+1):
            for k in range(1,min(R,d)+1):
                LHS, RHS, lst, dic = \
                        FromGolem95FortranCodeGenerator.generate_equations(R, k)
                block_info[(R,k)] = (lst, dic)
     
        # Helper function
        def format_power(pow):
            b, e = pow

            if e == 1:
                return str(b)
            else:
                return "%s^%d" % (b, e)
                
        # Write out one subroutine per rank
        for R in range(golem_max_rank+1):
            
            lines=[]
            
            if R==0:
                lines.append(
                """SUBROUTINE %(sub_prefix)sFILL_GOLEM_COEFFS_0(ML_COEFS,GOLEM_COEFS)
                            use precision_golem, only: ki
                            include 'coef_specs.inc'
                            include 'loop_max_coefs.inc'
                            %(coef_format)s ML_COEFS(0:LOOPMAXCOEFS-1)
                            complex(ki) GOLEM_COEFS"""
                %{'sub_prefix':self.sub_prefix,'coef_format':self.coef_format})
                lines.append("GOLEM_COEFS=ML_COEFS(0)")
                lines.append("end")
                subroutines.append('\n'.join(lines))
                continue
            
            # Start by writing out the header:
            lines.append(
              """SUBROUTINE %(sub_prefix)sFILL_GOLEM_COEFFS_%(rank)d(ML_COEFS,GOLEM_COEFS)
                            use tens_rec, only: coeff_type_%(rank)d
                            include 'coef_specs.inc'
                            include 'loop_max_coefs.inc'
                            %(coef_format)s ML_COEFS(0:LOOPMAXCOEFS-1)
                            type(coeff_type_%(rank)d) GOLEM_COEFS"""
                            %{'sub_prefix':self.sub_prefix,'rank':R,
                                                'coef_format':self.coef_format})

            if R > self.max_rank:
                lines.append('C Dummy routine for %(sub_prefix)sFILL_GOLEM_COEFS_%(rank)d'\
                             %{'sub_prefix':self.sub_prefix,'rank':R,
                                                'coef_format':self.coef_format})
                lines.append("STOP 'ERROR: %d > %d'"%(R,self.max_rank))
                lines.append('end')
                subroutines.append('\n'.join(lines))
                continue
                
            # The constant coefficient is treated separately
            lines.append("c Constant coefficient ")
            lines.append("GOLEM_COEFS%%c0=ML_COEFS(%d)"\
                                                 %self.pq.get_coef_position([]))            

            # Now write out the explicit mapping
            for k in range(1,min(R,d)+1):
                lst, dic = block_info[(R,k)]
                dim = len(lst)
                lab = 0
                for indices in FromGolem95FortranCodeGenerator.select(range(d), k):
                    lab += 1
                    sindices = map(lambda i: "q(%d)" % i, indices)
                    for i in range(dim):
                        coeff_str = "*".join(map(format_power,zip(sindices, lst[i])))
                        ML_indices = sum(
                          [[ind]*lst[i][j] for j, ind in enumerate(indices)],[])
                        ML_coef_pos = self.pq.get_coef_position(ML_indices)
                        ML_sign_convention = ' ' if len(ML_indices)%2==0 else '-'
                        lines.append("c Coefficient %s"%coeff_str)
                        lines.append("GOLEM_COEFS%%c%d(%d,%d)=%sML_COEFS(%d)"\
                               % (k, lab, i+1, ML_sign_convention, ML_coef_pos))
            
            subroutines.append('\n'.join(lines+['end']))
            
        return '\n\n'.join(subroutines)

    def write_compact_wl_updater(self,r_1,r_2,loop_over_vertex_coefs_first=True):
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
        # The situation is slightly improved by a smarter handling of the 
        # coefficients equal to zero
        
        lines=[]
        
        # Start by writing out the header:
        lines.append(
          """SUBROUTINE %(sub_prefix)sUPDATE_WL_%(r_1)d_%(r_2)d(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
  USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS      
  implicit none
  INTEGER I,J,K,L,M
  %(coef_format)s A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
  %(coef_format)s B(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
  %(coef_format)s OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
  INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE
  INTEGER NEW_POSITION
  %(coef_format)s UPDATER_COEF
"""%{'sub_prefix':self.sub_prefix,'proc_prefix':self.proc_prefix,
                           'r_1':r_1,'r_2':r_2,'coef_format':self.coef_format})
        
        # Start the loop on the elements i,j of the vector OUT(i,coef,j)
        lines.append("C Welcome to the computational heart of MadLoop...")
        if loop_over_vertex_coefs_first:
            lines.append("OUT(:,:,:)=%s"%self.czero)
            lines.append(
    """DO J=1,OUT_SIZE
      DO M=0,%d
        DO K=1,IN_SIZE
          UPDATER_COEF = B(J,M,K)
          IF (UPDATER_COEF.EQ.%s) CYCLE
          DO L=0,%d
            NEW_POSITION = COMB_COEF_POS(L,M)
            DO I=1,LCUT_SIZE
              OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + A(K,L,I)*UPDATER_COEF
            ENDDO
          ENDDO
        ENDDO
      ENDDO
    ENDDO
    """%(get_number_of_coefs_for_rank(r_2)-1,
         self.czero,
         get_number_of_coefs_for_rank(r_1)-1))
        else:
            lines.append("OUT(:,:,:)=%s"%self.czero)
            lines.append(
    """DO I=1,LCUT_SIZE
      DO L=0,%d
        DO K=1,IN_SIZE
          UPDATER_COEF = A(K,L,I)
          IF (UPDATER_COEF.EQ.%s) CYCLE
          DO M=0,%d
            NEW_POSITION = COMB_COEF_POS(L,M)
            DO J=1,OUT_SIZE
              OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + UPDATER_COEF*B(J,M,K)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
    ENDDO
    """%(get_number_of_coefs_for_rank(r_1)-1,
         self.czero,
         get_number_of_coefs_for_rank(r_2)-1))            
        
        lines.append("END")
        # return the subroutine
        return '\n'.join(lines)

    def write_expanded_wl_updater(self,r_1,r_2):
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
  USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
  INTEGER I,J,K
  %(coef_format)s A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
  %(coef_format)s B(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
  %(coef_format)s OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
  INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE
"""%{'sub_prefix':self.sub_prefix,'proc_prefix':self.proc_prefix,
                            'r_1':r_1,'r_2':r_2,'coef_format':self.coef_format})
        
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
                        USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS      
                        %(coef_format)s C(0:LOOPMAXCOEFS-1)
                        INTEGER R
                        %(coef_format)s Q(0:3)
                        %(coef_format)s OUT                                                 
                        """%self.rep_dict)
        
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
        lines.append(
"""SUBROUTINE %(sub_prefix)sMERGE_WL(WL,R,LCUT_SIZE,CONST,OUT)
  USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS      
  INTEGER I,J
  %(coef_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
  INTEGER R,LCUT_SIZE
  %(coef_format)s CONST
  %(coef_format)s OUT(0:LOOPMAXCOEFS-1)
"""%self.rep_dict)                    
     
        # Now scan them all progressively
        lines.append("DO I=1,LCUT_SIZE")
        lines.append("  DO J=0,NCOEF_R(R)-1")
        lines.append("      OUT(J)=OUT(J)+WL(I,J,I)*CONST")               
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
                        USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS      
                        INTEGER I
                        %(coef_format)s A(0:LOOPMAXCOEFS-1),B(0:LOOPMAXCOEFS-1)
                        INTEGER RA,RB
                        """%self.rep_dict) 

        # Now scan them all progressively
        lines.append("DO I=0,NCOEF_R(RB)-1")
        lines.append("  A(I)=A(I)+B(I)")               
        lines.append("ENDDO")
        lines.append("END")
        
        return '\n'.join(lines)
    
class FromIREGIFortranCodeGenerator():
    """ Back up of the class Polynomial, which uses the same coefficeints orders with IREGI.
    It is useful in the case that the order of MadLoop coefficients changes in the future."""
    
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


class FromGolem95FortranCodeGenerator():
    """ Just a container class with helper functions taken from the script 
    tens.py of golem which generates most of the golem95 tens_rec.f fortran
    code."""
    
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
       31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
       73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
       127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
       179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
       233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
       283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
       353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
       419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
       467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
       547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
       607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
       661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
       739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
       811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
       877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
       947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
       1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
       1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151,
       1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
       1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
       1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373]

    @classmethod    
    def combinat(cls, n, k):
        """
            Calculates the binomial coefficient (n atop k).
        """
        if k < 0 or k > n:
            return 0
        else:
            num = 1
            den = 1
            for i in range(1, k+1):
                num *= n-i+1
                den *= i
            return num/den

    @classmethod
    def generate_mapping(cls, R, k):
        """
            Generates a mapping from tensor components \hat{C}(a_1, ..., a_k)
            into a one dimensional array.
    
            PARAMETER
    
            R  -- rank
            k  -- number of non-zero components of q
    
            RETURN
    
            (lst, dic)
    
            lst -- list of (a_1, ..., a_k)
            dic -- mapping from (a_1, ..., a_k) -> int
    
            lst[dic[X]] = X if X in dic
        """
    
        def rec_generator(k, R):
            if k == 0:
                yield []
            elif k <= R:
                for a_1 in range(1, R - (k - 1) + 1):
                    if k > 1:
                        for tail in rec_generator(k - 1, R - a_1):
                            yield [a_1] + tail
                    else:
                        yield [a_1]
        
        lst = []
        dic = {}
        i = 0
        for indices in rec_generator(k, R):
            t = tuple(indices)
            lst.append(t)
            dic[t] = i
            i += 1
    
        assert i == cls.combinat(R, k), \
                "len(%s) != %d, R=%d,k=%d" % (lst,cls.combinat(R, k),R,k)
        return lst, dic

    @classmethod
    def generate_equations(cls, R, k):
        """
            Generates a set of equations for a given number of non-zero
            components and fixed maximum rank.
        
            PARAMETER
    
            R  -- rank
            k  -- number of non-zero components of q
    
            RETURN
    
            (LHS, RHS)
    
            LHS -- a matrix (i.e. list of lists) of coefficients
            RHS -- a list of values of q
        """
    
        lst, dic = cls.generate_mapping(R, k)
        l = len(lst)
        LHS = []
        RHS = []
        for num_eq in range(l):
            q = map(lambda i: cls.PRIMES[i], lst[num_eq])
            coeffs = [
                reduce(lambda x,y: x*y, map(lambda (b,e): b**e, zip(q, term)), 1)
                for term in lst]
            LHS.append(coeffs)
            RHS.append(q)
    
        return LHS, RHS, lst, dic

    @classmethod
    def select(cls, items, k):
        """
        Iterator over all selections of k elements from a given list.
    
        PARAMETER
    
        items  --  list of elements to choose from (no repetitions)
        k      --  number of elements to select.
        """
        n = len(items)
        # We use the fact that
        # (n choose k) = (1 choose 1)(n-1 choose k-1)+(1 choose 0)(n-1 choose k)
        if k == n:
            yield items[:]
        elif k == 0:
            yield []
        elif 0 < k and k < n:
            head = items[0:1]
            tail = items[1:]
            for result in cls.select(tail, k-1):
                yield head + result
            for result in cls.select(tail, k):
                yield result
                
if __name__ == '__main__':
    """I test here the write_golem95_mapping function"""
    
    P=Polynomial(7)
    print "Coef (6,0,0,0) is at pos %s"%P.get_coef_position([0,0,0,0,0,0])
    print "Coef (1,1,2,2) is at pos %s"%P.get_coef_position([0,1,2,2,3,3])
    print "Coef (7,0,0,0) is at pos %s"%P.get_coef_position([0,0,0,0,0,0,0])
    print "Coef (1,2,2,2) is at pos %s"%P.get_coef_position([0,1,1,2,2,3,3])
    
    sys.exit(0)

    max_rank=6
    FPR=FortranPolynomialRoutines(max_rank)
    print "Output of write_golem95_mapping function for max_rank=%d:\n\n"%max_rank

    import os
    import sys
    root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
    sys.path.insert(0, os.path.join(root_path,os.path.pardir))
    import madgraph.iolibs.file_writers as writers
    FWriter = writers.FortranWriter("GOLEM95_interface.f")
    FWriter.writelines(FPR.write_golem95_mapping())
    
