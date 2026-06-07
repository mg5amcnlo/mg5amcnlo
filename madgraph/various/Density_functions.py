import numpy as np
from numpy import linalg as la
from typing import Union

#%% Global variables for fermions and bosons
Identity2 = np.array([[1, 0], [0, 1]])
sigma1 = np.array([[0, 1], [1, 0]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma3 = np.array([[1, 0], [0, -1]])
sigma = [sigma1, sigma2, sigma3]

Identity3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Lambda2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
Lambda3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
Lambda4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
Lambda5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
Lambda6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Lambda7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
Lambda8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])/np.sqrt(3)
Lambda = [Lambda1, Lambda2, Lambda3, Lambda4, Lambda5, Lambda6, Lambda7, Lambda8]

# Momenta operations
def opp_momentum(p: list[float]) -> list[float]:
    """Returns the 4-momenta (E, -p)"""
    return [p[0], -p[1], -p[2], -p[3]]

def norm_momentum(p: list[float], epsilon=1e-10) -> float:
    """Returns the norm of the 4-momentum p"""
    norm2 = p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
    if abs(norm2) < epsilon:
        return 0.
    else:
        return np.sqrt(norm2)

def invert_momenta(p:list[float]) ->list[float]:
    """
    fortran/C-python do not order table in the same order, one should use this function on any momentum
    from fortran that is used in python functions
    """
    new_p = []
    for i in range(len(p[0])):  new_p.append([0]*len(p))
    for i, onep in enumerate(p):
        for j, x in enumerate(onep):
            new_p[j][i] = x
    return new_p

# Distances betweeen 2 matrices
def trace_distance(Matrix1: list[complex, complex], Matrix2: list[complex, complex]) -> float:
    """
       Input: Matrix1, Matrix2 -> 2 square matrices
       Output: trace distance between the two matrices
    """
    aux1 = Matrix1 - Matrix2
    aux2 = np.dot(np.conjugate(np.transpose(aux1)), aux1)
    eigvals, eigvecs = la.eigh(np.array(aux2))
    for o in range(len(eigvals)): #This is necessary because numerical errors can cause a almost zero eigenvalue to be negative.
                if eigvals[o]/max(eigvals) < 1e-20:
                        eigvals[o] = 0. 
    
    sqrt_rho = np.dot(eigvecs, np.dot(np.sqrt(np.diag(eigvals)), la.inv(eigvecs)))

    return (np.trace(sqrt_rho).real)/2. #we only keep the real part

def Fidelity(rho1: list[complex, complex], rho2: list[complex, complex]) -> float:
    """
       Input: rho1, rho2 -> density matrices
       Output: fidelity of the two density matrices
    """
    eigvals1, eigvecs1 = la.eigh(np.array(rho1))
    sqrt_rho1 = np.dot(eigvecs1, np.dot(np.sqrt(np.diag(eigvals1)), la.inv(eigvecs1)))
    aux = np.dot(sqrt_rho1, np.dot(rho2, sqrt_rho1)) #F = Tr[sqrt(aux)]
    eigvals2, _ = la.eigh(np.array(aux))

    return np.sum(np.sqrt(eigvals2)) #the trace is the sum of the eigenvalues

    
def Fidelity_distance(Matrix1: list[complex, complex], Matrix2: list[complex, complex]) -> float:
    """
       Input: Matrix1, Matrix2 -> 2 square matrices
       Output: fidelity distance between the two matrices
    """
    fidelity = Fidelity(Matrix1, Matrix2)
    return np.sqrt(1 - fidelity**2)

# Misc functions
def plot_hist(x:list[float], y:list[float], z:list[float], limitx:list[float], limity:list[float], n_binx:int, n_biny:int) -> tuple[list[float, float], list[float, float]]:
    """
    Input: x, y, z -> data
           limitx, limity -> bounds for the x and y-axis
           n_binx, n_biny -> number of bins for the x and y-axis
    Output: matrix of the 2D histogram, matrix of the number of events

    x, y, z must be vectors of same length
    The structure of the two matrices are : origin lower left, growing right and up
    """
    binsx = np.linspace(limitx[0], limitx[1], n_binx + 1)
    binsy = np.linspace(limity[0], limity[1], n_biny + 1)
    if isinstance(z[0], float) or isinstance(z[0], int) or isinstance(z[0], complex):
        Map = np.zeros((n_biny, n_binx))
    else: #if the object is a density matrix
        Map = np.zeros((n_biny, n_binx), dtype=object)
        
    N_Map = np.zeros((n_biny, n_binx))
    for k in range(len(z)):
        for i in range(len(binsx) - 1): #we need the -1 because we added +1 when defining binsx
            if x[k] >= binsx[i] and x[k] < binsx[i + 1]: #this means that the second index of Map is i
                for j in range(len(binsy) - 1):
                    if y[k] >= binsy[j] and y[k] < binsy[j + 1]: #this means that the first index of Map is j
                        Map[j][i] += z[k] #if x and y are in the correct bin, we add the value of z in the bin.
                        N_Map[j][i] += 1

    Number_Events = len(z)
    for i in range(len(Map)):
        for j in range(len(Map[0])):
            if N_Map[i][j] == 0:
                Map[i][j] = np.nan
            else:
                Map[i][j] = Map[i][j]/ N_Map[i][j]
                N_Map[i][j] = N_Map[i][j]/Number_Events

    return Map, N_Map


class DensityMatrixObservables(list):
    """
        This class represents a density matrix of any dimension.
        It can be called like: density_instance = DensityMatrixObservables(density_input) where density_input is a density matrix.
        Different methods are defined to allow generic operations on density matrices (trace, transpose, partial transpose, etc.)
        Different methods are also defined to calculate the Quantum Information observables (purity, concurrence, etc.)
        A class is defined for qubit-qubit systems, as well as qubit-qutrit and qutrit-qutrit systems. 
        In these classes we defined the observables specific to these systems like concurrence.
        This class redirects to the specific classes based on the user input. One should always call this generic class.
    """
    def __new__(cls, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None):
        """ 
        This method determines the dimension of the user input and decides which subclass to instanciate.
        For now, there are classes for density matrices of 2x2, 2x3 and 3x3 systems.
        """
        if len_user_input is None: #we add the optional information of the length of the line matrix to use for Fortran input
            len_user_input = len(user_input)

        if isinstance(user_input, list) or isinstance(user_input, np.ndarray):
            if isinstance(user_input[0], list) or isinstance(user_input[0], np.ndarray):
                dim_density_matrix = len(user_input) #if the density matrix is already square form, we let it like that
            else:                
                if isinstance(user_input, list):
                    for i in reversed(range(len(user_input))):
                        if i > len_user_input-1:
                            user_input.pop(i)
                        else:
                            break
                else:
                    for i in reversed(range(len(user_input))):
                        if i > len_user_input-1:
                            user_input = np.delete(user_input, i)
                        else:
                            break

                dim_density_matrix = (-1 + np.sqrt(1 + 8 * len(user_input)))/2
                if dim_density_matrix%1 == 0:
                    dim_density_matrix = int(dim_density_matrix)
                else:
                    raise ValueError('The dimension of the list/array used does not correspond to a square matrix')
                
        elif isinstance(user_input, str):
            temp = user_input.strip('<>density/[]').split()
            user_input = [complex(temp[o].strip(',()')) for o in range(len(temp))]
            for i in reversed(range(len(user_input))):
                if i > len_user_input-1:
                    user_input.pop(i)
                else:
                    break
            dim_density_matrix = (-1 + np.sqrt(1 + 8 * len(user_input)))/2
        else:
            raise TypeError("The type of the input of DensityMatrixObservables is not recognised")

        # According to the dimension of the density matrix given, we instanciate the specific class
        # If none corresponds, we stay in the generic class
        if dim_density_matrix == 4:
            return super(DensityMatrixObservables, cls).__new__(DensityMatrixObservables22, user_input, len_user_input)
        elif dim_density_matrix == 6:
            return super(DensityMatrixObservables, cls).__new__(DensityMatrixObservables23, user_input, len_user_input)
        elif dim_density_matrix == 9:
            return super(DensityMatrixObservables, cls).__new__(DensityMatrixObservables33, user_input, len_user_input)
        else:
            return super(DensityMatrixObservables, cls).__new__(DensityMatrixObservables, user_input, len_user_input)
        
              
    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """ 
            This method stores the density matrix and its dimension in a form that depends on the input of the user.
            If the input comes from Fortran, as a line matrix, we store it in line matrix form.
            If the input is directly a square matrix, we store it as is.
            The different functions of the class will adapt to the type the density matrix is stored in.
        """
        self.density_matrix = []
        self.density_matrix_dimension = 0

        if self.square_or_line_or_string(user_input) == 'square': #if the input is a square matrix, we store it as a matrix
            self.density_matrix_dimension = len(user_input)
            self.density_matrix = user_input

        elif self.square_or_line_or_string(user_input) == 'line': #if the input is a line matrix, we store it as a list
            if len_user_input:
                if isinstance(user_input, list):
                    for i in reversed(range(len(user_input))):
                        if i > len_user_input-1:
                            user_input.pop(i)
                        else:
                            break
                else:
                    for i in reversed(range(len(user_input))):
                        if i > len_user_input-1:
                            user_input = np.delete(user_input, i)
                        else:
                            break
            dim_density_matrix = (-1 + np.sqrt(1 + 8 * len(user_input)))/2
            if dim_density_matrix%1 == 0:
                self.density_matrix = [complex(elem) for elem in user_input]
                self.density_matrix_dimension = int(dim_density_matrix)
            else:
                raise ValueError("The dimension of the line density matrix is not correct")
            
        elif self.square_or_line_or_string(user_input) == 'string': #if the input is a string, we store it as a list
            temp = user_input.strip('<>density/[]').split()
            rho_temp = [complex(temp[o].strip(',()')) for o in range(len(temp))]
            for i in reversed(range(len(rho_temp))):
                if i > len_user_input-1:
                    rho_temp.pop(i)
                else:
                    break
            dim_density_matrix = (-1 + np.sqrt(1 + 8 * len(rho_temp)))/2
            if dim_density_matrix%1 == 0:
                self.density_matrix = rho_temp
                self.density_matrix_dimension = int(dim_density_matrix)
        
        if self.square_or_line_or_string(user_input) == 'square': #if the input is a square matrix, we store it as a matrix
            self.density_matrix_dimension = len(user_input)
            self.density_matrix = user_input

        elif self.square_or_line_or_string(user_input) == 'line': #if the input is a line matrix, we store it as a list
            dim_density_matrix = (-1 + np.sqrt(1 + 8 * len(user_input)))/2
            if dim_density_matrix%1 == 0:
                self.density_matrix = [complex(elem) for elem in user_input]
                self.density_matrix_dimension = int(dim_density_matrix)
            else:
                raise ValueError("The dimension of the line density matrix is not correct")
            
        elif self.square_or_line_or_string(user_input) == 'string': #if the input is a string, we store it as a list
            temp = user_input.strip('<>density/[]').split()
            rho_temp = [complex(temp[o].strip(',()')) for o in range(len(temp))]
            dim_density_matrix = (-1 + np.sqrt(1 + 8 * len(rho_temp)))/2
            if dim_density_matrix%1 == 0:
                self.density_matrix = rho_temp
                self.density_matrix_dimension = int(dim_density_matrix)

    def is_hermitian(self):
        Matrix = self.square_matrix()
        if np.allclose(Matrix, Matrix.conj().T):
            return True
        else:
            return False

    def square_or_line_or_string(self, user_input=None) -> str:
        """This method determines if user_input is a line matrix, a square matrix, a string or something else."""
        if user_input is None: #this allows to use it without argument after instancing the class
            user_input = self.density_matrix
        
        if isinstance(user_input, list) or isinstance(user_input, np.ndarray):
            if isinstance(user_input[0], list) or isinstance(user_input[0], np.ndarray):
                return 'square'
            else:
                return 'line'
        elif isinstance(user_input, str):
            return 'string'
        else:
            raise TypeError("The type of density matrix was not recognised. The density matrix can be a line matrix, a square matrix or a string.")

    def square_matrix(self) -> list[complex, complex]:
        """
        This function converts a hermitian matrix expressed as a line into a typical matrix form.
        Input: hermitian matrix in a line form or matrix form
        Output: hermitian matrix in the traditional square form
        Caution! The definition of matrices in fortran and python are not the same.
        """
        if self.square_or_line_or_string() == 'square':
            return self.density_matrix
        else:
            length_lin = len(self.density_matrix)
            n = (-1 + np.sqrt(1 + 8*length_lin))/2
            if n%1 != 0:
                    raise ValueError("Error :The line matrix given has incorrect length")
            else:
                    n = int(n)

            matrix_square = np.zeros((n, n), dtype=np.complex128)

            for i in range(n):
                    for k in range(n):
                            if k > i:
                                    matrix_square[i][k] = np.conjugate(self.density_matrix[i*n + k - i*(i + 1)//2]) 
                            elif k == i:
                                    matrix_square[i][k] = self.density_matrix[i*n + k - i*(i + 1)//2] #this line is just here because the diagonal elements should not be conjugated (they are real anyway)
                            else:
                                    matrix_square[i][k] = np.conjugate(matrix_square[k][i])
                                                                    
            return matrix_square

    def get_trace(self) -> float:
        """ Calculates the trace of the density matrix (line or matrix form)."""
        if self.square_or_line_or_string() == 'square':
            return np.trace(self.density_matrix)
        else:
            n_comb = self.density_matrix_dimension
            diagIndices = [0] * n_comb
            counter = 0
            for idiag in range(n_comb):
                diagIndices[idiag] = idiag * n_comb - counter
                counter += idiag
            trace = 0
            for i in range(n_comb):
                trace += np.real(self.density_matrix[diagIndices[i]])

            return trace

    def get_rho_normalised(self, epsilon=1e-10) -> Union[list[complex], list[complex, complex]]:
        """ Computes the trace of a matrix and normalises its elements by it."""
        trace = self.get_trace()

        if self.square_or_line_or_string() == 'line':
            for i in range(len(self.density_matrix)):
                    if abs(self.density_matrix[i].real / trace) < epsilon:
                            self.density_matrix[i] = 0. + self.density_matrix[i].imag *1j
                    if abs(self.density_matrix[i].imag / trace) < epsilon:
                            self.density_matrix[i] = self.density_matrix[i].real + 0. *1j

            aux = np.array(self.density_matrix) / trace
            return aux
        
        elif self.square_or_line_or_string() == 'square':
            for i in range(self.density_matrix_dimension):
                for j in range(self.density_matrix_dimension):
                    if abs(self.density_matrix[i][j].real / trace) < epsilon:
                            self.density_matrix[i][j] = 0. + self.density_matrix[i][j].imag *1j
                    if abs(self.density_matrix[i][j].imag / trace) < epsilon:
                            self.density_matrix[i][j] = self.density_matrix[i][j].real + 0. *1j
            
            aux = np.array(self.density_matrix) / trace
            return aux

    def Get_Purity(self) -> float:
        """ Computes the purity of a density matrix of any dimention."""
        rho = self.square_matrix()
        rho2 = np.dot(rho, rho)
        Purity = np.trace(rho2)
        return Purity.real
        
    def Get_Normalised_Purity(self) -> float:
        """ Different choice of normalisation for the purity."""
        d = self.density_matrix_dimension
        return (self.Get_Purity() - 1/d) * (d / (d - 1))

    def Shannon_Entropy(self, x:float) -> float:
        """ Computes the Shannon entropy of x."""
        return - x * np.log2(x) - (1 - x) * np.log2(1 - x)
    
    def Von_Neumann_entropy(self, rho=None, epsilon=1e-5) -> float:
        "The Von Neumann entropy of a density matrix can be seen as the Shannon entropy of its eigenvalues"
        if rho is None:
            rho = self.square_matrix()
        eigvals, _ = la.eigh(rho) #eigenvalues of the density matrix
        entropy = 0.
        for eig in eigvals:
            if abs(eig) < epsilon or eig < 0: #if the eigenvalue is zero (or negative because of numerical errors), we do not count it
                continue
            entropy += -eig * np.log2(eig)

        return entropy.real
    
    def Partial_Transpose(self, index:int, particle_type:list[str]) ->list[complex, complex]:
        """
            Input:  rho -> density matrix
                    index -> integer designating the subspace to transpose (1 or 2)
                    particle_type -> list of 2 elements, the type of both particles in the density matrix ('fermion' or 'boson')
            Output: the density matrix partially transposed
        """

        rho = self.square_matrix()
        
        if len(particle_type) != 2:
            raise ValueError("The particle_type input must be of length 2.")

        #Basis1 is the basis of the dimension of the first particle, Basis2 is the same for the second particle
        if particle_type[0] == 'fermion':
            n1 = 2
        elif particle_type[0] == 'boson':
            n1 = 3
        else:
            raise ValueError("The type of the first particle is not correct (either designs a scalar particle or a non recognised particle). Please change it.")

        if particle_type[1] == 'fermion':
            n2 = 2
        elif particle_type[1] == 'boson':
            n2 = 3
        else:
            raise ValueError("The type of the second particle is not correct (either designs a scalar particle or a non recognised particle). Please change it.")

        Basis1 = np.identity(n1)
        Basis2 = np.identity(n2)

        # Reference : rho_line.append(np.dot(np.kron(Basis3[i], Basis3[j]), np.dot(rho, np.kron(Basis3[k], Basis3[l]))))
        rho_line = []

        if index == 0: #no change
            for i in range(len(Basis1)):
                for j in range(len(Basis2)):
                    for k in range(len(Basis1)):
                        for l in range(len(Basis2)):
                            rho_line.append(np.dot(np.kron(Basis1[i], Basis2[j]), np.dot(rho, np.kron(Basis1[k], Basis2[l]))))  # exchanging j and l is transpose over A
                                                                                                                                # exchanging i and k is transpose over B
        elif index == 1: #transpose over first subspace
            for i in range(len(Basis1)):
                for j in range(len(Basis2)):
                    for k in range(len(Basis1)):
                        for l in range(len(Basis2)):
                            rho_line.append(np.dot(np.kron(Basis1[i], Basis2[l]), np.dot(rho, np.kron(Basis1[k], Basis2[j]))))
        elif index == 2:
            for i in range(len(Basis1)):
                for j in range(len(Basis2)):
                    for k in range(len(Basis1)):
                        for l in range(len(Basis2)):
                            rho_line.append(np.dot(np.kron(Basis1[k], Basis2[j]), np.dot(rho, np.kron(Basis1[i], Basis2[l]))))
        else:
            raise ValueError("The value of index is wrong, please use 1 to transpose over the first subspace or 2 to transpose over the second.")                   

        return np.array(rho_line).reshape(n1*n2,n1*n2)

    def Partial_Trace(self, index:int, particle_type:list[str]) -> list[complex, complex]:
        """
            Input: self -> density matrix in matrix format
                index -> number of the subspace to trace over (1 or 2)
                particle_type -> list of type of the particles in the density matrix ('fermion' or 'boson')
            Output: TracedRho -> the density matrix with one of its subspace traced over

            Calculates the partial trace of a density matrix of a pair of qubits or qutrits or a system qubit/qutrit
            index = 1, tracing over the first Hilbert Space
            index = 2, tracing over the second Hilbert space
        """
        rho = self.square_matrix()

        if rho.shape[0] != rho.shape[1]:
            raise ValueError('Asked the partial trace of a non-square matrix, problem')
        
        if np.sqrt(rho.shape[0])%1 == 0.: #the density matrix is composed of two systems of same dimension
            n = int(np.sqrt(rho.shape[0]))
            Basis = np.eye(n, dtype=int)
            TracedRho = 0
            if index == 1:
                for i in range(n):
                    Term1 = np.kron(Basis[i], Basis)
                    Term2 = np.transpose(np.kron(Basis[i], Basis))
                    TracedRho += np.dot(Term1, np.dot(rho, Term2))
            elif index == 2:
                for i in range(n):
                    Term1 = np.kron(Basis, Basis[i])
                    Term2 = np.transpose(np.kron(Basis, Basis[i]))
                    TracedRho += np.dot(Term1, np.dot(rho, Term2))
            else:
                raise ValueError('The value of index is not correct')

        #the density matrix is composed of two systems of different dimensions, from rho itself we can't know which one is the first particle so we need to propagate the pdg-id codes
        elif rho.shape[0] == 6: #this case is the one for fermion-boson or boson-fermion
            if particle_type[0] == 'fermion' and particle_type[1] == 'boson': #fermion 1st particle, boson second particle
                if index == 1:
                    n = 2
                    Basis = np.eye(n, dtype=int)
                    TracedRho = 0
                    for i in range(n):
                        Term1 = np.kron(Basis[i], Identity3)
                        Term2 = np.transpose(np.kron(Basis[i], Identity3))
                        TracedRho += np.dot(Term1, np.dot(rho, Term2))

                elif index == 2:
                    n = 3
                    Basis = np.eye(n, dtype=int)
                    TracedRho = 0
                    for i in range(n):
                        Term1 = np.kron(Identity2, Basis[i])
                        Term2 = np.transpose(np.kron(Identity2, Basis[i]))
                        TracedRho += np.dot(Term1, np.dot(rho, Term2))
                else:
                        raise ValueError('The value of index is not correct')
                    
            elif particle_type[0] == 'boson' and particle_type[1] == 'fermion': #boson 1st particle, fermion second particle
                if index == 1:
                    n = 3
                    Basis = np.eye(n, dtype=int)
                    TracedRho = 0
                    for i in range(n):
                        Term1 = np.kron(Basis[i], Identity2)
                        Term2 = np.transpose(np.kron(Basis[i], Identity2))
                        TracedRho += np.dot(Term1, np.dot(rho, Term2))

                elif index == 2:
                    n = 2
                    Basis = np.eye(n, dtype=int)
                    TracedRho = 0
                    for i in range(n):
                        Term1 = np.kron(Identity3, Basis[i])
                        Term2 = np.transpose(np.kron(Identity3, Basis[i]))
                        TracedRho += np.dot(Term1, np.dot(rho, Term2))
                else:
                        raise ValueError('The value of index is not correct')
        
        else:
            raise ValueError("Partial_Trace is not yet implemented for systems beyond 2x2, 2x3 and 3x3.")
        
        return TracedRho

    def Negativity(self, particle_type:list[str], epsilon=1e-10) -> tuple[float, float]:
            """
            Input:  self -> density matrix
                    pdg_pos -> list of the pdg code of the particles in the density matrix
            Output: negativity and logarithmic negativity
            This functions computes the negativity of a density matrix composed of any two particles.
            """
            from numpy import linalg as la
            #We take the partial transpose of the second particle because the negativity does not depend on it 
            rho_TB = self.Partial_Transpose(2, particle_type)
            aux = np.dot(np.transpose(np.conjugate(rho_TB)), rho_TB)

            eigvals, _ = la.eigh(aux) # diagonalization formula: M = P D P^{-1}
            for i in range(len(eigvals)): # we need to put this threshold because of numerical unstabilities small negative eigenvalues can appear
                if abs(eigvals[i]) < epsilon:
                        eigvals[i] = 0.
            norm_trace = np.sum(np.sqrt(eigvals))

            Negativity = (norm_trace - 1)/2.
            Log_Negativity = np.log2(norm_trace)

            if abs(Negativity.real) < epsilon:
                Negativity = 0.
            if (Log_Negativity.real) < epsilon:
                Log_Negativity = 0.

            return Negativity.real, Log_Negativity.real


    def PeresHorodecki_criterion(self, particle_type: list[str], epsilon=1e-7) -> tuple[bool, list[float]]:
        """
        Input:  rho -> density matrix
                particle_type -> type of the two particles in the density matrix ('fermion' or 'boson')
                epsilon -> treshold under which the eigenvalues are taken as 0 to deal with numerical instabilities
        Output: flag_entanglement -> True if the criterion is satisfied and shows entanglement
                eigvals -> eigenvalues of the partially transposed density matrix
        """
        from numpy import linalg as la
        rhoTB = self.Partial_Transpose(2, particle_type)
        eigvals, _ = la.eigh(rhoTB)
        for i in range(len(eigvals)):
            if abs(eigvals[i]) < epsilon:
                eigvals[i] = 0
        flag_entanglement = False
        for eig in eigvals:
            if eig < 0.:
                flag_entanglement = True
                break

        return flag_entanglement, eigvals
    
    def cut_digit_matrix(self, epsilon=1e-10) -> list[complex, complex]:
        """ This method puts to 0 all elements of the matrix whose norm are smaller than the threshold epsilon."""
        square_matrix = self.square_matrix()
        for i in range(self.density_matrix_dimension):
            for j in range(self.density_matrix_dimension):
                if np.abs(square_matrix[i][j]) < epsilon:
                    square_matrix[i][j] = 0.
        return square_matrix

    def shift_clock(self, d:int, epsilon=1e-10) -> tuple[list[complex, complex], list[int, int]]:
        """
        Input:  d -> dimension of the Hilbert space of the chosen particle. Fermion = 2, Massive boson = 3.
        Output: the clock operator Z and the shift operator X.
        """
        RootsUnity = [np.exp(2*np.pi*k*1j/d) for k in range(d)]
        Z = np.diag(RootsUnity)
        for i in range(len(Z)):
            for j in range(len(Z)):
                if abs(Z[i][j].real) < epsilon:
                    Z[i][j] = 0. + Z[i][j].imag *1j
                if abs(Z[i][j].imag) < epsilon:
                    Z[i][j] = Z[i][j].real + 0. *1j
            
        X = np.diag([1 for o in range(d - 1)], -1)
        X[0][d - 1] = 1
        return Z, X
    
    def Displacement_Operator(self, d:int, epsilon=1e-10) -> list[list[complex, complex]]:
        """
            Input:  d -> dimension of the Hilbert space of the chosen particle
            Output: list of all the displacement operator
            The convention in the ordering of the indices follows [2003.02717]
            Note: the convention differs from the one of qbism.
        """
        Displacement_list = []

        Z, X = self.shift_clock(d, epsilon)
        RootUnity = np.exp(2*np.pi*1j/d)
        for u in range(d):
            for v in range(d):
                Prefactor = RootUnity ** (u * v * (d + 1) / 2)
                Xu = la.matrix_power(X, u)
                Zv = la.matrix_power(Z, v)
                D = Prefactor * np.dot(Xu, Zv)
                for i in range(len(D)):
                    for j in range(len(D)):
                        if abs(D[i][j].real) < epsilon:
                            D[i][j] = 0. + D[i][j].imag * 1j
                        if abs(D[i][j].imag) < epsilon:
                            D[i][j] = D[i][j].real + 0. * 1j            

                Displacement_list.append(D)
        return Displacement_list

    def Discrete_phase_point_operator_00(self, d1:int, d2:int, epsilon=1e-10) -> list[complex, complex]:
        """
            Input:  d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
            Output: The operator A_00 
        """
        if d2 == 0:
            A00 = np.zeros((d1,d1), dtype=np.complex128)
            Displacement_operators = self.Displacement_Operator(d1, epsilon)
            for i in range(d1):
                A00 += Displacement_operators[i]
            
            A00/= d1
        
        else:
            A00 = np.zeros((d1 * d2, d1 * d2), dtype=np.complex128)
            Displacement_operators1 = self.Displacement_Operator(d1, epsilon)
            Displacement_operators2 = self.Displacement_Operator(d2, epsilon)
            for i in range(len(Displacement_operators1)):
                for j in range(len(Displacement_operators2)):
                    A00 += np.kron(Displacement_operators1[i], Displacement_operators2[j])
            A00 /= (d1 * d2)

        for i in range(len(A00)):
            for j in range(len(A00)):
                if abs(A00[i][j].real) < epsilon:
                    A00[i][j] = 0. + A00[i][j].imag * 1j
                if abs(A00[i][j].imag) < epsilon:
                    A00[i][j] = A00[i][j].real + 0. * 1j

        return A00

    def Discrete_phase_point_operators(self, d1: int, d2: int, epsilon=1e-10) ->list[list[complex, complex]]:
        """
        Input:   d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
        Output:   The list of operators A_uv
        """
        if d2 == 0:
            #this first block calculates A00
            A00 = np.zeros((d1,d1), dtype=np.complex128)
            Displacement_operators = self.Displacement_Operator(d1)
            for i in range(len(Displacement_operators)):
                A00 += Displacement_operators[i]
            A00/= d1

            for i in range(len(A00)):
                for j in range(len(A00)):
                    if abs(A00[i][j].real) < epsilon:
                        A00[i][j] = 0. + A00[i][j].imag * 1j
                    if abs(A00[i][j].imag) < epsilon:
                        A00[i][j] = A00[i][j].real + 0. * 1j

            #this second block calculates Auv
            Amatrices = []
            for k in range(len(Displacement_operators)):
                aux = np.dot(A00, np.transpose(np.conjugate(Displacement_operators[k])))
                result = np.dot(Displacement_operators[k], aux)
                Amatrices.append(result)
            return Amatrices
            
        else:
            A00 = np.zeros((d1 * d2, d1 * d2), dtype=np.complex128)
            Displacement_operators1 = self.Displacement_Operator(d1)
            Displacement_operators2 = self.Displacement_Operator(d2)
            for i in range(len(Displacement_operators1)):
                for j in range(len(Displacement_operators2)):
                    A00 += np.kron(Displacement_operators1[i], Displacement_operators2[j])
            A00 /= (d1 * d2)

            for i in range(len(A00)):
                for j in range(len(A00)):
                    if abs(A00[i][j].real) < epsilon:
                        A00[i][j] = 0. + A00[i][j].imag * 1j
                    if abs(A00[i][j].imag) < epsilon:
                        A00[i][j] = A00[i][j].real + 0. * 1j


            Dmatrices = [] #tensor product of the displacement operators
            for i in range(len(Displacement_operators1)):
                for j in range(len(Displacement_operators2)):
                    Dmatrices.append(np.kron(Displacement_operators1[i], Displacement_operators2[j]))

            Amatrices = []
            for k in range(len(Dmatrices)):
                aux = np.dot(A00, np.transpose(np.conjugate(Dmatrices[k])))
                result = np.dot(Dmatrices[k], aux)
                Amatrices.append(result)

            return Amatrices

    def Sum_Discrete_Wigner(self, d1:int, d2:int, epsilon=1e-10) -> float:
        """
            Input:  rho -> density matrix 
                    d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
            Output: The sum of the absolute value of the discrete Wigner function for all the possible A_uv
        """
        Amatrices = self.Discrete_phase_point_operators(d1, d2, epsilon)
        Wigner = 0
        for i in range(len(Amatrices)):
            Wigner += np.abs(np.trace(np.dot(self.square_matrix(), Amatrices[i])))

        if d2 == 0:
            return Wigner / d1
        else:
            return Wigner / (d1 * d2)


class DensityMatrixObservables22(DensityMatrixObservables):
    """
        Class reprensenting density matrices of qubit-qubit systems.
        It is a child of the generic class and thus inherits its methods.
        We define here the observables specific to qubit-qubit systems.
    """
    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """The initialisation is the same as the parent class because it needs to accept the same input"""
        super().__init__(user_input, len_user_input)
        if self.density_matrix_dimension != 4:
            raise ValueError("The dimension of a qubit-qubit density matrix should be 4.")

    def Get_Correlations(self, epsilon=1e-10) -> tuple[list[float, float], list[float, float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Correlations matrix C and the square C^T . C
        """
        Correlations = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                tensorprod = np.kron(sigma[i], sigma[j])
                Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, self.square_matrix())))
                if np.abs(Correlations[i][j]) < epsilon:
                    Correlations[i][j] = 0.
        
        Correlations_squared = np.dot(np.transpose(Correlations),Correlations)
        return Correlations, Correlations_squared
    
    def Get_Polarisations(self, epsilon=1e-10) -> tuple[list[float, float], list[float, float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Polarisation vectors of the two particles of the system
        """
        Polarisation1 = np.zeros(3, dtype=np.complex128)
        Polarisation2 = np.zeros(3, dtype=np.complex128)
        for i in range(3):
            tensorprod1 = np.kron(sigma[i], Identity2)
            tensorprod2 = np.kron(Identity2, sigma[i])
            Polarisation1[i] = np.trace(np.dot(tensorprod1, self.square_matrix()))
            Polarisation2[i] = np.trace(np.dot(tensorprod2, self.square_matrix()))

        if np.abs(Polarisation1[i].real) < epsilon:
            Polarisation1[i] = 0. + Polarisation1[i].imag*1j
        if np.abs(Polarisation1[i].imag) < epsilon:
            Polarisation1[i] = Polarisation1[i].real + 0.*1j
        
        if np.abs(Polarisation2[i].real) < epsilon:
            Polarisation2[i] = 0. + Polarisation2[i].imag*1j
        if np.abs(Polarisation2[i].imag) < epsilon:
            Polarisation2[i] = Polarisation2[i].real +0.*1j

        return Polarisation1, Polarisation2

    def spin_expectation(self, epsilon=1e-10) -> tuple[list[float, float], list[float, float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Spin expectation value of the two particles of the system.
        """
        Polarisation1, Polarisation2 = self.Get_Polarisations(epsilon)
        spin_exp1 = np.array(Polarisation1)/2.
        spin_exp2 = np.array(Polarisation2)/2.
        return np.real(spin_exp1), np.real(spin_exp2)
    
    def spinspin_expectation(self, epsilon=1e-10) -> list[float, float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Expectation value of the spin-spin correlations of the system
        """
        spin_spin_exp = np.zeros((3,3), dtype=np.complex128)
        spin_spin_exp = np.array(self.Get_Correlations(epsilon)[0]) / 4.
        return np.real(spin_spin_exp)
    
    def add_phase_density(self, phases, epsilon=1e-10) -> list[complex, complex]:
        """
            Input:  self -> density matrix
                    phases -> list of phases to add to the compoenents of the spinors
                    epsilon -> threshold parameter
            Output: density matrix with the new definition of spinors
        """
        alpha = phases[0]
        beta = phases[1]
        gamma = phases[2]
        delta = phases[3]
        U = np.diag([np.exp((alpha + delta)*1j), np.exp((alpha + gamma)*1j), np.exp((beta + delta)*1j), np.exp((beta + gamma)*1j)])

        aux = np.dot(self.square_matrix(), U)
        rho_corrected = np.dot(np.conjugate(U), aux)

        for i in range(len(rho_corrected)):
            for j in range(len(rho_corrected)):
                if abs(rho_corrected[i][j].real) < epsilon:
                    rho_corrected[i][j] = 0. + rho_corrected[i][j].imag * 1j
                if abs(rho_corrected[i][j].imag) < epsilon:
                    rho_corrected[i][j] = rho_corrected[i][j].real + 0. * 1j
        
        self.density_matrix = rho_corrected

    def CHSH_inequality(self) -> tuple[list[float], bool]:
        '''
            Input: self -> density matrix
            Output: value of the critation, boolean which is True if the inegality is violated
            This method computes whether the CHSH is violated using the eigenvalues of C^T . C
            This criterion only works for pair of qubits with no polarisation.
        '''
        Correlations = self.Get_Correlations()[0]
        CtransposeC = np.dot(np.transpose(Correlations), Correlations)

        eigvals, _ = la.eigh(np.array(CtransposeC))
        sorted_eigs = sorted(eigvals, reverse=True)
        crit = 1 - sorted_eigs[-1] - sorted_eigs[-2]
        return crit, crit < 0

    def Get_Concurrence(self) -> float:
        """
        Input:  self -> density matrix
        Output: concurrence of the system
        """
        if len(self.square_matrix()) != 4:
            raise ValueError('The length of the density matrix is not correct. This function only handles systems of 2 qubits!')
        
        aux = np.kron(sigma2, sigma2)
        rho_tilde = np.dot(aux, np.dot(np.conjugate(self.square_matrix()), aux)) # rhotilde = aux * rho* * aux
        eigvals, eigvecs = la.eigh(np.array(self.square_matrix())) # diagonalization formula: M = P D P^{-1}
        for o in range(len(eigvals)): #This is necessary because numerical errors can cause a almost zero eigenvalue to be negative.
            if eigvals[o]/max(eigvals) < 1e-20:
                eigvals[o] = 0.       

        sqrt_rho = np.dot(eigvecs, np.dot(np.sqrt(np.diag(eigvals)), la.inv(eigvecs)))

        rho_aux = np.dot(sqrt_rho, np.dot(rho_tilde, sqrt_rho))

        eigvals2, _ = la.eigh(rho_aux)
        for o in range(len(eigvals2)): #This is necessary because numerical errors can cause a almost zero eigenvalue to be negative.
            if eigvals2[o]/max(eigvals2) < 1e-20:
                eigvals2[o] = 0.

        final_eigvals = np.sqrt(eigvals2)
        final_eigs_sorted = np.sort(final_eigvals)

        Concurrence = max(0, final_eigs_sorted[3] - final_eigs_sorted[2] - final_eigs_sorted[1] - final_eigs_sorted[0]) # = max(0, lambda1 - lambda2 - lambda3 - lambda4)

        return Concurrence

    def Get_Dcoef(self) -> tuple[float, float, float, float, bool]:
            """
            Input:  self -> density matrix
            Output: D1, Dr, Dn, Dk -> the 4 D-coefficients
                    boolD -> True if there entanglement
            In MadGraph we use the momenta referential as: {n, r, k} with:
            k : top direction, r = (p - k cos(theta))/sin(theta), n = p x k/sin(theta)
            So C_nn = C[0][0], C_rr = C[1][1], C_kk = C[2][2]
            """
            Correlations = self.Get_Correlations()[0]
            if len(Correlations) != 3:
                    raise ValueError('The length of the correlation matrix is not correct. This function only handles systems of 2 qubits!')

            Done = np.real((Correlations[0][0] + Correlations[1][1] + Correlations[2][2])/3.) #D1 = (C_nn + C_rr + C_kk)/3
            Dn = np.real((Correlations[0][0] - Correlations[1][1] - Correlations[2][2])/3.) #Dn = -(-C_nn + C_rr + C_kk)/3
            Dr = np.real((-Correlations[0][0] + Correlations[1][1] - Correlations[2][2])/3.) #Dr = -(C_nn - C_rr + C_kk)/3
            Dk = np.real((-Correlations[0][0] - Correlations[1][1] + Correlations[2][2])/3.) #Dk = -(C_nn + C_rr - C_kk)/3
            boolD = Done < - 1/3 or Dn < - 1/3 or Dr < - 1/3 or Dk < - 1/3 #if boolD = True, then there is entanglement
            return Done, Dn, Dr, Dk, boolD
    
    def Get_Entanglement_Formation(self) -> float:
        """
            Input:  self -> density matrix
            Output: entanglement of formation of the system
        """
        concurrence = self.Get_Concurrence()
        return self.Shannon_Entropy((1 + np.sqrt(1 - concurrence**2))/2)

    def get_Pauli_string(self, n:int) -> list[list[complex, complex]]:
        """
        Input:   self -> density matrix
                 n -> dimension of the Pauli string
        Output:  List of the elements of the Pauli string
        """
        Paulistring1 = [Identity2]
        for i in range(len(sigma)):
            Paulistring1.append(sigma[i])
        
        Pauli_string = []

        if n == 1:
            Pauli_string = Paulistring1
            return Pauli_string
        else:
            for elem1 in Paulistring1:
                for elem2 in self.get_Pauli_string(n - 1):
                    Pauli_string.append(np.kron(elem1, elem2))
            return Pauli_string

    def Magic_Pure(self, n=2) -> float:
            """
            Input:  self -> density matrix
                    n -> number of qubits (usually taken as 2)  
            Output: magic of the system
            Computes the quantity M2 for a density matrix which represents a pure state for a system of n qubits
            """
            Xi = 0
            Pauli_strings = self.get_Pauli_string(n)
            for Pstring in Pauli_strings:
                    Xi += np.trace(np.dot(Pstring, self.square_matrix())) ** 4
            return (- np.log2(Xi / 2**n)).real

    def Magic_Mixed(self, n=2) -> float:
        """
        Input:  self -> density matrix in matrix format
                n -> number of qubits (usually taken as 2)  
        Output: magic of the system
        Computes the quantity M2~ for a density matrix which represents a mixed state for a system of n qubits
        """
        XiNum = 0
        XiDenom = 0
        Pauli_strings = self.get_Pauli_string(n)
        for Pstring in Pauli_strings:
                XiNum += np.trace(np.dot(Pstring, self.square_matrix())) ** 4
                XiDenom += np.trace(np.dot(Pstring, self.square_matrix())) ** 2
        Magic = - np.log2(XiNum / XiDenom)
        return Magic.real

    def Get_Discord(self, maxiter=100) -> float:
        """
        Algorithm based on formula (3) from [2209.03969]. It computes Discord for a given density matrix rho.
        Input: self -> density matrix
               maxiter -> maximum number of iterations for the minimisation
        Output: float -> Discord
        """
        from scipy.optimize import minimize

        Srho = self.Von_Neumann_entropy() # S(rho)
        rhoB = self.Partial_Trace(2, ['fermion', 'fermion'])
        SrhoB = self.Von_Neumann_entropy(rho=rhoB) #S(rho_B)

        B_plus, B_minus = self.Get_Polarisations()
        Corr, _ = self.Get_Correlations()

        def proba_n(n: list[float]) -> float: #B- = B2
            return (1 + np.dot(B_minus, n)) / 2
        def Bn_plus(n: list[float]) -> list[complex]:
            return np.array(B_plus) + np.dot(Corr,n) / (1 + np.dot(B_minus, n))
        def rho_n(n: list[float]) -> list[complex, complex]:
            Bnp = Bn_plus(n)
            return (np.eye(2) + Bnp[0]*sigma[0] + Bnp[1]*sigma[1] + Bnp[2]*sigma[2])/2


        #Here we have the minimisation problem to do on the sphere
        def objective_function(n: list[float]) -> float: #n is the Boch vector
            n_opp = [-n[0], -n[1], -n[2]]
            fun = proba_n(n)*self.Von_Neumann_entropy(rho=rho_n(n)) + proba_n(n_opp)*self.Von_Neumann_entropy(rho=rho_n(n_opp)) #function to minimise
            return fun.real
        
        # Define the constraint for the unit sphere
        constraints = ({'type': 'eq', 'fun': lambda n: n[0]**2 + n[1]**2 + n[2]**2 - 1})

        # Initial guess
        vec = np.random.randn(3, 1)
        vec /= np.linalg.norm(vec, axis=0)
        x0 = [vec[0][0], vec[1][0], vec[2][0]]

        # Perform the optimization
        result = minimize(objective_function, x0, constraints=constraints, options={'maxiter': maxiter})
        min_result = result.fun

        Discord = SrhoB - Srho + min_result
        
        return Discord.real



class DensityMatrixObservables23(DensityMatrixObservables):
    """
        Class reprensenting density matrices of qubit-qutrit systems.
        It is a child of the generic class and thus inherits its methods.
        We define here the observables specific to qubit-qubit systems.
    """

    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """The initialisation is the same as the parent class because it needs to accept the same input"""
        super().__init__(user_input, len_user_input)
        if self.density_matrix_dimension != 6:
            raise ValueError("The dimension of a qubit-qutrit density matrix should be 6.")
    
    def Get_Correlations(self, particle_type: list[str], epsilon=1e-10) -> tuple[list[float, float], list[float, float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Correlations matrix C and the square C^T . C
        """
        if particle_type[0] == 'fermion':
            Correlations = np.zeros((3, 8), dtype=float)
            for i in range(3):
                for j in range(8):
                    tensorprod = np.kron(sigma[i], Lambda[j])
                    Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, self.square_matrix()))/2.)
                    if np.abs(Correlations[i][j]) < epsilon:
                        Correlations[i][j] = 0.
        elif particle_type[0] == 'boson':
            Correlations = np.zeros((8, 3), dtype=float)
            for i in range(8):
                for j in range(3):
                    tensorprod = np.kron(Lambda[i], sigma[j])
                    Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, self.square_matrix()))/2.)
                    if np.abs(Correlations[i][j]) < epsilon:
                        Correlations[i][j] = 0.
        else:
            raise ValueError("particle_type only accepts elements that are 'fermion' or 'boson'.")
        
        Correlations_squared = np.dot(np.transpose(Correlations),Correlations)
        return Correlations, Correlations_squared

    def Get_Polarisations(self, particle_type: list[str], epsilon=1e-10) -> tuple[list[float], list[float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Polarisation vectors of the two particles of the system
        """
        if particle_type[0] == 'boson':
            Polarisation1 = np.zeros(8, dtype=np.complex128)
            Polarisation2 = np.zeros(3, dtype=np.complex128)
            for i in range(8):
                tensorprod1 = np.kron(Lambda[i], Identity2)
                Polarisation1[i] = np.trace(np.dot(tensorprod1, self.square_matrix()))/2.
            for j in range(3):
                tensorprod2 = np.kron(Identity3, sigma[j])
                Polarisation2[j] = np.trace(np.dot(tensorprod2, self.square_matrix()))
        elif particle_type[0] == 'fermion':
            Polarisation1 = np.zeros(3, dtype=np.complex128)
            Polarisation2 = np.zeros(8, dtype=np.complex128)
            for i in range(3):
                tensorprod1 = np.kron(sigma[i], Identity3)
                Polarisation1[i] = np.trace(np.dot(tensorprod1, self.square_matrix()))
            for j in range(8):
                tensorprod2 = np.kron(Identity2, Lambda[j])
                Polarisation2[j] = np.trace(np.dot(tensorprod2, self.square_matrix()))/2.
        else:
            raise ValueError("particle_type only accepts elements that are 'fermion' or 'boson'.")
        
        if np.abs(Polarisation1[i].real) < epsilon:
            Polarisation1[i] = 0. + Polarisation1[i].imag*1j
        if np.abs(Polarisation1[i].imag) < epsilon:
            Polarisation1[i] = Polarisation1[i].real + 0.*1j
        
        if np.abs(Polarisation2[i].real) < epsilon:
            Polarisation2[i] = 0. + Polarisation2[i].imag*1j
        if np.abs(Polarisation2[i].imag) < epsilon:
            Polarisation2[i] = Polarisation2[i].real +0.*1j

        return Polarisation1, Polarisation2

    def spin_expectation(self, particle_type:list[str]) -> tuple[list[float], list[float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Spin expectation value of the two particles of the system.
        """
        spin_exp1 = np.zeros(3, dtype=np.complex128)
        spin_exp2 = np.zeros(3, dtype=np.complex128)
        Polarisation1, Polarisation2 = self.Get_Polarisations(particle_type)

        if particle_type[0] == 'boson':
            spin_exp1[0] = np.sqrt(2) * (Polarisation1[0] + Polarisation1[5])
            spin_exp1[1] = np.sqrt(2) * (Polarisation1[1] + Polarisation1[6])
            spin_exp1[2] = Polarisation1[2] + np.sqrt(3) * Polarisation1[7]
            spin_exp2 = np.array(Polarisation2)
        elif particle_type[0] == 'fermion':
            spin_exp1 = np.array(Polarisation1)
            spin_exp2[0] = np.sqrt(2) * (Polarisation2[0] + Polarisation2[5])
            spin_exp2[1] = np.sqrt(2) * (Polarisation2[1] + Polarisation2[6])
            spin_exp2[2] = Polarisation2[2] + np.sqrt(3) * Polarisation2[7]
        else:
            raise ValueError("particle_type only accepts elements that are 'fermion' or 'boson'.")
        
        return np.real(spin_exp1), np.real(spin_exp2)

    def spinspin_expectation(self, particle_type:list[str]) -> list[float, float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Expectation value of the spin-spin correlations of the system
        """
        spin_spin_exp = np.zeros((3,3), dtype=np.complex128)
        Corr = self.Get_Correlations(particle_type)[0]

        if particle_type[0] == 'boson':
            spin_spin_exp[0][0] = (Corr[0][0] + Corr[5][0]) / np.sqrt(2)
            spin_spin_exp[0][1] = (Corr[0][1] + Corr[5][1]) / np.sqrt(2)
            spin_spin_exp[0][2] = (Corr[0][2] + Corr[5][2]) / np.sqrt(2)

            spin_spin_exp[1][0] = (Corr[1][0] + Corr[6][0]) / np.sqrt(2)
            spin_spin_exp[1][1] = (Corr[1][1] + Corr[6][1]) / np.sqrt(2)
            spin_spin_exp[1][2] = (Corr[1][2] +  Corr[6][2]) / np.sqrt(2)

            spin_spin_exp[2][0] = (Corr[2][0] + np.sqrt(3) * Corr[7][0]) / 2.
            spin_spin_exp[2][1] = (Corr[2][1] + np.sqrt(3) * Corr[7][1]) / 2.
            spin_spin_exp[2][2] = (Corr[2][2] + np.sqrt(3) * Corr[7][2]) / 2.

        elif particle_type[0] == 'fermion':
            spin_spin_exp[0][0] = (Corr[0][0] + Corr[0][5]) / np.sqrt(2)
            spin_spin_exp[0][1] = (Corr[0][1] + Corr[0][6]) / np.sqrt(2)
            spin_spin_exp[0][2] = (Corr[0][2] + np.sqrt(3) * Corr[0][7]) / 2.

            spin_spin_exp[1][0] = (Corr[1][0] + Corr[1][5]) / np.sqrt(2)
            spin_spin_exp[1][1] = (Corr[1][1] + Corr[1][6]) / np.sqrt(2)
            spin_spin_exp[1][2] = (Corr[1][2] + np.sqrt(3) * Corr[1][7]) / 2.

            spin_spin_exp[2][0] = (Corr[2][0] + Corr[2][5]) / np.sqrt(2)
            spin_spin_exp[2][1] = (Corr[2][1] + Corr[2][6]) / np.sqrt(2)
            spin_spin_exp[2][2] = (Corr[2][2] + np.sqrt(3) * Corr[2][7]) / 2.

        else:
            raise ValueError("particle_type only accepts elements that are 'fermion' or 'boson'.")
        
        return np.real(spin_spin_exp)
    

class DensityMatrixObservables33(DensityMatrixObservables):
    """
        Class reprensenting density matrices of qubit-qubit systems.
        It is a child of the generic class and thus inherits its methods.
        We define here the observables specific to qubit-qubit systems.
    """
    
    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """The initialisation is the same as the parent class because it needs to accept the same input"""
        super().__init__(user_input, len_user_input)
        if self.density_matrix_dimension != 9:
            raise ValueError("The dimension of a qutrit-qutrit density matrix should be 9.")

    def Get_Correlations(self, epsilon=1e-10) -> tuple[list[float, float], list[float, float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Correlations matrix C and the square C^T . C
        """
        Correlations = np.zeros((8, 8), dtype=float)
        for i in range(8):
            for j in range(8):
                tensorprod = np.kron(Lambda[i], Lambda[j]) #the tensor product is with Gell-Mann matrices
                Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, self.square_matrix()))/4.) #there is an additional 1/4 factor
                if np.abs(Correlations[i][j]) < epsilon:
                    Correlations[i][j] = 0.
        
        Correlations_squared = np.dot(np.transpose(Correlations),Correlations)
        return Correlations, Correlations_squared


    def Get_Polarisations(self, epsilon=1e-10) -> tuple[list[float], list[float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Polarisation vectors of the two particles of the system
        """
        Polarisation1 = np.zeros(8, dtype=np.complex128)
        Polarisation2 = np.zeros(8, dtype=np.complex128)
        for i in range(8):
            tensorprod1 = np.kron(Lambda[i], Identity3)
            tensorprod2 = np.kron(Identity3, Lambda[i])
            Polarisation1[i] = np.trace(np.dot(tensorprod1, self.square_matrix()))/2.
            Polarisation2[i] = np.trace(np.dot(tensorprod2, self.square_matrix()))/2.

        if np.abs(Polarisation1[i].real) < epsilon:
            Polarisation1[i] = 0. + Polarisation1[i].imag*1j
        if np.abs(Polarisation1[i].imag) < epsilon:
            Polarisation1[i] = Polarisation1[i].real + 0.*1j
        
        if np.abs(Polarisation2[i].real) < epsilon:
            Polarisation2[i] = 0. + Polarisation2[i].imag*1j
        if np.abs(Polarisation2[i].imag) < epsilon:
            Polarisation2[i] = Polarisation2[i].real +0.*1j

        return Polarisation1, Polarisation2

    def spin_expectation(self) -> tuple[list[float], list[float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Spin expectation value of the two particles of the system.
        """
        spin_exp1 = np.zeros(3, dtype=np.complex128)
        spin_exp2 = np.zeros(3, dtype=np.complex128)
        Polarisation1, Polarisation2 = self.Get_Polarisations()

        spin_exp1[0] = np.sqrt(2) * (Polarisation1[0] + Polarisation1[5])
        spin_exp1[1] = np.sqrt(2) * (Polarisation1[1] + Polarisation1[6])
        spin_exp1[2] = Polarisation1[2] + np.sqrt(3) * Polarisation1[7]
        spin_exp2[0] = np.sqrt(2) * (Polarisation2[0] + Polarisation2[5])
        spin_exp2[1] = np.sqrt(2) * (Polarisation2[1] + Polarisation2[6])
        spin_exp2[2] = Polarisation2[2] + np.sqrt(3) * Polarisation2[7]

        return np.real(spin_exp1), np.real(spin_exp2)
    
    def spinspin_expectation(self) -> list[float, float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Expectation value of the spin-spin correlations of the system
        """
        spin_spin_exp = np.zeros((3,3), dtype=np.complex128)
        Corr = self.Get_Correlations()[0]

        spin_spin_exp[0][0] = 2 * (Corr[0][0] + Corr[0][5] + Corr[5][0] + Corr[5][5])
        spin_spin_exp[0][1] = 2 * (Corr[0][1] + Corr[0][6] + Corr[5][1] + Corr[5][6])
        spin_spin_exp[0][2] = np.sqrt(2) *(Corr[0][2] + Corr[5][2] + np.sqrt(3) * (Corr[0][7] + Corr[5][7]))

        spin_spin_exp[1][0] = 2 * (Corr[1][0] + Corr[6][0] + Corr[1][5] + Corr[6][5])
        spin_spin_exp[1][1] = 2 * (Corr[1][1] + Corr[1][6] + Corr[6][1] + Corr[6][6])
        spin_spin_exp[1][2] = np.sqrt(2) *(Corr[1][2] + Corr[6][2] + np.sqrt(3) * (Corr[1][7] + Corr[6][7]))

        spin_spin_exp[2][0] = np.sqrt(2) * (Corr[2][0] + Corr[2][5] + np.sqrt(3) * (Corr[7][0] + Corr[7][5]))
        spin_spin_exp[2][1] = np.sqrt(2) * (Corr[2][1] + Corr[2][6] + np.sqrt(3) * (Corr[7][1] + Corr[7][6]))
        spin_spin_exp[2][2] = Corr[2][2] + np.sqrt(3) * (Corr[2][7] + Corr[7][2]) + 3 * Corr[7][7]
        
        return np.real(spin_spin_exp)


    def add_phase_density(self, phases, epsilon=1e-10) -> list[complex, complex]:
        """
            Input:  self -> density matrix
                    phases -> list of phases to add to the compoenents of the spinors
                    epsilon -> threshold parameter
            Output: density matrix with the new definition of spinors
        """
        alpha = phases[0]
        beta = phases[1]
        gamma = phases[2]
        delta = phases[3]
        epsilon = phases[4]
        tau = phases[5]
        U = np.diag([np.exp((alpha + tau)*1j), np.exp((alpha + epsilon)*1j), np.exp((alpha + delta)*1j), np.exp((beta + tau)*1j), np.exp((beta + epsilon)*1j), np.exp((beta + delta)*1j), np.exp((gamma + tau)*1j), np.exp((gamma + epsilon)*1j), np.exp((gamma + delta)*1j)])
        
        aux = np.dot(self.square_matrix(), U)
        rho_corrected = np.dot(np.conjugate(U), aux)
        
        for i in range(len(rho_corrected)):
            for j in range(len(rho_corrected)):
                if abs(rho_corrected[i][j].real) < epsilon:
                    rho_corrected[i][j] = 0. + rho_corrected[i][j].imag * 1j
                if abs(rho_corrected[i][j].imag) < epsilon:
                    rho_corrected[i][j] = rho_corrected[i][j].real + 0. * 1j
        
        return rho_corrected

    def ConcLB2(self, epsilon=1e-10) -> float:
        """
            Input:  self -> density matrix
                    particle_type -> type of the particles in the density matrix ('fermion' or 'boson')
            Output: square of the lower bound of the concurrence for a system composed of a pair of qutrits.
        """

        RhoA = self.Partial_Trace(2, ['boson', 'boson'])
        RhoB = self.Partial_Trace(1, ['boson', 'boson'])

        if ((np.trace(RhoA) - 1) > epsilon) or ((np.trace(RhoB) - 1) > epsilon):
            print('Warning: the traced-out density matrices have non unitary trace!')

        aux1 = np.trace(np.dot(self.square_matrix(), self.square_matrix())) - np.trace(np.dot(RhoA, RhoA))
        aux2 = np.trace(np.dot(self.square_matrix(), self.square_matrix())) - np.trace(np.dot(RhoB, RhoB))
        ConcLB2 = 2 * max(0, aux1, aux2)

        return ConcLB2.real
    
    def ConcUB2(self, epsilon=1e-10) -> float:
        """
            Input:  self -> density matrix
                    particle_type -> type of the particles in the density matrix ('fermion' or 'boson')
            Output: square of the upper bound of the concurrence for a system composed of a pair of qutrits.
        """

        RhoA = self.Partial_Trace(2, ['boson', 'boson'])
        RhoB = self.Partial_Trace(1, ['boson', 'boson'])

        if ((np.trace(RhoA) - 1) > epsilon) or ((np.trace(RhoB) - 1) > epsilon):
            print('Warning: the traced-out density matrices have non unitary trace!')

        aux1 = 1 - np.trace(np.dot(RhoA, RhoA))
        aux2 = 1 - np.trace(np.dot(RhoB, RhoB))
        ConcUB2 = 2 * min(aux1, aux2)
        return ConcUB2.real

    def Get_Mana(self, d1=3, d2=3, epsilon=1e-10) -> float:
        """
        Input:  rho -> density matrix 
                d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
        Output: the observable Mana. Caution: it is only defined for d1,d2 = 3, 5
        """
        if (d1 not in [3, 5]) or (d2 not in [3, 5]):
            raise ValueError("Mana is only defined for odd-dimension Hilbert spaces. Please use it for adapted systems.")
        return np.log2(self.Sum_Discrete_Wigner(d1, d2, epsilon))