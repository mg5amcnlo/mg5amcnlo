#We add a try/except to the import of numpy because all the unittests need to pass without numpy
try:
    import numpy as np
    from numpy import linalg as la

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

except:
    pass

try:
    from typing import Union
except:
    pass


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

# Generating a random density matrix
def random_density_matrix_qubitqubit(seed=None):
    """Random 4x4 density matrix."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real

def su2_from_so3(O, tol=1e-8):
    """
    Returns the SU(2) matrix U such that U.sigma_j.U^dagger = sum_k O[k, j] sigma_k for a rotation O of SO(3).
    It is used to convert the matrices W and Z from the Singular Value Decomposition of the correlation matrix into
    local unitary transformations of the density matrix (see Get_Local_Normal_Form). It is only possible if the determinant is 1.
    """
    O = np.asarray(O, dtype=float)
    if not np.allclose(O @ O.T, np.eye(3), atol=tol):
        raise ValueError("matrix is not orthogonal")
    if np.linalg.det(O) < 0:
        raise ValueError("determinant is -1: reflections have no SU(2) preimage")

    # quaternion extraction (Shepperd's method: use the largest component)
    t = np.trace(O)
    if t > 0:
        w = np.sqrt(1.0 + t) / 2.0
        x = (O[2, 1] - O[1, 2]) / (4 * w)
        y = (O[0, 2] - O[2, 0]) / (4 * w)
        z = (O[1, 0] - O[0, 1]) / (4 * w)
    else:
        k = int(np.argmax(np.diag(O)))
        if k == 0:
            x = np.sqrt(max(0.0, 1 + O[0, 0] - O[1, 1] - O[2, 2])) / 2.0
            w = (O[2, 1] - O[1, 2]) / (4 * x)
            y = (O[0, 1] + O[1, 0]) / (4 * x)
            z = (O[0, 2] + O[2, 0]) / (4 * x)
        elif k == 1:
            y = np.sqrt(max(0.0, 1 - O[0, 0] + O[1, 1] - O[2, 2])) / 2.0
            w = (O[0, 2] - O[2, 0]) / (4 * y)
            x = (O[0, 1] + O[1, 0]) / (4 * y)
            z = (O[1, 2] + O[2, 1]) / (4 * y)
        else:
            z = np.sqrt(max(0.0, 1 - O[0, 0] - O[1, 1] + O[2, 2])) / 2.0
            w = (O[1, 0] - O[0, 1]) / (4 * z)
            x = (O[0, 2] + O[2, 0]) / (4 * z)
            y = (O[1, 2] + O[2, 1]) / (4 * z)

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q

    # U = cos(th/2) 1 - i sin(th/2) n.sigma
    # this is the general local unitary transformation of SU(2)
    U = w * Identity2 - 1j * (x * sigma[0] + y * sigma[1] + z * sigma[2])

    # self-check of equations in paragraph below equation (15) of https://journals.aps.org/pra/abstract/10.1103/PhysRevA.77.042303
    for j in range(3):
        lhs = U @ sigma[j] @ U.conj().T
        rhs = sum(O[k, j] * sigma[k] for k in range(3))
        if not np.allclose(lhs, rhs, atol=tol):
            raise ValueError(f"translation of SO(3) into SU(2) failed:\n{lhs}\n!=\n{rhs}")

    return U

# Distances betweeen 2 matrices
def trace_distance(Matrix1: list[complex, complex], Matrix2: list[complex, complex]) -> float:
    """
       Input: Matrix1, Matrix2 -> 2 square matrices
       Output: trace distance between the two matrices
    """
    if isinstance(Matrix1, list):
        Matrix1 = np.array(Matrix1)
    if isinstance(Matrix2, list):
        Matrix2 = np.array(Matrix2)
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

    if isinstance(z[0], (float, int, complex, np.number)) or isinstance(z[0], np.generic):
        Map = np.zeros((n_biny, n_binx))
    elif isinstance(z[0], np.ndarray): #if the object is a density matrix
        Map = np.zeros((n_biny, n_binx), dtype=object)
        shape_input = z[0].shape
        for i in range(len(Map)):
            for j in range(len(Map[0])):
                Map[i][j] = np.zeros(shape_input, dtype=np.complex128)
    else:
        raise TypeError("The argument z does not accept lists, please use a numpy array.")
        
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
        A class is defined for single qubits and qutrits, for qubit-qubit systems, as well as qubit-qutrit and qutrit-qutrit systems. 
        In these classes we defined the observables specific to these systems like concurrence.
        This class redirects to the specific classes based on the user input. One should always call this generic class.
    """
    def __new__(cls, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None):
        """ 
        This method determines the dimension of the user input and decides which subclass to instanciate.
        For now, there are classes for density matrices of single qubits and qutrits, and of 2x2, 2x3, 3x3, 2x2x2 and 2x2x3 systems.
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
        subclass = DensityMatrixObservables.class_for_dimension(dim_density_matrix)
        return super(DensityMatrixObservables, cls).__new__(subclass, user_input, len_user_input)
        
              
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

    @staticmethod
    def class_for_dimension(dim_density_matrix:int) -> type:
        """Returns the class to use for a density matrix of dimension dim_density_matrix (generic class if no specific one)."""
        if dim_density_matrix == 2:
            return DensityMatrixObservables2
        elif dim_density_matrix == 3:
            return DensityMatrixObservables3
        elif dim_density_matrix == 4:
            return DensityMatrixObservables22
        elif dim_density_matrix == 6:
            return DensityMatrixObservables23
        elif dim_density_matrix == 9:
            return DensityMatrixObservables33
        elif dim_density_matrix == 8:
            return DensityMatrixObservables222
        elif dim_density_matrix == 12:
            return DensityMatrixObservables223
        else:
            return DensityMatrixObservables

    @staticmethod
    def get_subsystem_dims(particle_type:list[str]=None, dims:list[int]=None, D:int=None) -> list[int]:
        """
            Input:  particle_type -> list of type of the particles ('fermion' or 'boson')
                    dims -> list of the dimension of each subspace (overrides particle_type)
                    D -> dimension of the full density matrix, used to check dims or to guess it
            Output: dims -> list of the dimension of each subspace

            If neither dims nor particle_type is given, the density matrix is assumed to be made of two subsystems of same dimension.
        """
        if dims is None:
            if particle_type is not None:
                dim_of_type = {'fermion': 2, 'boson': 3}
                try:
                    dims = [dim_of_type[p] for p in particle_type]
                except KeyError as error:
                    raise ValueError('Unknown particle type %s, expected fermion or boson' % error)
            elif D is not None and np.sqrt(D)%1 == 0.: #two systems of same dimension
                dims = [int(np.sqrt(D))]*2
            else:
                raise ValueError('Cannot guess the dimension of the subspaces, please give dims or particle_type')
        
        dims = [int(d) for d in dims]
        if D is not None and int(np.prod(dims)) != D:
            raise ValueError('The density matrix has dimension %d, not compatible with subspaces of dimensions %s' % (D, dims))
        return dims

    @staticmethod
    def get_subsystem_indices(index:Union[int, list[int]], n:int) -> list[int]:
        """Converts the subsystem index(es) given by the user (starting at 0) to a sorted list of indices and checks they are valid."""
        if np.isscalar(index):
            index = [index]
        indices = sorted({int(i) for i in index})
        if any(i < 0 or i >= n for i in indices):
            raise ValueError('The value of index is not correct, it should be between 0 and %d' % (n - 1))
        return indices      

    def is_hermitian(self):
        Matrix = self.square_matrix()
        if np.allclose(Matrix, Matrix.conj().T):
            return True
        else:
            return False
    
    def density_array(self):
        return np.array(self.density_matrix)

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
        The line matrix is the upper triangle stored row by row, as filled by GET_DENSITY in Fortran:
        line[i*n + k - i*(i + 1)//2] = M_i M_k^* = rho[i][k] for k >= i. The lower triangle is its conjugate.
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
                            if k >= i:
                                    matrix_square[i][k] = self.density_matrix[i*n + k - i*(i + 1)//2]
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

    def partial_transpose(self, dims, systems, aux_matrix=None):
        """Partial transpose of a density matrix over a set of subsystems.

        rho        : (D, D) array, with D = prod(dims)
        dims       : sequence of subsystem dimensions, e.g. [2, 3, 2]
        systems    : int or sequence of ints, indices of the subsystems to transpose (starts at index 0)
        aux_matrix : matrix that we want to partially transpose that is not the density matrix
        """

        if aux_matrix is None:
            rho = self.square_matrix()
        else:
            rho = aux_matrix
        
        dims = [int(d) for d in dims] #dimension of each subspace
        n = len(dims) #number of subspaces
        D = int(np.prod(dims)) #dimension of the density matrix
        if np.isscalar(systems):
            systems = [systems]
        systems = {s % n for s in systems}

        # cvxpy variables/expressions cannot be reshaped by numpy,
        # so we use the cvxpy partial transpose, one subsystem at a time
        if not isinstance(rho, np.ndarray): #we only want to import cvxpy if the aux_matrix given is not a numpy array (so if it is a cvxpy variable)
            try: 
                import cvxpy as cp
                if isinstance(rho, cp.Expression):
                    if rho.shape != (D, D):
                        raise ValueError("rho has shape %s, expected (%d, %d)" % (rho.shape, D, D))
                    for s in sorted(systems):
                        rho = cp.partial_transpose(rho, dims, axis=s)
                    return rho
            except ImportError:
                pass

        rho = np.asarray(rho)
        if rho.shape != (D, D):
            raise ValueError("rho has shape %s, expected (%d, %d)" % (rho.shape, D, D))

        # rho[i1..in, j1..jn] -> swap the row/column index of each transposed subsystem
        perm = list(range(2 * n))
        for s in systems:
            perm[s], perm[n + s] = n + s, s
        return rho.reshape(dims + dims).transpose(perm).reshape(D, D)
    

    def partial_trace(self, index:Union[int, list[int]], particle_type:list[str]=None, dims:list[int]=None, inplace=False) -> list[complex, complex]:
        """
            Input: self -> density matrix in matrix format
                index -> index of the subspace(s) to trace over, starting at 0 (int or list of int)
                particle_type -> list of type of the particles in the density matrix ('fermion' or 'boson'),
                                 used to get the dimension of each subspace if dims is not given
                dims -> list of the dimension of each subspace, e.g. [2, 3, 2] (overrides particle_type)
                inplace -> if True, the instance is also changed: it then holds the traced density matrix (in square form)
                           and its class is updated to the one matching the new dimension
            Output: TracedRho -> the density matrix with the subspace(s) traced over

            Calculates the partial trace of a density matrix made of any number of subsystems of any dimension.
            If neither dims nor particle_type is given, the density matrix is assumed to be made of two subsystems of same dimension.
            Ex: index = 0, tracing over the first Hilbert Space
                index = [0, 2], tracing over the first and third Hilbert spaces
        """
        rho = np.asarray(self.square_matrix())

        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError('Asked the partial trace of a non-square matrix, problem')
        D = rho.shape[0]

        dims = self.get_subsystem_dims(particle_type, dims, D)
        n = len(dims)
        traced = self.get_subsystem_indices(index, n) #subspaces to trace over
        kept_dims = [dims[i] for i in range(n) if i not in traced]
        d_kept = int(np.prod(kept_dims))

        # rho[i1..in, j1..jn] -> sum over i_s = j_s for each traced subspace s
        # we trace starting from the last subspace so the axes of the remaining ones are unchanged
        TracedRho = rho.reshape(dims + dims)
        n_left = n
        for s in reversed(traced):
            TracedRho = np.trace(TracedRho, axis1=s, axis2=s + n_left)
            n_left -= 1
        TracedRho = TracedRho.reshape(d_kept, d_kept)

        if inplace:
            self.density_matrix = TracedRho
            self.density_matrix_dimension = d_kept
            self.__class__ = DensityMatrixObservables.class_for_dimension(d_kept)

        return TracedRho

    def Negativity(self, particle_type:list[str]=None, epsilon=1e-10, dims:list[int]=None, index:Union[int, list[int]]=None) -> tuple[float, float]:
        """
        Input:  self -> density matrix
                particle_type -> list of the particle type: "fermion" or "boson" (used to get dims if dims is not given)
                epsilon -> threshold under which the results are set to 0 to deal with numerical instabilities
                dims -> list of the dimension of each subspace, e.g. [2, 3, 2] (overrides particle_type)
                index -> subsystem(s) on one side of the bipartition, starting at 0 (int or list of int),
                         by default the last subsystem, i.e. the bipartition (0...n-2)|n-1
        Output: negativity and logarithmic negativity

        This functions computes the negativity of a density matrix with respect to the bipartition index|rest,
        for any number of subsystems of any dimension: N = (||rho^T_index||_1 - 1)/2, E_N = log2(||rho^T_index||_1).
        The result does not depend on which side of the bipartition is transposed.
        """
        dims = self.get_subsystem_dims(particle_type, dims, self.density_matrix_dimension)
        if index is None: #by default we transpose the last subsystem
            index = len(dims) - 1
        rho_T = self.partial_transpose(dims, self.get_subsystem_indices(index, len(dims)))
        # rho^T is hermitian so its trace norm is the sum of the absolute values of its eigenvalues
        norm_trace = np.sum(np.abs(la.eigvalsh(rho_T)))

        Negativity = (norm_trace - 1)/2.
        Log_Negativity = np.log2(norm_trace)

        if abs(Negativity) < epsilon:
            Negativity = 0.
        if Log_Negativity < epsilon:
            Log_Negativity = 0.

        return float(Negativity), float(Log_Negativity)

    def PeresHorodecki_criterion(self, particle_type:list[str]=None, epsilon=1e-7, dims:list[int]=None, index:Union[int, list[int]]=None) -> tuple[bool, list[float]]:
        """
        Input:  rho -> density matrix
                particle_type -> type of the particles in the density matrix ('fermion' or 'boson')
                epsilon -> treshold under which the eigenvalues are taken as 0 to deal with numerical instabilities
                dims -> list of the dimension of each subspace (overrides particle_type)
                index -> subsystem(s) on one side of the bipartition, starting at 0 (last subsystem by default)
        Output: flag_entanglement -> True if the criterion is satisfied and shows entanglement across the bipartition index|rest
                eigvals -> eigenvalues of the partially transposed density matrix
        """
        dims = self.get_subsystem_dims(particle_type, dims, self.density_matrix_dimension)
        if index is None: #by default we transpose the last subsystem
            index = len(dims) - 1
        rhoTB = self.partial_transpose(dims, self.get_subsystem_indices(index, len(dims)))
        eigvals = la.eigvalsh(rhoTB)
        eigvals[np.abs(eigvals) < epsilon] = 0.
        flag_entanglement = bool(np.any(eigvals < 0.))

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

    def Get_Coherence(self):
        """Computes the coherence (the sum of the absolute value of the non-diagonal elements of the density matrix)."""
        rho_square = self.square_matrix()
        coherence = 0
        for i in range(len(rho_square)):
            for j in range(len(rho_square[0])):
                if j != i:
                    coherence += abs(rho_square[i][j])
        
        return coherence

class DensityMatrixObservables2(DensityMatrixObservables):
    """
        Class reprensenting density matrices of a single qubit (spin-1/2 particle).
        It is a child of the generic class and thus inherits its methods.
        The basis is the one of the helicities given by MadGraph: (+1/2, -1/2).
        The density matrix is rho = (1 + B.sigma)/2, with B the polarisation (Bloch) vector, B_i = <sigma_i>.
    """
    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """The initialisation is the same as the parent class because it needs to accept the same input"""
        super().__init__(user_input, len_user_input)
        if self.density_matrix_dimension != 2:
            raise ValueError("The dimension of a qubit density matrix should be 2.")

    def Get_Polarisations(self, epsilon=1e-10) -> list[float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Polarisation vector B_i = Tr(sigma_i rho), same normalisation as DensityMatrixObservables22
        """
        Polarisation = np.zeros(3, dtype=float)
        for i in range(3):
            Polarisation[i] = np.real(np.trace(np.dot(sigma[i], self.square_matrix())))
            if np.abs(Polarisation[i]) < epsilon:
                Polarisation[i] = 0.
        return Polarisation

    def spin_expectation(self, epsilon=1e-10) -> list[float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Spin expectation value <S_i> = B_i/2
        """
        return self.Get_Polarisations(epsilon)/2.


class DensityMatrixObservables3(DensityMatrixObservables):
    """
        Class reprensenting density matrices of a single qutrit (massive spin-1 particle).
        It is a child of the generic class and thus inherits its methods.
        The basis is the one of the helicities given by MadGraph: (+1, 0, -1).
        The density matrix is rho = 1/3 + P_a lambda_a, with lambda_a the Gell-Mann matrices and P_a = <lambda_a>/2,
        the same normalisation as DensityMatrixObservables33.
    """
    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """The initialisation is the same as the parent class because it needs to accept the same input"""
        super().__init__(user_input, len_user_input)
        if self.density_matrix_dimension != 3:
            raise ValueError("The dimension of a qutrit density matrix should be 3.")

    def Get_Polarisations(self, epsilon=1e-10) -> list[float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Polarisation vector P_a = Tr(lambda_a rho)/2 (8 components)
        """
        Polarisation = np.zeros(8, dtype=float)
        for i in range(8):
            Polarisation[i] = np.real(np.trace(np.dot(Lambda[i], self.square_matrix())))/2.
            if np.abs(Polarisation[i]) < epsilon:
                Polarisation[i] = 0.
        return Polarisation

    def spin_expectation(self, epsilon=1e-10) -> list[float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter
            Output: Spin expectation value <J_i>, same relations to P_a as in DensityMatrixObservables33
        """
        Polarisation = self.Get_Polarisations(epsilon)
        spin_exp = np.zeros(3, dtype=float)
        spin_exp[0] = np.sqrt(2) * (Polarisation[0] + Polarisation[5])
        spin_exp[1] = np.sqrt(2) * (Polarisation[1] + Polarisation[6])
        spin_exp[2] = Polarisation[2] + np.sqrt(3) * Polarisation[7]
        return spin_exp

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
            Output: Polarisation vectors of the two particles of the system,
                    B1_i = Tr(rho sigma_i x I), B2_i = Tr(rho I x sigma_i)
        """
        Polarisation1 = np.zeros(3, dtype=float)
        Polarisation2 = np.zeros(3, dtype=float)
        for i in range(3):
            tensorprod1 = np.kron(sigma[i], Identity2)
            tensorprod2 = np.kron(Identity2, sigma[i])
            Polarisation1[i] = np.real(np.trace(np.dot(tensorprod1, self.square_matrix())))
            Polarisation2[i] = np.real(np.trace(np.dot(tensorprod2, self.square_matrix())))

            if np.abs(Polarisation1[i]) < epsilon:
                Polarisation1[i] = 0.
            if np.abs(Polarisation2[i]) < epsilon:
                Polarisation2[i] = 0.

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

    def Get_Discord(self, method="quick", maxiter=100) -> float:
        """
        Algorithm based on formula (3) from [2209.03969]. It computes Discord for a given density matrix rho.
        Input: self -> density matrix
               method -> "quick" or "basinhopper" for local or global minimisation 
               maxiter -> maximum number of iterations for the minimisation
        Output: float -> Discord
        """

        Srho = self.Von_Neumann_entropy() # S(rho)
        rhoB = self.partial_trace(0, ['fermion', 'fermion'])
        SrhoB = self.Von_Neumann_entropy(rho=rhoB) #S(rho_B)

        B_plus, B_minus = self.Get_Polarisations()
        Corr, _ = self.Get_Correlations()

        def proba_n(n: list[float]) -> float: #B- = B2
            return (1 + np.dot(B_minus, n)) / 2
        def Bn_plus(n: list[float]) -> list[complex]:
            return (np.array(B_plus) + np.dot(Corr,n)) / (1 + np.dot(B_minus, n))
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
        if method == "quick": #gradient-descent based algorithm. Quick but can be stuck in local minima
            from scipy.optimize import minimize
            result = minimize(objective_function, x0, constraints=constraints, options={'maxiter': maxiter})
        elif method == "basinhopper": #global minimiser, slower but is less sensitive to non-convex minimisation
            from scipy.optimize import basinhopping
            kwargs = {"constraints": constraints, "options": {'maxiter': maxiter}}
            result = basinhopping(objective_function, x0, minimizer_kwargs=kwargs)
        else:
            raise ValueError("Argument method must be 'quick' or 'basinhopper'")

        min_result = result.fun
        Discord = SrhoB - Srho + min_result
        
        return Discord.real

    def Get_Steering(self, nmeas=2) -> float:
        """
        Computation of the steering of a 2 qubits system based on the formula of https://arxiv.org/pdf/1510.08030v2
        Input:  nmeas -> number of measurements per site (2 or 3)
        Output: steering
        """
        s = np.linalg.svd(self.Get_Correlations()[0], compute_uv=False) # singular values, positive and in decreasing order
        if nmeas == 2:
            Lambda_nmeas = np.hypot(s[0], s[1]) # the two largest singular values
        elif nmeas == 3:
            Lambda_nmeas = np.linalg.norm(s)
        else:
            raise ValueError("The steering computation is only available for 2 or 3 measurements per site")

        return max(0., (Lambda_nmeas - 1)/(np.sqrt(nmeas) - 1))

    def Get_Local_Normal_Form(self, tol=1e-8) -> dict:
        """
        Transforms the density matrix with a local unitary U x V such that its correlation matrix is diagonal (equation 5 of https://arxiv.org/pdf/1510.08030v2).
        It is the form used to derive the steering formula of Get_Steering.
        Input:  tol -> tolerance of the self-checks
        Output: dictionary with the transformed density matrix rho_lnf, the SU(2) matrices U and V, the corresponding SO(3) rotations O_A and O_B,
                the rotated polarisations a = O_A.a and b = O_B.b, and the diagonal correlation matrix c
        """
        W, s, ZT = np.linalg.svd(self.Get_Correlations()[0])
        Z = ZT.T
        S = np.diag(s)

        # W and Z are elements of O(3): if the determinant is -1, the sign of the last column is absorbed in S to have rotations
        if np.linalg.det(W) < 0:
            W[:, 2] *= -1
            S[2, 2] *= -1
        if np.linalg.det(Z) < 0:
            Z[:, 2] *= -1
            S[2, 2] *= -1

        O_A, O_B = W.T, Z.T
        U, V = su2_from_so3(O_A, tol), su2_from_so3(O_B, tol)
        K = np.kron(U, V)
        rho_lnf = K @ np.asarray(self.square_matrix()) @ K.conj().T
        rho_lnf = 0.5 * (rho_lnf + rho_lnf.conj().T)   # kill numerical asymmetry

        # self-check: the correlation matrix of rho_lnf is the diagonal matrix from the SVD
        correlation_lnf = np.array([[np.trace(rho_lnf @ np.kron(si, sj)).real for sj in sigma] for si in sigma])
        if not np.allclose(S, correlation_lnf, atol=tol):
            raise ValueError(f"The check on the values of the correlation matrix has not passed:\n{correlation_lnf}\nexpected\n{S}")

        a, b = self.Get_Polarisations()
        return {"rho_lnf": rho_lnf,
                "U": U, "V": V,
                "O_A": O_A, "O_B": O_B,
                "a": O_A @ a, "b": O_B @ b,
                "c": S}


class DensityMatrixObservables23(DensityMatrixObservables):
    """
        Class reprensenting density matrices of qubit-qutrit systems.
        It is a child of the generic class and thus inherits its methods.
        We define here the observables specific to qubit-qutrit systems.
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
            Output: Polarisation vectors of the two particles of the system,
                    B_i = Tr(rho_qubit sigma_i) for the qubit and P_a = Tr(rho_qutrit lambda_a)/2 for the qutrit
        """
        if particle_type[0] == 'boson':
            Polarisation1 = np.zeros(8, dtype=float)
            Polarisation2 = np.zeros(3, dtype=float)

            for i in range(8):
                tensorprod1 = np.kron(Lambda[i], Identity2)
                Polarisation1[i] = np.real(np.trace(np.dot(tensorprod1, self.square_matrix())))/2.
                if np.abs(Polarisation1[i]) < epsilon:
                    Polarisation1[i] = 0.

            for j in range(3):
                tensorprod2 = np.kron(Identity3, sigma[j])
                Polarisation2[j] = np.real(np.trace(np.dot(tensorprod2, self.square_matrix())))
                if np.abs(Polarisation2[j]) < epsilon:
                    Polarisation2[j] = 0.

        elif particle_type[0] == 'fermion':
            Polarisation1 = np.zeros(3, dtype=float)
            Polarisation2 = np.zeros(8, dtype=float)

            for i in range(3):
                tensorprod1 = np.kron(sigma[i], Identity3)
                Polarisation1[i] = np.real(np.trace(np.dot(tensorprod1, self.square_matrix())))
                if np.abs(Polarisation1[i]) < epsilon:
                    Polarisation1[i] = 0.

            for j in range(8):
                tensorprod2 = np.kron(Identity2, Lambda[j])
                Polarisation2[j] = np.real(np.trace(np.dot(tensorprod2, self.square_matrix())))/2.
                if np.abs(Polarisation2[j]) < epsilon:
                    Polarisation2[j] = 0.
        else:
            raise ValueError("particle_type only accepts elements that are 'fermion' or 'boson'.")

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
            spin_exp2 = np.array(Polarisation2)/2.
        elif particle_type[0] == 'fermion':
            spin_exp1 = np.array(Polarisation1)/2.
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
        Class reprensenting density matrices of qutrit-qutrit systems.
        It is a child of the generic class and thus inherits its methods.
        We define here the observables specific to qutrit-qutrit systems.
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
            Output: Polarisation vectors of the two particles of the system,
                    P1_a = Tr(rho lambda_a x I)/2, P2_a = Tr(rho I x lambda_a)/2, i.e. rho_1 = 1/3 + P1_a lambda_a
        """
        Polarisation1 = np.zeros(8, dtype=float)
        Polarisation2 = np.zeros(8, dtype=float)
        for i in range(8):
            tensorprod1 = np.kron(Lambda[i], Identity3)
            tensorprod2 = np.kron(Identity3, Lambda[i])
            Polarisation1[i] = np.real(np.trace(np.dot(tensorprod1, self.square_matrix())))/2.
            Polarisation2[i] = np.real(np.trace(np.dot(tensorprod2, self.square_matrix())))/2.

            if np.abs(Polarisation1[i]) < epsilon:
                Polarisation1[i] = 0.
            if np.abs(Polarisation2[i]) < epsilon:
                Polarisation2[i] = 0.

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

        RhoA = self.partial_trace(1, ['boson', 'boson'])
        RhoB = self.partial_trace(0, ['boson', 'boson'])

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

        RhoA = self.partial_trace(1, ['boson', 'boson'])
        RhoB = self.partial_trace(0, ['boson', 'boson'])

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


class DensityMatrixObservables222(DensityMatrixObservables):
    """
        Class reprensenting density matrices of qubit-qubit-qubit systems.
        It is a child of the generic class and thus inherits its methods.
        We define here the observables specific to qubit-qubit-qubit systems.
    """

    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """The initialisation is the same as the parent class because it needs to accept the same input"""
        super().__init__(user_input, len_user_input)
        if self.density_matrix_dimension != 8:
            raise ValueError("The dimension of a qubit-qubit-qubit density matrix should be 8.")

    def operator_expectation(self, operators:list, rho=None, epsilon=1e-10) -> float:
        """
            Input:  operators -> list of the three 2x2 matrices acting on each qubit (Pauli matrices or Identity2)
                    rho -> density matrix in square form (computed from self if not given)
                    epsilon -> threshold parameter
            Output: Tr(rho . operators[0] x operators[1] x operators[2]), set to 0 if smaller than epsilon

            The result is real since rho and the Pauli strings are hermitian. An error is raised if the imaginary
            part is bigger than epsilon, as this means the density matrix is not hermitian.
        """
        if rho is None:
            rho = self.square_matrix()
        observable = np.kron(np.kron(operators[0], operators[1]), operators[2])
        expectation = np.trace(observable @ rho)

        if abs(expectation.imag) > epsilon:
            raise ValueError(f"The expectation value of a Pauli string has an imaginary part bigger than {epsilon}. The density matrix is probably not hermitian, please check.")
        if abs(expectation.real) < epsilon:
            return 0.
        return expectation.real

    def Get_Polarisations(self, epsilon=1e-10) -> tuple[list[float], list[float], list[float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter, values smaller than it are set to 0
            Output: Polarisation vectors (B1, B2, B3) of the three particles, each a real array of size 3
                    with B1_i = Tr(rho sigma_i x I x I), B2_i = Tr(rho I x sigma_i x I), B3_i = Tr(rho I x I x sigma_i)
        """
        Polarisation1, Polarisation2, Polarisation3 = np.zeros(3), np.zeros(3), np.zeros(3)
        rho = self.square_matrix()

        for i in range(3):
            Polarisation1[i] = self.operator_expectation([sigma[i], Identity2, Identity2], rho, epsilon)
            Polarisation2[i] = self.operator_expectation([Identity2, sigma[i], Identity2], rho, epsilon)
            Polarisation3[i] = self.operator_expectation([Identity2, Identity2, sigma[i]], rho, epsilon)

        return Polarisation1, Polarisation2, Polarisation3

    def Get_2particles_correlations(self, epsilon=1e-10) -> tuple[list[float, float], list[float, float], list[float, float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter, values smaller than it are set to 0
            Output: Two-qubit correlation matrices (C12, C13, C23), each a real 3x3 array
                    with C12_ij = Tr(rho sigma_i x sigma_j x I), C13_ij = Tr(rho sigma_i x I x sigma_j), C23_ij = Tr(rho I x sigma_i x sigma_j)
                    The first index always refers to the particle that comes first in the density matrix.
        """
        Correlations12, Correlations13, Correlations23 = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
        rho = self.square_matrix()

        for i in range(3):
            for j in range(3):
                Correlations12[i][j] = self.operator_expectation([sigma[i], sigma[j], Identity2], rho, epsilon)
                Correlations13[i][j] = self.operator_expectation([sigma[i], Identity2, sigma[j]], rho, epsilon)
                Correlations23[i][j] = self.operator_expectation([Identity2, sigma[i], sigma[j]], rho, epsilon)

        return Correlations12, Correlations13, Correlations23

    def Get_3particles_correlations(self, epsilon=1e-10) -> list[float, float, float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter, values smaller than it are set to 0
            Output: Three-qubit correlation tensor D, a real 3x3x3 array with D_ijk = Tr(rho sigma_i x sigma_j x sigma_k)
        """
        Correlations123 = np.zeros((3, 3, 3))
        rho = self.square_matrix()

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    Correlations123[i][j][k] = self.operator_expectation([sigma[i], sigma[j], sigma[k]], rho, epsilon)

        return Correlations123


class DensityMatrixObservables223(DensityMatrixObservables):
    """
        Class reprensenting density matrices of qubit-qubit-qutrit systems.
        It is a child of the generic class and thus inherits its methods.
        We define here the observables specific to qubit-qubit-qutrit systems.
        The two qubits are assumed to be the first two particles of the density matrix and the qutrit the last one.
        As for the qubit-qutrit and qutrit-qutrit classes, each Gell-Mann matrix in an expectation value comes with a factor 1/2.
    """

    def __init__(self, user_input:Union[list[complex, complex], list[complex], str], len_user_input=None) -> None:
        """The initialisation is the same as the parent class because it needs to accept the same input"""
        super().__init__(user_input, len_user_input)
        if self.density_matrix_dimension != 12:
            raise ValueError("The dimension of a qubit-qubit-qutrit density matrix should be 12.")

    def operator_expectation(self, operators:list, rho=None, epsilon=1e-10) -> float:
        """
            Input:  operators -> list of the three matrices acting on each particle (2x2 Pauli matrices or Identity2 for the qubits,
                                 3x3 Gell-Mann matrices or Identity3 for the qutrit)
                    rho -> density matrix in square form (computed from self if not given)
                    epsilon -> threshold parameter
            Output: Tr(rho . operators[0] x operators[1] x operators[2]), set to 0 if smaller than epsilon

            The result is real since rho and the operators are hermitian. An error is raised if the imaginary
            part is bigger than epsilon, as this means the density matrix is not hermitian.
        """
        if rho is None:
            rho = self.square_matrix()
        observable = np.kron(np.kron(operators[0], operators[1]), operators[2])
        expectation = np.trace(observable @ rho)

        if abs(expectation.imag) > epsilon:
            raise ValueError(f"The expectation value of an operator has an imaginary part bigger than {epsilon}. The density matrix is probably not hermitian, please check.")
        if abs(expectation.real) < epsilon:
            return 0.
        return expectation.real

    def Get_Polarisations(self, epsilon=1e-10) -> tuple[list[float], list[float], list[float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter, values smaller than it are set to 0
            Output: Polarisation vectors (B1, B2, B3) of the three particles, B1 and B2 real arrays of size 3, B3 a real array of size 8
                    with B1_i = Tr(rho sigma_i x I x I), B2_i = Tr(rho I x sigma_i x I), B3_a = Tr(rho I x I x lambda_a)/2
        """
        Polarisation1, Polarisation2, Polarisation3 = np.zeros(3), np.zeros(3), np.zeros(8)
        rho = self.square_matrix()

        for i in range(3):
            Polarisation1[i] = self.operator_expectation([sigma[i], Identity2, Identity3], rho, epsilon)
            Polarisation2[i] = self.operator_expectation([Identity2, sigma[i], Identity3], rho, epsilon)
        for a in range(8):
            Polarisation3[a] = self.operator_expectation([Identity2, Identity2, Lambda[a]], rho, epsilon)/2.

        return Polarisation1, Polarisation2, Polarisation3

    def Get_2particles_correlations(self, epsilon=1e-10) -> tuple[list[float, float], list[float, float], list[float, float]]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter, values smaller than it are set to 0
            Output: Two-particle correlation matrices (C12, C13, C23), C12 a real 3x3 array, C13 and C23 real 3x8 arrays
                    with C12_ij = Tr(rho sigma_i x sigma_j x I), C13_ia = Tr(rho sigma_i x I x lambda_a)/2, C23_ia = Tr(rho I x sigma_i x lambda_a)/2
                    The first index always refers to the particle that comes first in the density matrix.
        """
        Correlations12, Correlations13, Correlations23 = np.zeros((3, 3)), np.zeros((3, 8)), np.zeros((3, 8))
        rho = self.square_matrix()

        for i in range(3):
            for j in range(3):
                Correlations12[i][j] = self.operator_expectation([sigma[i], sigma[j], Identity3], rho, epsilon)
            for a in range(8):
                Correlations13[i][a] = self.operator_expectation([sigma[i], Identity2, Lambda[a]], rho, epsilon)/2.
                Correlations23[i][a] = self.operator_expectation([Identity2, sigma[i], Lambda[a]], rho, epsilon)/2.

        return Correlations12, Correlations13, Correlations23

    def Get_3particles_correlations(self, epsilon=1e-10) -> list[float, float, float]:
        """
            Input:  self -> density matrix
                    epsilon -> threshold parameter, values smaller than it are set to 0
            Output: Three-particle correlation tensor D, a real 3x3x8 array with D_ija = Tr(rho sigma_i x sigma_j x lambda_a)/2
        """
        Correlations123 = np.zeros((3, 3, 8))
        rho = self.square_matrix()

        for i in range(3):
            for j in range(3):
                for a in range(8):
                    Correlations123[i][j][a] = self.operator_expectation([sigma[i], sigma[j], Lambda[a]], rho, epsilon)/2.

        return Correlations123