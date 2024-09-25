from sympy import *
import numpy as np
from numpy.linalg import norm
import scipy
from scipy import linalg as la
from scipy.special import factorial
from qutip import *
import time
import warnings
"""
Simulator Class

subclasses for both engines, in engine there is only the code to go from the input and circuit to 
the measurement outcomes, everything else goes into the class



"""


def _is_state_bell_state_up_to_phase( state, tolerance=10 ** -10):
    state = _remove_global_phase(state)
    non_zero_indices = np.nonzero(state)
    relative_phase = np.angle(state[non_zero_indices[-1]])
    state[non_zero_indices[-1]] *= np.exp(-1j * relative_phase)
    return _is_state_bell_state(state, tolerance)


def _is_state_bell_state( state, tolerance=10 ** -6):
    phi_plus = np.array([1, 0, 0, 0], dtype=complex)
    phi_minus = np.array([0, 1, 0, 0], dtype=complex)
    psi_plus = np.array([0, 0, 1, 0], dtype=complex)
    psi_minus = np.array([0, 0, 0, 1], dtype=complex)
    state_in_bell_basis = computational_to_Bell_basis(state)
    state_in_bell_basis = _remove_global_phase(state_in_bell_basis)
    minimal_state_distance = min([np.linalg.norm(state_in_bell_basis - bell_state) for bell_state in
                                  (phi_plus, phi_minus, psi_plus, psi_minus)])
    return minimal_state_distance < tolerance


def _reduced_density_matrix(state):
    return np.array([[np.absolute(state[0]) ** 2 + np.absolute(state[2]) ** 2,
                      state[0] * np.conjugate(state[1]) + state[2] * np.conjugate(state[3])],
                     [state[1] * np.conjugate(state[0]) + state[3] * np.conjugate(state[2]),
                      np.absolute(state[1]) ** 2 + np.absolute(state[3]) ** 2]])


def _is_state_maximally_entangled( state, tolerance=10 ** -10):
    return np.abs(1 - _entanglement_entropy(state)) < tolerance


def _entanglement_entropy( state):
    reduced_density_matrix = _reduced_density_matrix(state)
    return min(max(calculate_entropy(reduced_density_matrix), 0.), 1.)


def _is_state_partially_entangled(state, threshold):
    return _entanglement_entropy(state) > threshold

def _remove_global_phase( state, tolerance=14):
    state_without_small_numbers = np.around(state, decimals=tolerance)
    first_vector_with_maximum = np.argmax(np.abs(state_without_small_numbers))
    global_phase = np.angle(state_without_small_numbers[first_vector_with_maximum])
    state_without_globalphase = state_without_small_numbers * np.exp(-1j * global_phase)
    state_without_globalphase = np.around(state_without_globalphase, decimals=tolerance)
    return state_without_globalphase

def _Bellness(state, order):
    if order == np.inf:
        return min(1,max(np.abs(computational_to_Bell_basis(state))**order))
    else:
        return min(1, np.sum(np.abs(computational_to_Bell_basis(state))**order))

def computational_to_Bell_basis(state):
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    phi_minus = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    transformation =(np.outer(phi_plus, np.array([1,0,0,0], dtype=complex))
                    + np.outer(phi_minus, np.array([0,1,0,0], dtype=complex))
                    + np.outer(psi_plus, np.array([0,0,1,0], dtype=complex))
                    + np.outer(psi_minus, np.array([0,0,0,1], dtype=complex)))
    return np.linalg.inv(transformation) @ state

def Bell_to_computational_basis(state):
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    phi_minus = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    transformation =(np.outer(phi_plus, np.array([1,0,0,0], dtype=complex))
                    + np.outer(phi_minus, np.array([0,1,0,0], dtype=complex))
                    + np.outer(psi_plus, np.array([0,0,1,0], dtype=complex))
                    + np.outer(psi_minus, np.array([0,0,0,1], dtype=complex)))
    return transformation @ state

def is_phase_matrix(matrix, decimals=10):
    dimension = len(matrix)
    matrix = np.around(np.abs(matrix), decimals=decimals)
    return np.array_equal(matrix, np.eye(dimension))


def is_unitary(matrix, tolerance =10**-10):
    dims = np.shape(matrix)
    if len(dims) != 2:
        return False
    if dims[0] != dims[1]:
        return False
    return (np.linalg.norm(matrix@ matrix.conj().T -np.eye(dims[0]))<tolerance and
            np.linalg.norm(matrix.conj().T@matrix -np.eye(dims[0]))<tolerance)


def unitary_from_clements(number_of_modes, coefficients, order= None):
    n=number_of_modes
    def odd_layer_even_modes(coeffs):
        matrix = np.eye(number_of_modes, dtype=complex)
        for i in range(int(number_of_modes/2)):
            matrix = T_matrix(number_of_modes,2*i, 2*i+1, coeffs[2*i], coeffs[2*i+1]) @ matrix
        return matrix

    def even_layer_even_modes(coeffs):
        matrix = np.eye(number_of_modes, dtype=complex)
        for i in range(int(number_of_modes / 2) - 1):
            matrix = T_matrix(number_of_modes,2 * i + 1, 2 * i + 2, coeffs[2 * i], coeffs[2 * i + 1]) @ matrix
        return matrix

    def odd_layer_odd_modes(coeffs):
        matrix = np.eye(number_of_modes, dtype=complex)
        for i in range(int((number_of_modes-1)/2)):
            matrix = T_matrix(number_of_modes,2*i, 2*i+1, coeffs[2*i], coeffs[2*i+1]) @ matrix
        return matrix

    def even_layer_odd_modes(coeffs):
        matrix = np.eye(number_of_modes, dtype=complex)
        for i in range(int((number_of_modes-1 )/ 2)):
            matrix = T_matrix(number_of_modes, 2 * i + 1, 2 * i + 2, coeffs[2 * i], coeffs[2 * i + 1]) @ matrix
        return matrix

    matrix = np.eye(number_of_modes, dtype=complex)
    if len(coefficients) != number_of_modes**2-number_of_modes:
        raise Exception('Invalid Dimensions: Expected vector of length' +
                        str(number_of_modes**2-number_of_modes))
    if order is not None:
        for i in range(len(order)):
            matrix= T_matrix(number_of_modes,order[i][0]-1,order[i][1]-1, coefficients[2 * i], coefficients[2 * i + 1]) @ matrix
    else:
        if number_of_modes%2==0:
            for i in range(int(number_of_modes/2)):
                matrix = odd_layer_even_modes(coefficients[i*(2*n-2):n+ i*(2*n-2)]) @ matrix
                matrix = even_layer_even_modes(coefficients[n + i*(2*n-2):2*n-2 +i*(2*n-2)]) @ matrix
        elif number_of_modes%2==1:
            for i in range(int(number_of_modes/2)):
                matrix = odd_layer_odd_modes(coefficients[2*(n-1)*i:n-1+2*(n-1)*i]) @ matrix
                matrix = even_layer_odd_modes(coefficients[n-1+2*(n-1)*i:2*(n-1)+2*(n-1)*i]) @ matrix
            matrix = odd_layer_odd_modes(coefficients[-(n-1):]) @ matrix
    return matrix


def complex_into_re_and_im(number):
    return (np.real(number), np.imag(number))

def get_matrix_diagonal_elements(matrix):
    return [matrix[i,i] for i in range(len(matrix))]
def _nullify_off_diagonal_matrix_elements(unitary, tolerance =10 ** -13, decimals = 10):
    dimension = np.shape(unitary)[0]
    coeffs  = []
    coeffs_inv = []
    for i in range(1, dimension):
        if i % 2 == 1:
            for j in range(0, i):
                nullify = lambda vars: complex_into_re_and_im(
                    (unitary @ np.linalg.inv(T_matrix(dimension, i - j - 1, i - j + 1 - 1, *vars)
                                             ))[dimension - j - 1, i - j - 1])
                optimizer = scipy.optimize.root(nullify, (0., 0.), tol=tolerance)
                vars = optimizer.x % (2 * np.pi)
                if not optimizer.success:
                    warnings.warn('Mimizer did not converge.')
                unitary = unitary @ np.linalg.inv(T_matrix(dimension, i - j - 1, i - j + 1 - 1, *vars))
                coeffs_inv.append([(i - j, i - j + 1), vars])
        else:
            for j in range(1, i + 1):
                nullify = lambda vars: complex_into_re_and_im(
                    (T_matrix(dimension, dimension + j - i - 2, dimension + j - i - 1, *vars) @ unitary)
                    [dimension + j - i - 1, j - 1])
                optimizer = scipy.optimize.root(nullify, (0., 0.), tol=tolerance)
                vars = optimizer.x % (2 * np.pi)
                if not optimizer.success:
                    warnings.warn('Mimizer did not converge.')
                unitary = T_matrix(dimension, dimension + j - i - 2, dimension + j - i - 1, *vars) @ unitary
                coeffs.append([(dimension + j - i - 1, dimension + j - i), vars])
    diagonal_matrix = unitary
    if not is_phase_matrix(unitary, decimals=decimals):
        raise Exception('Nullifying of Matrix failed.')

    return coeffs,coeffs_inv, diagonal_matrix

def _transform_nullify_outcome_into_clements_coefficients(coeffs, coeffs_inv, diagonal):
    order =[coeff[0] for coeff in coeffs_inv]
    final_coeffs = [coeff[1][j] for coeff in coeffs_inv for j in (0, 1)]
    for coeff in reversed(coeffs):
        order.append(coeff[0])
        psi1 = np.angle(diagonal[coeff[0][0] - 1])
        psi2 = np.angle(diagonal[coeff[0][1] - 1])
        phi = coeff[1][0]
        psi1prime = psi2 - phi + np.pi
        phiprime = psi1 - psi2 - np.pi
        diagonal[coeff[0][0] - 1] = np.exp(1j * psi1prime)
        final_coeffs.append(phiprime)
        final_coeffs.append(coeff[1][1])
    return final_coeffs, order

def unitary_to_clements_coeffs(unitary,unitary_tolerance=10**-10, decimals= 10):
    if is_unitary(unitary, unitary_tolerance) == False:
        raise ValueError('Input matrix is not unitary.')
    coeffs, coeffs_inv, diagonal_matrix = _nullify_off_diagonal_matrix_elements(unitary, decimals=decimals)
    diagonal = get_matrix_diagonal_elements(diagonal_matrix)
    return _transform_nullify_outcome_into_clements_coefficients(coeffs, coeffs_inv, diagonal, )


def T_matrix(dimension, i, j, phi, theta):
    matrix = np.eye(dimension, dtype=complex)
    matrix[i, i] = np.exp(1j*phi)*np.cos(theta)
    matrix[i, j] = -sin(theta)
    matrix[j, i] = np.exp(1j * phi) * sin(theta)
    matrix[j, j] = cos(theta)
    return matrix

def precompute_creation_operators(qpcn, qpcm, ancilla):
    number_of_ancilla_modes = len(ancilla)
    ancilla_photon_number = sum(ancilla)
    number_of_photons = 2 * qpcn * qpcm + ancilla_photon_number
    number_of_modes = 4 * qpcn * qpcm + number_of_ancilla_modes
    hilbert_space = [1 + number_of_photons]*number_of_modes
    creation_operators = []
    identity_qubits = identity([2, 2])
    creation_operators_optical = enr_destroy(hilbert_space, excitations=number_of_photons)
    for operator in creation_operators_optical:
        creation_operators.append(tensor(identity_qubits, operator).dag())
    return creation_operators
def calculate_entropy(rho):
    EV = la.eigvals(rho)

    # Drop zero eigenvalues so that log2 is defined
    my_list = [x for x in EV.tolist() if x]
    EV = np.array(my_list)

    log2_EV = np.matrix(np.log2(EV))
    EV = np.matrix(EV)
    S = np.real(-np.dot(EV, log2_EV.H))[0,0]
    return(S)

def multiply_matrix_times_vector_of_matrices(matrix, vector):
    if len(vector) != len(matrix[0]):
        raise TypeError('Invalid Dimensions: dimension of vector '+str(len(vector))
                        +' does not match dimension of matrix '+str(len(matrix[0])))
    results = []
    for i in range(len(matrix)):
        init_matrix = matrix[i,0] * vector[0]
        for j in range(1, len(matrix[0])):
            init_matrix += matrix[i, j] * vector[j]
        results.append(init_matrix)
    return results
class QPCFusion:

    def __init__(self, qpc_parameter, ancilla_photon_distribution, engine='sympy'):
        self.qpcn, self.qpcm = qpc_parameter
        self.ancilla = ancilla_photon_distribution
        self.number_of_ancilla_modes = int(len(self.ancilla))
        self.engine = engine
        self.total_number_of_modes = int(self.qpcn*self.qpcm*4+self.number_of_ancilla_modes)
        self.lo_unitary = np.eye(self.total_number_of_modes)
        self.measurement_outcome = {}
        self.successful_patterns = {}


    def give_current_lo_unitary(self):
        return np.linalg.inv(self.lo_unitary)

    def apply_unitary(self, unitary):
        if np.shape(unitary) != np.shape(self.lo_unitary):
            raise ValueError('Dimension of the unitary do not match.')
        else:
            self.lo_unitary = self.lo_unitary @ np.linalg.inv(unitary)

    def give_qpc_mode_label(self, code_label, n_value, m_value, qubit_value):
        return qubit_value + 4 * (m_value - 1) + 4 * self.qpcm * (n_value - 1) +2*(code_label-1)

    def give_ancilla_mode_label(self, ancilla_label):
        return 4*self.qpcn*self.qpcm+ancilla_label-1

    def apply_phase_shifter(self, mode, angle):
        matrix = np.eye(self.total_number_of_modes, dtype='cdouble')
        matrix[mode] = np.exp(1j*angle)
        self.lo_unitary = self.lo_unitary @ np.linalg.inv(matrix)



    def entanglement_of_all_states(self, order):
        result = 0
        for id in self.measurement_outcome:
            result += self.measurement_outcome[id][1]*_entanglement_entropy(self.measurement_outcome[id][0])**order
        return result

    def Bellness_of_all_states(self, order=4):
        result = 0
        for id in self.measurement_outcome:
            result += self.measurement_outcome[id][1] * _Bellness(self.measurement_outcome[id][0], order)
        return result


    def select_partially_entangling_measurement_patterns(self,threshold):
        self.successful_patterns = {}
        for i in self.measurement_outcome:
            if _is_state_partially_entangled(self.measurement_outcome[i][0],threshold):
                self.successful_patterns[i] = (self.measurement_outcome[i])
    def select_entangling_measurement_patterns(self):
        self.successful_patterns = {}
        for i in self.measurement_outcome:
            if _is_state_maximally_entangled(self.measurement_outcome[i][0]):
                self.successful_patterns[i] = (self.measurement_outcome[i])

    def select_bell_measurement_patterns(self, tolerance = 10**-4):
        self.successful_patterns = {}
        for i in self.measurement_outcome:
            if _is_state_bell_state(self.measurement_outcome[i][0], tolerance):
                self.successful_patterns[i] = (self.measurement_outcome[i])

    def select_bell_measurement_up_to_phase_patterns(self, tolerance = 10**-12):
        self.successful_patterns = {}
        for i in self.measurement_outcome:
            if _is_state_bell_state_up_to_phase(self.measurement_outcome[i][0], tolerance):
                self.successful_patterns[i] = (self.measurement_outcome[i])

    def probability_of_selected_events(self):
        probability = 0
        for i in self.successful_patterns:
            probability += self.successful_patterns[i][1]
        return probability


    def sorted_event_outcomes(self, tolerance=14):
        sorted_fusions = {}
        for i in self.successful_patterns:
            state = self.successful_patterns[i][0]
            state_without_globalphase = _remove_global_phase(state, tolerance)
            probability = self.successful_patterns[i][1]
            state_without_globalphase = tuple(state_without_globalphase.tolist())
            if state_without_globalphase not in sorted_fusions:
                sorted_fusions[state_without_globalphase] = [ probability, [i]]
            else:
                sorted_fusions[state_without_globalphase][1].append(i)
                sorted_fusions[state_without_globalphase][0] += probability
        return sorted_fusions
    def apply_beam_splitter(self, modes, angle = np.pi / 4):
        matrix = np.eye(self.total_number_of_modes, dtype='cdouble')
        i, k = modes
        matrix[i, i] = np.cos(angle)
        matrix[i, k] = np.sin(angle)
        matrix[k, i] = np.sin(angle)
        matrix[k, k] = -np.cos(angle)
        self.lo_unitary = self.lo_unitary @ np.linalg.inv(matrix)

    def apply_dft_splitter(self, modes):
        matrix = np.eye(self.total_number_of_modes, dtype='cdouble')
        d = len(modes)
        for k in range(d):
            for i in range(d):
                matrix[modes[k], modes[i]] = 1 / np.sqrt(d) * np.exp(1j * 2 * np.pi / d * k * i)
        self.lo_unitary = self.lo_unitary @ np.linalg.inv(matrix)

    def apply_unitary_from_clements(self, coefficients, order= None):
        unitary = unitary_from_clements(self.total_number_of_modes, coefficients, order=order)
        self.apply_unitary(unitary)


class MatrixQPCFusion(QPCFusion):
    def __init__(self, qpc_parameter, ancilla_photon_distribution, precomputed_creation_operator = None):
        super().__init__(qpc_parameter, ancilla_photon_distribution)
        self.number_of_photons =  2 * self.qpcn * self.qpcm +self._ancilla_photon_number()
        self.number_of_modes =  4 * self.qpcn * self.qpcm + self.number_of_ancilla_modes
        self.hilbert_space = None
        self._generate_hilbert_space()
        self.creation_operator = None
        if precomputed_creation_operator != None :
            self.creation_operator = precomputed_creation_operator
        else:
            self._setting_up_creation_operators()
        self._lookup = enr_state_dictionaries(self.hilbert_space, self.number_of_photons)
        self._transformed_creation_operator = None
        self.state = None



    def _generate_hilbert_space(self):
        hilbert_space_dimension = []
        for _ in range(self.total_number_of_modes):
            hilbert_space_dimension.append(1+self.number_of_photons)
        self.hilbert_space = hilbert_space_dimension
    def _ancilla_photon_number(self):
        return sum(self.ancilla)

    def _dagger_operators(self, operators):
        for i in operators:
            i=i.dag()

    def _generate_vaccum(self):
        ground_state_fusion_modes = basis([2,2])
        ground_state_optical_modes = enr_fock(self.hilbert_space,self.number_of_photons,
                                              [0]*self.total_number_of_modes)
        return tensor(ground_state_fusion_modes, ground_state_optical_modes)
    def _setting_up_creation_operators(self):
        creation_operators = []
        identity_qubits = identity([2,2])
        creation_operators_optical = enr_destroy(self.hilbert_space, excitations=self.number_of_photons)
        for operator in creation_operators_optical:
            creation_operators.append(tensor(identity_qubits, operator).dag())
        self.creation_operator = creation_operators
    def _stateid_to_state(self,id):
        nmax_optical = self._lookup[0]
        optical_state = self._lookup[2][id % nmax_optical]
        qubit_value = id // nmax_optical
        qubit_state = [0]*4
        qubit_state[qubit_value] = 1
        return (qubit_state, optical_state)

    def _transforming_creation_operators(self):
        self._transformed_creation_operator = multiply_matrix_times_vector_of_matrices(self.lo_unitary,
                                                                                       self.creation_operator)


    def _get_state(self):
        def inner_block( code_label, n_value, qubit_value):
            expression = 1
            for i in range(1, self.qpcm + 1):
                expression *= self._transformed_creation_operator[
                    self.give_qpc_mode_label(code_label, n_value, i, qubit_value)]
            return expression
        def outer_block(code_label, sign):
            expression= 1
            for i in range(1,self.qpcn+1):
                expression *= inner_block(code_label,i,0) + sign * inner_block(code_label,i,1)
            return expression

        def consider_ancilla():
            expression = 1
            for i in range(1,len(self.ancilla)+1):
                expression*= self._transformed_creation_operator[self.give_ancilla_mode_label(i)]**self.ancilla[i-1]/np.sqrt(factorial(self.ancilla[i-1]))
            return expression
        self._transforming_creation_operators()
        I = identity([2])
        X = sigmax()
        I_opt =enr_identity(self.hilbert_space, self.number_of_photons)
        state = ((tensor(I, I, I_opt) + tensor(X, I, I_opt)) * outer_block(1, +1)
                 + (tensor(I, I , I_opt) - tensor(X, I, I_opt)) * outer_block(1, -1))

        state *=((tensor(I, I, I_opt) + tensor(I, X, I_opt)) *  outer_block(2, +1)
                 + (tensor(I, I, I_opt) - tensor(I, X, I_opt)) * outer_block(2, -1))
        state *= consider_ancilla()
        state *=1/2**(self.qpcn+2)
        vacuum = self._generate_vaccum()
        self.state = state * vacuum

    def _state_to_unnormalized_dict(self):
        state_non_zero = scipy.sparse.coo_matrix(self.state.data)
        measurement_unnormalized_dict={}
        for row, coefficient in zip(state_non_zero.row, state_non_zero.data):
            qubit_state, optical_state = self._stateid_to_state(row)
            if optical_state not in measurement_unnormalized_dict:
                measurement_unnormalized_dict[optical_state] = np.array([0, 0, 0, 0],dtype=complex)
            measurement_unnormalized_dict[optical_state] += np.array(qubit_state, dtype=complex) * coefficient
        return measurement_unnormalized_dict

    def _normalize_dict(self,measurement_unnormalized_dict):
        normalized_dict = {}
        for optical_modes in measurement_unnormalized_dict:
            state = measurement_unnormalized_dict[optical_modes]
            probability = np.linalg.norm(state)**2
            state = state / np.sqrt(probability)
            normalized_dict[optical_modes] = (state, probability)
        self.measurement_outcome = normalized_dict
    def _photon_counting_measurement(self):
        measurement_unnormalized_dict = self._state_to_unnormalized_dict()
        self._normalize_dict(measurement_unnormalized_dict)

    def measure(self):
        self._get_state()
        self._photon_counting_measurement()


class SymbolicQPCFusion(QPCFusion):
    def __init__(self, qpc_parameter, ancilla_photon_distribution):
        super().__init__(qpc_parameter, ancilla_photon_distribution)
        self.variables = {}
        self.transformed_variables={}
        self.list_of_variables = None
        self.list_of_transformed_variables = None
        self.__setting_up_variables()
        self.polynom = None

    def __setting_up_variables(self):
        for i in ['y10', 'y11', 'y20', 'y21']:
            self.variables[i] = symbols(i)
        for i in range(1,self.qpcn+1):
            for j in range(1,self.qpcm+1):
                self.variables['x1' + str(i) + str(j) + '0'] = symbols('x1' + str(i) + str(j) + '0')
                self.variables['x1' + str(i) + str(j) + '1'] = symbols('x1' + str(i) + str(j) + '1')
                self.variables['x2' + str(i) + str(j) + '0'] = symbols('x2' + str(i) + str(j) + '0')
                self.variables['x2' + str(i) + str(j) + '1'] = symbols('x2' + str(i) + str(j) + '1')
        for i in range(1,self.number_of_ancilla_modes+1):
            self.variables['a' + str(i)] = symbols('a' + str(i))
        self.list_of_variables = np.array([self.variables[i] for i in self.variables])
    def __perform_linear_optical_transformation(self):
        self.list_of_transformed_variables = self.lo_unitary @ self.list_of_variables[4:]
        for i in range(self.qpcn):
            for j in range(self.qpcm):
                self.transformed_variables['x1' + str(i+1) + str(j+1) + '0'] = self.list_of_transformed_variables[0+4*j+4*self.qpcm*i]
                self.transformed_variables['x1' + str(i+1) + str(j+1) + '1'] = self.list_of_transformed_variables[1+4*j+4*self.qpcm*i]
                self.transformed_variables['x2' + str(i+1) + str(j+1) + '0'] = self.list_of_transformed_variables[2+4*j+4*self.qpcm*i]
                self.transformed_variables['x2' + str(i+1) + str(j+1) + '1'] = self.list_of_transformed_variables[3+4*j+4*self.qpcm*i]
        for i in range(1,self.number_of_ancilla_modes+1):
            self.transformed_variables['a'+str(i)] = self.list_of_transformed_variables[4*self.qpcn*self.qpcm+i-1]
    def __polynom_initialization(self):
        temporary_plus_code1 = temporary_plus_code2 = temporary_minus_code1 = temporary_minus_code2 = 1
        self.__perform_linear_optical_transformation()
        for i in range(1, self.qpcn + 1):
            block0_code1 = block1_code1 = block0_code2 = block1_code2 = 1
            for j in range(1, self.qpcm +1):
                block0_code1 *= self.transformed_variables['x1' + str(i) + str(j) + '0']
                block1_code1 *= self.transformed_variables['x1' + str(i) + str(j) + '1']
                block0_code2 *= self.transformed_variables['x2' + str(i) + str(j) + '0']
                block1_code2 *= self.transformed_variables['x2' + str(i) + str(j) + '1']
            temporary_plus_code1 *= (block0_code1 + block1_code1)
            temporary_plus_code2 *= (block0_code2 + block1_code2)
            temporary_minus_code1 *= (block0_code1 - block1_code1)
            temporary_minus_code2 *= (block0_code2 - block1_code2)
        self.polynom= ((self.variables['y10'] + self.variables['y11']) * temporary_plus_code1
                       + (self.variables['y10'] - self.variables['y11']) * temporary_minus_code1)
        self.polynom *=((self.variables['y20'] + self.variables['y21']) * temporary_plus_code2
                        + (self.variables['y20'] - self.variables['y21']) * temporary_minus_code2  )
        for i in range(1,self.number_of_ancilla_modes+1):
            self.polynom *= self.transformed_variables['a'+str(i)]**self.ancilla[i-1]/np.sqrt(factorial(self.ancilla[i-1]))
        self.polynom *=1/2**(self.qpcn+1)/2
        self.polynom = Poly(self.polynom, *self.list_of_variables)

    def __check_polynom_normalization(self):
        coeffs, monoms = self.__convert_polynom_into_coeffs_and_monoms()
        coeffs = self.__sympy_complex_into_numpy_complex_coeffs(coeffs)
        monoms = np.array(monoms)
        probability = 0
        for i in range(len(coeffs)):
            probability += np.prod(factorial(monoms[i])) * np.abs(coeffs[i]) ** 2
        return probability

    def __convert_polynom_into_coeffs_and_monoms(self):
        return self.polynom.coeffs(), self.polynom.monoms()

    def __sympy_complex_into_numpy_complex_coeffs(self, coeffs):
        coeffs_result = np.zeros(len(coeffs), dtype='cdouble')
        for i in range(len(coeffs)):
            coeffs_result[i] = re(coeffs[i]) + im(coeffs[i]) * 1j
        return coeffs_result

    def __give_measurement_patterns(self, coeffs, monoms):
        patterns = {}
        for i in range(len(monoms)):
            measurement_pattern = monoms[i][4:]
            remaining_state = np.array(monoms[i][:4])
            if measurement_pattern not in patterns:
                patterns[measurement_pattern] = self.__remaining_monom_to_numpy_vector(remaining_state) * coeffs[
                    i] * np.sqrt(np.prod(factorial(measurement_pattern)))
            else:
                patterns[measurement_pattern] += self.__remaining_monom_to_numpy_vector(remaining_state) * coeffs[
                    i] * np.sqrt(np.prod(factorial(measurement_pattern)))
        normalized_patterns = {}
        for i in patterns:
            temp_state_storage = patterns[i]
            probability = norm(temp_state_storage) ** 2
            normalized_patterns[i] = (temp_state_storage / np.sqrt(probability), probability)
        return normalized_patterns

    def __remaining_monom_to_numpy_vector(self, remaining_state):
        if np.array_equal(remaining_state, np.array([1, 0, 1, 0])):
            return np.array([1., 0., 0, 0], dtype='cdouble')
        elif np.array_equal(remaining_state, np.array([1, 0, 0, 1])):
            return np.array([0, 1., 0, 0], dtype='cdouble')
        elif np.array_equal(remaining_state, np.array([0, 1, 1, 0])):
            return np.array([0, 0., 1, 0], dtype='cdouble')
        elif np.array_equal(remaining_state, np.array([0, 1, 0, 1])):
            return np.array([0, 0., 0, 1], dtype='cdouble')
        else:
            raise ValueError

    def __give_measurement_patterns(self,coeffs, monoms):
        patterns = {}
        for i in range(len(monoms)):
            measurement_pattern = monoms[i][4:]
            remaining_state = np.array(monoms[i][:4])
            if measurement_pattern not in patterns:
                patterns[measurement_pattern] = self.__remaining_monom_to_numpy_vector(remaining_state) * coeffs[i] * np.sqrt(
                    np.prod(factorial(measurement_pattern)))
            else:
                patterns[measurement_pattern] += self.__remaining_monom_to_numpy_vector(remaining_state) * coeffs[i] * np.sqrt(
                    np.prod(factorial(measurement_pattern)))
        normalized_patterns = {}
        for i in patterns:
            temp_state_storage = patterns[i]
            probability = norm(temp_state_storage) ** 2
            normalized_patterns[i] = (temp_state_storage / np.sqrt(probability), probability)
        return normalized_patterns

    def measure(self):
        self.__polynom_initialization()
        coeffs, monoms = self.__convert_polynom_into_coeffs_and_monoms()
        coeffs = self.__sympy_complex_into_numpy_complex_coeffs(coeffs)
        self.measurement_outcome = self.__give_measurement_patterns(coeffs, monoms)


