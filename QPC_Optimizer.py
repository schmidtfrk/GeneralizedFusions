from scipy.optimize import minimize
from FusionSimulator import *
rng = np.random.default_rng()


def select_best_coeffs_from_simulation_runs(fname_input):
    data = read_all_data_from_file(fname_input)
    probs = []
    coeffs = []
    for i in range(int(len(data) / 2)):
        coeffs.append(data[2 * i])
        probs.append(data[2 * i + 1])
    index_of_best_run = np.argmax(np.array(probs))
    best_coeffs = coeffs[index_of_best_run]
    return best_coeffs


def save_best_matrix_from_simulation_runs(fname_input,fname_output,qpc_parameter=(2,1),ancilla=[]):
    number_of_modes = 4 * qpc_parameter[0] * qpc_parameter[1] + len(ancilla)
    best_coeffs = select_best_coeffs_from_simulation_runs(fname_input)
    co, order = unitary_to_clements_coeffs(np.eye(number_of_modes), decimals=1)
    best_matrix = unitary_from_clements(number_of_modes, best_coeffs, order=order)
    with open(fname_output,"wb") as f:
        np.save(f, best_matrix)

def setting_up_Simulator_with_unitary(unitary,qpc_parameter=(2,1),ancilla=[]):
    psi= MatrixQPCFusion(qpc_parameter,ancilla)
    psi.apply_unitary(unitary)
    psi.measure()
    return psi

def probability_of_partial_entanglement(clements_coeffs,arrangement_order, threshold):
    psi=MatrixQPCFusion((2,1),[],precomputed_creation_operator=precomputed)
    psi.apply_unitary_from_clements(clements_coeffs,order=arrangement_order)
    psi.measure()
    psi.select_partially_entangling_measurement_patterns(threshold)
    return psi.probability_of_selected_events()
def entanglement_of_all_states(clements_coeffs, arrangement_order, polynom_order, precomputed,
                               qpc_parameter=(2,1), ancilla =()):
    psi = MatrixQPCFusion(qpc_parameter, ancilla, precomputed_creation_operator=precomputed)
    psi.apply_unitary_from_clements(clements_coeffs, order= arrangement_order)
    psi.measure()
    return psi.entanglement_of_all_states(order=polynom_order)

def Bellness_of_all_states(clements_coeffs, arrangement_order, polynom_order, precomputed,
                               qpc_parameter=(2,1), ancilla =()):
    psi = MatrixQPCFusion(qpc_parameter, ancilla, precomputed_creation_operator=precomputed)
    psi.apply_unitary_from_clements(clements_coeffs, order= arrangement_order)
    psi.measure()
    return psi.Bellness_of_all_states(order=polynom_order)


def QPC_optimizer_entanglement_of_all_states(clements_coeffs, arrangement_order, precomputed,
                                             qpc_parameter = (2,1), ancilla =[], polynom_order=1):
    target_function = lambda coeffs: (-1) * entanglement_of_all_states(coeffs, arrangement_order,
                                                                       polynom_order, precomputed,
                                                                       qpc_parameter, ancilla)
    optimizer = minimize(target_function, clements_coeffs)
    return (optimizer.x, -optimizer.fun)

def QPC_optimizer_Bellness_of_all_states(clements_coeffs, arrangement_order, precomputed,
                                             qpc_parameter = (2,1), ancilla =[], polynom_order=1):
    target_function = lambda coeffs: (-1) * Bellness_of_all_states(coeffs, arrangement_order,
                                                                       polynom_order, precomputed,
                                                                       qpc_parameter, ancilla)
    optimizer = minimize(target_function, clements_coeffs)
    return (optimizer.x, -optimizer.fun)


def QPC21_complete_optimizer(starting_matrix,sample_size ,sigma ,file_name,polynom_order=1, decimals =8):
    coeffs, arrangement_order = unitary_to_clements_coeffs(starting_matrix, decimals=decimals)
    for _ in range(sample_size):
        clements_coeffs = coeffs + np.random.normal(0, sigma, len(coeffs))
        opt_coeff, opt_entanglement =QPC_optimizer_entanglement_of_all_states(clements_coeffs,
                                                                              arrangement_order,
                                                                              polynom_order=polynom_order)
        append_to_file(file_name,opt_coeff)
        append_to_file(file_name, opt_entanglement)

def read_all_data_from_file(file_name):
    stored_data = []
    reader = True
    try:
        with open(file_name, 'rb') as f:
            while reader:
                try:
                    stored_data.append(np.load(f))
                except ValueError:
                    reader = False
    finally:
        return stored_data
def append_to_file(file_name, data_to_append):
    #todo: make data_to_append iterable such that it also allows
    #for a single argument without iterating over the entries of the array
    stored_data = read_all_data_from_file(file_name)
    with open(file_name, 'wb') as f:
        for data in stored_data:
            np.save(f, data)
        np.save(f, data_to_append)
