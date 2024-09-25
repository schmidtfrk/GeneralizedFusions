from QPC_Optimizer import *

qpc_parameter = (2,1)
ancilla = [1]
number_of_iterations = 4
sample_sizes =[10, 1, 1, 1]
sigmas = [0.3, 0.2, 0.1, 0.05]
fname_coeffs=['unitaries/ancilla_photon/qpc21_ancilla1_bell_1.npy',
              'unitaries/ancilla_photon/qpc21_ancilla1_bell_2.npy',
              'unitaries/ancilla_photon/qpc21_ancilla1_bell_3.npy',
              'unitaries/ancilla_photon/qpc21_ancilla1_bell_4.npy']
fname_final_matrix = 'unitaries/ancilla_photon/qpc21_ancilla1_bell_matrix.npy'
polynom_orders = [15, 30, 60, 100]
#generate initial matrix
n,m = qpc_parameter
number_of_modes = 4*n*m+len(ancilla)
from scipy.stats import unitary_group
initial_matrix = matrix1 = unitary_group.rvs(number_of_modes)
psi = setting_up_Simulator_with_unitary(initial_matrix,qpc_parameter, ancilla)
co, order =unitary_to_clements_coeffs(np.eye(number_of_modes))
print('initial Bellness :'+str(psi.Bellness_of_all_states(polynom_orders[0])))


#optional Parameters
decimals=3

#checking that parameters are valid
if not number_of_iterations == len(sample_sizes) == len(sigmas) == len(fname_coeffs) == len(polynom_orders):
    raise ValueError('Parameters do not fit.')
if np.shape(initial_matrix) != (number_of_modes, number_of_modes):
    raise ValueError('Matrix dimension does not fit')




creation_operator = precompute_creation_operators(n,m, ancilla)
coeffs, arrangement_order = unitary_to_clements_coeffs(initial_matrix, decimals=decimals)
for iteration in range(number_of_iterations):
    print('interation:'+str(iteration+1))
    for _ in range(sample_sizes[iteration]):
        clements_coeffs = coeffs + np.random.normal(0, sigmas[iteration], len(coeffs))
        opt_coeff, opt_entanglement = QPC_optimizer_Bellness_of_all_states(clements_coeffs,
                                                                               arrangement_order,
                                                                               creation_operator,
                                                                               qpc_parameter, ancilla,
                                                                               polynom_orders[iteration])
        append_to_file(fname_coeffs[iteration], opt_coeff)
        append_to_file(fname_coeffs[iteration], opt_entanglement)
        psi = MatrixQPCFusion(qpc_parameter,ancilla)
        psi.apply_unitary_from_clements(opt_coeff, order)
        psi.measure()
        print(opt_entanglement, psi.Bellness_of_all_states(polynom_orders[iteration]))
save_best_matrix_from_simulation_runs(fname_coeffs[-1], fname_final_matrix, qpc_parameter, ancilla)
best_matrix = read_all_data_from_file(fname_final_matrix)
psi = setting_up_Simulator_with_unitary(best_matrix[0])
print(psi.Bellness_of_all_states(100))