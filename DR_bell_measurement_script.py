from QPC_Optimizer import *
from scipy.stats import unitary_group

qpc_parameter = (1,1)
ancilla = ()
number_of_iterations = 3
sample_sizes =(20, 10, 10)
sigmas = (0.5, 0.2, 0.1)
fname_coeffs=['DR_BS_1.npy', 'DR_BS_2.npy', 'DR_BS_3.npy']
fname_final_matrix = 'DR_BS_matrix.npy'
polynom_orders = [8, 20, 50]
#generate initial matrix
DR =MatrixQPCFusion((1,1),())
initial_matrix = unitary_group.rvs(4)

#optional Parameters
decimals=3

#checking that parameters are valid
n,m = qpc_parameter
number_of_modes = 4*n*m+sum(ancilla)
if not number_of_iterations == len(sample_sizes) == len(sigmas) == len(fname_coeffs) == len(polynom_orders):
    raise ValueError
if np.shape(initial_matrix) != (number_of_modes, number_of_modes):
    raise ValueError




creation_operator = precompute_creation_operators(n,m, ancilla)
coeffs, arrangement_order = unitary_to_clements_coeffs(initial_matrix, decimals=decimals)
for iteration in range(number_of_iterations):
    for _ in range(sample_sizes[iteration]):
        clements_coeffs = coeffs + np.random.normal(0, sigmas[iteration], len(coeffs))
        opt_coeff, opt_entanglement = QPC_optimizer_Bellness_of_all_states(clements_coeffs,
                                                                               arrangement_order,
                                                                               creation_operator,
                                                                               qpc_parameter, ancilla,
                                                                               polynom_orders[iteration])
        append_to_file(fname_coeffs[iteration], opt_coeff)
        append_to_file(fname_coeffs[iteration], opt_entanglement)
    coeffs = select_best_coeffs_from_simulation_runs(fname_coeffs[iteration])
save_best_matrix_from_simulation_runs(fname_coeffs[-1], fname_final_matrix, qpc_parameter, ancilla)
best_matrix = read_all_data_from_file(fname_final_matrix)
psi = setting_up_Simulator_with_unitary(best_matrix[0],qpc_parameter, ancilla)
print(psi.Bellness_of_all_states(100))
print(psi.give_current_lo_unitary())
print(np.abs(psi.give_current_lo_unitary()))
print(np.angle(psi.give_current_lo_unitary())/np.pi)
psi.select_bell_measurement_patterns()
print(psi.probability_of_selected_events())
psi.select_partially_entangling_measurement_patterns(0.999)
print(psi.probability_of_selected_events())
