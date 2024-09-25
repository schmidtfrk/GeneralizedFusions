import numpy as np

from QPC_Optimizer import *

qpc_parameter = (1,1)
ancilla = [1,1]
number_of_iterations = 4
sample_sizes =(10,3,3, 3)
sigmas = (0.4, 0.05, 0.02, 0.01)
fname_coeffs=['unitaries/ancilla_photon/DR_ancilla11_entanglement_convergence_test_1.npy',
              'unitaries/ancilla_photon/DR_ancilla11_entanglement_convergence_test_2.npy',
              'unitaries/ancilla_photon/DR_ancilla11_entanglement_convergence_test_3.npy',
              'unitaries/ancilla_photon/DR_ancilla11_entanglement_convergence_test_4.npy']
fname_final_matrix = 'unitaries/ancilla_photon/DR_ancilla_11_entanglement_matrix_convergence_test_1.npy'
polynom_orders = [2, 5, 10, 20]

#generate initial matrix
n,m = qpc_parameter
number_of_modes = 4*n*m+len(ancilla)
from scipy.stats import unitary_group
#matrix1=np.eye(number_of_modes,dtype=complex)
#foo = read_all_data_from_file('unitaries/ancilla_photon/DR_ancilla111_entanglement_matrix_1.npy')[0]
#matrix1[0:7,0:7]=foo
#initial_matrix =matrix1
initial_matrix = matrix1 = unitary_group.rvs(number_of_modes)

#optional Parameters
decimals=3

#checking that parameters are valid

if not number_of_iterations == len(sample_sizes) == len(sigmas) == len(fname_coeffs) == len(polynom_orders):
    raise ValueError
if np.shape(initial_matrix) != (number_of_modes, number_of_modes):
    raise ValueError




creation_operator = precompute_creation_operators(n,m, ancilla)
coeffs, arrangement_order = unitary_to_clements_coeffs(initial_matrix, decimals=decimals)
for iteration in range(number_of_iterations):
    print('iteration:'+str(1+iteration))
    for _ in range(sample_sizes[iteration]):
        clements_coeffs = coeffs + np.random.normal(0, sigmas[iteration], len(coeffs))
        opt_coeff, opt_entanglement = QPC_optimizer_entanglement_of_all_states(clements_coeffs,
                                                                               arrangement_order,
                                                                               creation_operator,
                                                                               qpc_parameter, ancilla,
                                                                               polynom_orders[iteration])
        append_to_file(fname_coeffs[iteration], opt_coeff)
        append_to_file(fname_coeffs[iteration], opt_entanglement)
        print(opt_entanglement)
    coeffs = select_best_coeffs_from_simulation_runs(fname_coeffs[iteration])
save_best_matrix_from_simulation_runs(fname_coeffs[-1], fname_final_matrix, qpc_parameter, ancilla)
best_matrix = read_all_data_from_file(fname_final_matrix)
psi = setting_up_Simulator_with_unitary(best_matrix[0],qpc_parameter, ancilla)
print(psi.entanglement_of_all_states(100))
print(psi.give_current_lo_unitary())
print('___________________________')
print(np.abs(psi.give_current_lo_unitary()))
