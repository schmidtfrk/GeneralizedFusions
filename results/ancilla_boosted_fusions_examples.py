from QPC_Optimizer import *

#one ancilla photon
single_ancilla_photon_fusion = MatrixQPCFusion((1, 1), np.array([1]))
single_ancilla_photon_fusion.apply_beam_splitter((single_ancilla_photon_fusion.give_qpc_mode_label(1, 1, 1, 0), single_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 1)))
single_ancilla_photon_fusion.apply_beam_splitter((single_ancilla_photon_fusion.give_qpc_mode_label(1, 1, 1, 1), single_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 0)))
single_ancilla_photon_fusion.apply_dft_splitter((single_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 0), single_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 1),
                                                 single_ancilla_photon_fusion.give_ancilla_mode_label(1)))
single_ancilla_photon_fusion.measure()
single_ancilla_photon_fusion.select_entangling_measurement_patterns()
single_photon_eff = single_ancilla_photon_fusion.probability_of_selected_events()
print('The obtainable generalized fusion efficiency with one single ancilla photons is '+str(single_photon_eff)+'.')


#two ancilla photons
two_ancilla_photon_fusion = MatrixQPCFusion((1, 1), np.array([1, 1]))
two_ancilla_photon_fusion.apply_beam_splitter((two_ancilla_photon_fusion.give_qpc_mode_label(1, 1, 1, 0), two_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 1)))
two_ancilla_photon_fusion.apply_beam_splitter((two_ancilla_photon_fusion.give_qpc_mode_label(1, 1, 1, 1), two_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 0)))
two_ancilla_photon_fusion.apply_dft_splitter((two_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 0), two_ancilla_photon_fusion.give_qpc_mode_label(2, 1, 1, 1),
                                              two_ancilla_photon_fusion.give_ancilla_mode_label(1)))
two_ancilla_photon_fusion.apply_dft_splitter(
    (two_ancilla_photon_fusion.give_ancilla_mode_label(2), two_ancilla_photon_fusion.give_qpc_mode_label(1, 1, 1, 0),
     two_ancilla_photon_fusion.give_qpc_mode_label(1, 1, 1, 1)))
two_ancilla_photon_fusion.measure()
two_ancilla_photon_fusion.select_entangling_measurement_patterns()
two_photon_eff = two_ancilla_photon_fusion.probability_of_selected_events()

print('The obtainable generalized fusion efficiency with two single ancilla photons is '+str(two_photon_eff)+'.')

#three ancilla photons
three_ancilla_photon_fusion= MatrixQPCFusion((1, 1), (1, 1, 1))
three_ancilla_photon_fusion.apply_beam_splitter((0, 3))
three_ancilla_photon_fusion.apply_beam_splitter((1, 2))
three_ancilla_photon_fusion.apply_dft_splitter((1, 3, 4))
three_ancilla_photon_fusion.apply_beam_splitter((5, 6))
three_ancilla_photon_fusion.apply_beam_splitter((0, 5))
three_ancilla_photon_fusion.apply_beam_splitter((2, 6))
three_ancilla_photon_fusion.measure()
three_ancilla_photon_fusion.select_entangling_measurement_patterns()
three_photon_eff = three_ancilla_photon_fusion.probability_of_selected_events()
print('The obtainable generalized fusion efficiency with three single ancilla photons is '+str(three_photon_eff)+'.')

#four ancilla photons
four_ancilla_photon_fusion= MatrixQPCFusion((1, 1), (1, 1, 1, 1))
four_ancilla_photon_fusion.apply_beam_splitter((0,2))
four_ancilla_photon_fusion.apply_beam_splitter((1,3))
four_ancilla_photon_fusion.apply_beam_splitter((4,5))
four_ancilla_photon_fusion.apply_beam_splitter((6,7))
four_ancilla_photon_fusion.apply_beam_splitter((0,4))
four_ancilla_photon_fusion.apply_beam_splitter((1,5))
four_ancilla_photon_fusion.apply_beam_splitter((2,6))
four_ancilla_photon_fusion.apply_beam_splitter((3,7))
four_ancilla_photon_fusion.measure()
four_ancilla_photon_fusion.select_entangling_measurement_patterns()
four_photon_eff = four_ancilla_photon_fusion.probability_of_selected_events()
print('The obtainable generalized fusion efficiency with four single ancilla photons is '+str(four_photon_eff)+'.')