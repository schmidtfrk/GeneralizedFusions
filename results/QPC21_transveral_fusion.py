from FusionSimulator import *
QPC_simulator = MatrixQPCFusion((2,1),(1,1))
QPC_simulator.apply_beam_splitter((QPC_simulator.give_qpc_mode_label(1, 1, 1, 0), QPC_simulator.give_qpc_mode_label(2, 1, 1, 1)))
QPC_simulator.apply_beam_splitter((QPC_simulator.give_qpc_mode_label(1, 1, 1, 1), QPC_simulator.give_qpc_mode_label(2, 1, 1, 0)))
QPC_simulator.apply_dft_splitter((QPC_simulator.give_qpc_mode_label(2, 1, 1, 0), QPC_simulator.give_qpc_mode_label(2, 1, 1, 1),
                                  QPC_simulator.give_ancilla_mode_label(1)))
QPC_simulator.apply_beam_splitter((QPC_simulator.give_qpc_mode_label(1, 2, 1, 0), QPC_simulator.give_qpc_mode_label(2, 2, 1, 1)))
QPC_simulator.apply_beam_splitter((QPC_simulator.give_qpc_mode_label(1, 2, 1, 1), QPC_simulator.give_qpc_mode_label(2, 2, 1, 0)))
QPC_simulator.apply_dft_splitter((QPC_simulator.give_qpc_mode_label(2, 2, 1, 0), QPC_simulator.give_qpc_mode_label(2, 2, 1, 1),
                                  QPC_simulator.give_ancilla_mode_label(2)))
QPC_simulator.measure()
QPC_simulator.select_entangling_measurement_patterns()
print(QPC_simulator.probability_of_selected_events())