from QPC_Optimizer import *

fname='unitaries/QPC/qpc21_random_entanglement_matrix_final.npy'
QPC_matrix= read_all_data_from_file(fname)[0]

QPC21_gen_fusion= MatrixQPCFusion((2, 1), [])
QPC21_gen_fusion.apply_unitary(QPC_matrix)
QPC21_gen_fusion.measure()
QPC21_gen_fusion.select_entangling_measurement_patterns()
QPC_eff = QPC21_gen_fusion.probability_of_selected_events()
print('The optimal generalized fusion efficiency of QPC(2,1) is given by '+str(QPC_eff)+'.')
