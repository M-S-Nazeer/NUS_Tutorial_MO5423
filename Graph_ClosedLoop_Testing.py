import tensorflow as tf
from tensorflow.python.platform import gfile
from scipy.io import loadmat
import numpy as np
from project_utils import normalize_it, un_normalize_it, Comparison_animation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------- Robot Dataset ------------------------ #
file_data = "data/Pneumatic_ThreeSegments_4Hz.mat"
loaded_matrices = loadmat(file_data)
segment_tip_positions = np.squeeze(loaded_matrices["positions"])       # in millimeters
segment_tip_quaterions = np.squeeze(loaded_matrices["orientations"])
pneumatic_pressure = np.squeeze(loaded_matrices["actuation_pressure"])

# Remove the origin frame
segment_tip_positions = segment_tip_positions[:, 1:, :]
frames_positions_un = segment_tip_positions.reshape(segment_tip_positions.shape[0], -1)   # flattened --> N x 9

file_name = "LSTM_model/normalization_parameters_dict.mat"
normalization_dict = loadmat(file_name)
total_act_min = np.squeeze(normalization_dict["actuations_min"])
total_act_max = np.squeeze(normalization_dict["actuations_max"])
total_frame_min = np.squeeze(normalization_dict["frame_min"])
total_frame_max = np.squeeze(normalization_dict["frame_max"])

frames_positions_n = normalize_it(frames_positions_un, total_p_min=total_frame_min, total_p_max=total_frame_max)
pneumatic_pressure_n = normalize_it(pneumatic_pressure, total_p_min=total_act_min, total_p_max=total_act_max)

# ---------------------- Closed loop Model or Graph Inference ------------------------ #
# For inference with Keras model
# keras_model_location = "LSTM_model/FDM_LSTM_model_weights.keras"
# model = tf.keras.models.load_model(keras_model_location)
# print('Keras model is loaded for CL testing...')

# For inference with the frozen graph (*.pb file)
model_location = "LSTM_model/Bellow_Dynamics_Graph.pb"
# Load the frozen trained dynamics model for faster predictions
with gfile.GFile(model_location,'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read(-1))

with tf.Graph().as_default() as graph1:
    tf.import_graph_def(graph_def)

sess1 = tf.compat.v1.Session(graph=graph1)
FKM_op_tensor = sess1.graph.get_tensor_by_name('import/Identity:0')

frames_positions_n = frames_positions_n[10:2000, :]
pneumatic_pressure_n = pneumatic_pressure_n[10:2000, :]

original_states, predicted_states = [], []

next_inst_state = frames_positions_n[3, :]
current_inst_state = frames_positions_n[2, :]
past_inst_state = frames_positions_n[1, :]
past_past_inst_state = frames_positions_n[0, :]

current_inst_tau = pneumatic_pressure_n[2, :]     # for comparison
past_inst_tau = pneumatic_pressure_n[1, :]
for i in range(3, frames_positions_n.shape[0] - 1):

    temp_inp_features = np.hstack((past_inst_state, current_inst_state, past_inst_tau, current_inst_tau))

    temp_inp_features = np.array(temp_inp_features).reshape(1, 1, (frames_positions_n.shape[1] * 2) +
                                                            (pneumatic_pressure_n.shape[1] * 2))
    # print("CL features: ", temp_inp_features)
    pred_s = sess1.run(FKM_op_tensor, {'import/x:0': temp_inp_features})
    pred_s = np.squeeze((pred_s).reshape(1, frames_positions_n.shape[1]))

    # val1 = un_normalize_it(input_data=next_inst_state, total_p_max=total_frame_max, total_p_min=total_frame_min)
    # val2 = un_normalize_it(input_data=pred_temp, total_p_max=total_frame_max, total_p_min=total_frame_min)
    # mse_frame = np.linalg.norm(val1 - val2)
    # print("MSE Frame: ", round(mse_frame, 5))
    original_states.append([next_inst_state])
    predicted_states.append([pred_s])

    past_inst_tau = current_inst_tau.copy()
    current_inst_tau = pneumatic_pressure_n[i, :]

    past_past_inst_state = past_inst_state.copy()
    past_inst_state = current_inst_state.copy()
    current_inst_state = pred_s.copy()
    next_inst_state = frames_positions_n[i + 1, :]

original_states = np.array(original_states).reshape(len(original_states), frames_positions_n.shape[1])
predicted_states = np.array(predicted_states).reshape(len(predicted_states), frames_positions_n.shape[1])

# ---------------------- UN-NORMALIZATION ORIGINAL AND PREDICTED LABELS ------------------------- #
original_frame = un_normalize_it(input_data=original_states, total_p_max=total_frame_max, total_p_min=total_frame_min)
predicted_frame = un_normalize_it(input_data=predicted_states, total_p_max=total_frame_max, total_p_min=total_frame_min)

Comparison_animation(original_frame, predicted_frame)

# Calculations and prints r2 score of traing and testing data
# fig_frame, ax_frame = plt.subplots(3, 3, constrained_layout=True)
# ax_frame = ax_frame.flatten()
# for i in range(9):
#     ax_frame[i].plot(original_frame[:, i + 18], 'r.-')
#     ax_frame[i].plot(predicted_frame[:, i + 18], 'b.-')
# plt.show()

labels = ['x3', 'y3', 'z3']
for i in range(6, 9):
    print('R2 score of ' + labels[i - 6] + ' is:\t{}'.format(r2_score(original_frame[:, i], predicted_frame[:, i])))
    print('MAE of ' + labels[i - 6] + 'is:\t{}'.format(mean_absolute_error(original_frame[:, i], predicted_frame[:, i])))
    print('RMSE of ' + labels[i - 6] + 'is:\t{}'.format(np.sqrt(mean_squared_error(original_frame[:, i], predicted_frame[:, i]))))

