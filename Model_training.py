
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from project_utils import (normalize_it, un_normalize_it, B10_FKM_time_series_conversion,
                           data_visualization, Comparison_animation)

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ---------------------- Robot Dataset ------------------------ #
file_data = "data/Pneumatic_ThreeSegments_4Hz.mat"
loaded_matrices = loadmat(file_data)
segment_tip_positions = np.squeeze(loaded_matrices["positions"])       # in millimeters
segment_tip_quaterions = np.squeeze(loaded_matrices["orientations"])
pneumatic_pressure = np.squeeze(loaded_matrices["actuation_pressure"])

# Data shapes
print("Robot positions shape: ", segment_tip_positions.shape)
print("Robot quaternions shape: ", segment_tip_quaterions.shape)
print("Robot actuations shape: ", pneumatic_pressure.shape)

# ---------------------- Data Visualization and Animation ------------------------ #
Data_Animation = True
data_visualization(positions_data=segment_tip_positions,
                   orientations_data=segment_tip_quaterions,
                   pressure_data=pneumatic_pressure,
                   Data_Animation=Data_Animation
                   )

# Remove the origin frame
segment_tip_positions = segment_tip_positions[:, 1:, :]
frames_positions_un = segment_tip_positions.reshape(segment_tip_positions.shape[0], -1)   # flattened --> N x 9
segment_tip_quaterions = segment_tip_quaterions[:, 1:, :]

# ---------------------- Data Normalization for Training ------------------------ #
model_directory = "LSTM_model/"
file_name = model_directory + "normalization_parameters_dict.mat"

print("calculating normalization parameters...")
total_frame_max, total_frame_min = np.zeros(frames_positions_un.shape[1]), np.zeros(frames_positions_un.shape[1])
for i in range(frames_positions_un.shape[1]):
    total_frame_min[i] = round(min(frames_positions_un[:, i]), 5)
    total_frame_max[i] = round(max(frames_positions_un[:, i]), 5)

total_frame_min, total_frame_max = np.array(total_frame_min), np.array(total_frame_max)

total_act_max, total_act_min = np.zeros(pneumatic_pressure.shape[1]), np.zeros(pneumatic_pressure.shape[1])
for i in range(pneumatic_pressure.shape[1]):
    total_act_min[i] = round(min(pneumatic_pressure[:, i]), 5)
    total_act_max[i] = round(max(pneumatic_pressure[:, i]), 5)

total_act_min, total_act_max = np.array(total_act_min), np.array(total_act_max)

savemat(file_name, {'actuations_min': total_act_min, "actuations_max": total_act_max,
                    "frame_min": total_frame_min, "frame_max": total_frame_max})


frames_positions_n = normalize_it(frames_positions_un, total_p_min=total_frame_min, total_p_max=total_frame_max)
pneumatic_pressure_n = normalize_it(pneumatic_pressure, total_p_min=total_act_min, total_p_max=total_act_max)

MODEL_TO_GRAPH = True
ANN_file_name = model_directory + "FDM_LSTM_model_weights.keras"

# ------------------------------- Data Pre-processing FOR TRAINING ----------------------------------- #
input_features, output_transitions = B10_FKM_time_series_conversion(Positions=frames_positions_n,
                                                                    Actions=pneumatic_pressure_n)

# Split the data into training and testing dataset
train_input, test_input, train_output, test_output = train_test_split(
    input_features, output_transitions, test_size=0.3, shuffle=True
)
print('Data is successfully split into training and testing data set.')

# Reshape the data into None x 1 x n_inputs and None x 1 X n_outputs
train_input = np.asarray(train_input).reshape(train_input.shape[0], 1, input_features.shape[1])
train_output = np.asarray(train_output).reshape(train_output.shape[0], 1, output_transitions.shape[1])
test_input = np.asarray(test_input).reshape(test_input.shape[0], 1, input_features.shape[1])
test_output = np.asarray(test_output).reshape(test_output.shape[0], 1, output_transitions.shape[1])
print('Data is reshaped.')

def build_model():
    model = Sequential()

    model.add(LSTM(units=256, input_shape=(None, train_input.shape[2]),
                   activation="tanh", recurrent_activation=None, return_sequences=True))
    model.add(LSTM(units=256, activation="tanh", recurrent_activation=None, return_sequences=True))
    model.add(layers.Dropout(rate=0.25))
    model.add(Dense(train_output.shape[2], activation="tanh"))

    learning_rate = 0.001
    model.compile(Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=["mean_absolute_error"])
    return model

BATCH_SIZE = 25
EPOCHS = 150
model = build_model()

my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath=model_directory + 'Bellow_robot.keras', save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=model_directory + 'logs')
]

history = model.fit(
    train_input, train_output,
    batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS,
    shuffle=True, verbose=2, callbacks=my_callbacks
)

# Saving the trained model
model.save(ANN_file_name)
print('Training is done and model is saved...')

# Training history plots
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

plt.figure(1)
plt.plot(loss_values, 'bo-', label='training loss')
plt.plot(val_loss_values, 'r.:', label='training loss val')
plt.show()

print("==" * 20)
print("==" * 8 + "Openloop testing" + "==" * 8)
print("==" * 20)

# train_label_pred = model.predict(train_data, batch_size=30, verbose=1)
test_output_pred = model.predict(test_input, verbose=1)

# Inverse transform of the data
test_output = test_output.reshape(test_output.shape[0], int(test_output.shape[2]))
test_output_pred = test_output_pred.reshape(test_output.shape[0], int(test_output.shape[1]))

# ---------------------- UN-NORMALIZATION ORIGINAL and PREDICTED LABELS ------------------------- #
test_output_un = un_normalize_it(
    input_data=test_output,
    total_p_min=total_frame_min, total_p_max=total_frame_max
)

test_output_pred_un = un_normalize_it(
    input_data=test_output_pred,
    total_p_min=total_frame_min,total_p_max=total_frame_max
)

Comparison_animation(test_output_un, test_output_pred_un)

# fig_frame, ax_frame = plt.subplots(3, 1, constrained_layout=True)
labels = ['x3', 'y3', 'z3']
for i in range(6, 9):
    print('R2 score of ' + labels[i - 6] + ' is:\t{}'.format(r2_score(test_output_un[:, i], test_output_pred_un[:, i])))
    print('MAE of ' + labels[i - 6] + 'is:\t{}'.format(mean_absolute_error(test_output_un[:, i], test_output_pred_un[:, i])))
    print('MSE of ' + labels[i - 6] + 'is:\t{}'.format(mean_squared_error(test_output_un[:, i], test_output_pred_un[:, i])))

    # ax_frame[i - 6].plot(test_output_un[:, i], 'r.-')
    # ax_frame[i - 6].plot(test_output_pred_un[:, i], 'b.-')

# plt.show()

if MODEL_TO_GRAPH:
    from tensorflow import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    print('outputs: ', model.outputs)
    print('inputs: ', model.inputs)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=model_directory,
                      name=f"Bellow_Dynamics_Graph.pb",
                      as_text=False)
    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=model_directory,
                      name=f"Bellow_Dynamics_Graph.pbtxt",
                      as_text=True)


