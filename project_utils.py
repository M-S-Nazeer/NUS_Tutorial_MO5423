import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation


def data_visualization(positions_data, orientations_data, pressure_data, Data_Animation):
    gradient_colors = ['cyan', 'red', 'darkgreen', 'blue']
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlabel('X - axis')
    ax1.set_ylabel('Y - axis')
    ax1.set_zlabel('Z - axis')
    for segment in range(positions_data.shape[1]):
        ax1.scatter(
            positions_data[:, segment, 0],
            positions_data[:, segment, 1],
            positions_data[:, segment, 2],
            c=gradient_colors[segment], marker='o', s=10
        )

    fig2, axes2 = plt.subplots(3, 3, constrained_layout=True)
    axes2 = axes2.flatten()

    for j in range(pressure_data.shape[1]):
        axes2[j].plot(pressure_data[:, j], 'k.-')

    if Data_Animation:
        animate_data(positions_data, orientations_data, pressure_data, gradient_colors)
    else:
        plt.show()


def animate_data(positions_data, orientations_data, pressure_data, gradient_colors):
    def update_animation(i):
        ax2.cla()

        ax2.set_xlim3d([-200, 200])
        ax2.set_ylim3d([-200, 200])
        ax2.set_zlim3d([-450, 10])

        ax2.set_ylabel('y-axis')
        ax2.set_xlabel('x-axis')
        ax2.set_zlabel('z-axis')

        # Apply the rotating camera view
        degrees_rotation = 0.5
        azimuth_angle = (i * degrees_rotation) % 360
        ax2.view_init(elev=15, azim=azimuth_angle)
        arrow_length = 50

        # for all segments
        for seg in range(positions_data.shape[1]):
            ax2.scatter(
                positions_data[i, seg, 0],
                positions_data[i, seg, 1],
                positions_data[i, seg, 2],
                color=gradient_colors[seg],
                marker='o', s=50, edgecolor='k'
            )
            # quat - to - rotation
            object_rotation = R.from_quat(orientations_data[i, seg, :]).as_matrix()
            ax2.quiver(
                positions_data[i, seg, 0],
                positions_data[i, seg, 1],
                positions_data[i, seg, 2],
                object_rotation[0, 0],
                object_rotation[1, 0],
                object_rotation[2, 0],
                length=arrow_length, color='r'
            )
            ax2.quiver(
                positions_data[i, seg, 0],
                positions_data[i, seg, 1],
                positions_data[i, seg, 2],
                object_rotation[0, 1],
                object_rotation[1, 1],
                object_rotation[2, 1],
                length=arrow_length, color='g'
            )
            ax2.quiver(
                positions_data[i, seg, 0],
                positions_data[i, seg, 1],
                positions_data[i, seg, 2],
                object_rotation[0, 2],
                object_rotation[1, 2],
                object_rotation[2, 2],
                length=arrow_length, color='b'
            )

        x = [0.0] + [positions_data[i, j, 0] for j in range(positions_data.shape[1])]
        y = [0.0] + [positions_data[i, j, 1] for j in range(positions_data.shape[1])]
        z = [0.0] + [positions_data[i, j, 2] for j in range(positions_data.shape[1])]
        ax2.plot3D(x, y, z, 'g.-', linewidth=5)
        ax2.scatter(0.0, 0.0, 0.0, marker='s', c='gray', s=50, edgecolor='k')

        ax_actions[0].plot(i, pressure_data[i, 0], c='r', marker='.', linestyle='--')
        ax_actions[0].plot(i, pressure_data[i, 1], c='g', marker='.', linestyle='--')
        ax_actions[0].plot(i, pressure_data[i, 2], c='b', marker='.', linestyle='--')
        ax_actions[1].plot(i, pressure_data[i, 3], c='r', marker='.', linestyle='--')
        ax_actions[1].plot(i, pressure_data[i, 4], c='g', marker='.', linestyle='--')
        ax_actions[1].plot(i, pressure_data[i, 5], c='b', marker='.', linestyle='--')
        ax_actions[1].plot(i, pressure_data[i, 6], c='k', marker='.', linestyle='--')
        ax_actions[2].plot(i, pressure_data[i, 7], c='g', marker='.', linestyle='--')
        ax_actions[2].plot(i, pressure_data[i, 8], c='b', marker='.', linestyle='--')

    robot_actuation_freq = 4.0  # Hz
    sampling_time = 1 / robot_actuation_freq  # ms
    frame_rate = robot_actuation_freq

    positions_data = positions_data[:400, :, :]
    orientations_data = orientations_data[:400, :, :]

    fig2 = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig2)
    # Grid for the 3D
    ax2 = fig2.add_subplot(gs[:, 0], projection='3d')
    ax_actions = []
    ax_actions.append(fig2.add_subplot(gs[0, 1]))
    ax_actions.append(fig2.add_subplot(gs[1, 1]))
    ax_actions.append(fig2.add_subplot(gs[2, 1]))

    fig2.tight_layout()

    ani = animation.FuncAnimation(
        fig2, update_animation,
        interval=sampling_time,
        frames=positions_data.shape[0])

    plt.show()


def Comparison_animation(original_positions, predicted_positions):
    def update_animation(i):
        ax2.cla()

        ax2.set_xlim3d([-200, 200])
        ax2.set_ylim3d([-200, 200])
        ax2.set_zlim3d([-450, 10])

        ax2.set_ylabel('y-axis')
        ax2.set_xlabel('x-axis')
        ax2.set_zlabel('z-axis')

        # Apply the rotating camera view
        degrees_rotation = 0.5
        azimuth_angle = (i * degrees_rotation) % 360
        ax2.view_init(elev=15, azim=azimuth_angle)

        # for all segments
        for seg in range(original_data.shape[1]):
            ax2.scatter(
                original_data[i, seg, 0],
                original_data[i, seg, 1],
                original_data[i, seg, 2],
                color=gradient_colors[seg],
                marker='o', s=50, edgecolor='k'
            )

            ax2.scatter(
                predicted_data[i, seg, 0],
                predicted_data[i, seg, 1],
                predicted_data[i, seg, 2],
                color=gradient_colors[seg + 3],
                marker='*', s=50, edgecolor='k'
            )

        x_original = [0.0] + [original_data[i, j, 0] for j in range(original_data.shape[1])]
        y_original = [0.0] + [original_data[i, j, 1] for j in range(original_data.shape[1])]
        z_original = [0.0] + [original_data[i, j, 2] for j in range(original_data.shape[1])]
        ax2.plot3D(x_original, y_original, z_original, 'g.-', linewidth=5)

        x_predicted = [0.0] + [predicted_data[i, j, 0] for j in range(predicted_data.shape[1])]
        y_predicted = [0.0] + [predicted_data[i, j, 1] for j in range(predicted_data.shape[1])]
        z_predicted = [0.0] + [predicted_data[i, j, 2] for j in range(predicted_data.shape[1])]
        ax2.plot3D(x_predicted, y_predicted, z_predicted, 'k.-', linewidth=5)

        ax_MSE.scatter(
            i,
            np.linalg.norm(original_data[i, original_data.shape[1] - 1, :] - predicted_data[i, original_data.shape[1] - 1, :]),
            marker='o', s=50, c='b', edgecolor='k', label='MSE'
        )

        ax_MAE.scatter(
            i,
            np.mean(np.abs(original_data[i, original_data.shape[1] - 1, :] - predicted_data[i, original_data.shape[1] - 1, :])),
            marker='o', s=50, c='b', edgecolor='k', label='MAE'
        )

        ax2.scatter(0.0, 0.0, 0.0, marker='s', c='gray', s=50, edgecolor='k')

    robot_actuation_freq = 4.0  # Hz
    sampling_time = (1 / robot_actuation_freq)*1000.0
    gradient_colors = ["r", "g", "b", "y", "m", "k"]
    original_data = original_positions[:400, :]
    original_data = original_data.reshape(original_data.shape[0], 3, 3)
    predicted_data = predicted_positions[:400, :]
    predicted_data = predicted_data.reshape(predicted_data.shape[0], 3, 3)

    fig2 = plt.figure(figsize=(16, 12))
    gs = GridSpec(6, 2, figure=fig2)
    # Grid for the 3D
    ax2 = fig2.add_subplot(gs[2: , :], projection='3d')
    ax_MSE = fig2.add_subplot(gs[:2 , 0])
    ax_MAE = fig2.add_subplot(gs[:2 , 1])
    fig2.tight_layout()
    fig2.legend()
    ani = animation.FuncAnimation(
        fig2, update_animation,
        interval=sampling_time,
        frames=original_data.shape[0])

    plt.show()


def normalize_it(input_data, total_p_min, total_p_max, scaling_factor=1):
    if input_data.ndim == 1:
        input_array = np.zeros((1, len(input_data)))
        for ind in range(len(input_data)):
            input_array[0, ind] = 1 + ((2 / (total_p_max[ind] - total_p_min[ind])) * (input_data[ind] - total_p_max[ind]))
            input_array[0, ind] = input_array[0, ind] * scaling_factor

    elif input_data.ndim == 2:
        input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
        for ind in range(input_data.shape[1]):
            input_array[:, ind] = 1 + ((2 / (total_p_max[ind] - total_p_min[ind])) * (input_data[:, ind] - total_p_max[ind]))
            input_array[:, ind] = input_array[:, ind] * scaling_factor

    else:
        raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

    return input_array


def un_normalize_it(input_data, total_p_min, total_p_max, scaling_factor=1):
    if input_data.ndim == 1:
        input_array = np.zeros((1, len(input_data)))
        for ind in range(len(input_data)):
            input_data[ind] = input_data[ind] / scaling_factor
            input_array[0, ind] = total_p_max[ind] + ((input_data[ind] - 1) * (total_p_max[ind] - total_p_min[ind]) / 2)

    elif input_data.ndim == 2:
        input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
        for ind in range(input_data.shape[1]):
            input_data[:, ind] = input_data[:, ind] / scaling_factor
            input_array[:, ind] = total_p_max[ind] + ((input_data[:, ind] - 1) * (total_p_max[ind] - total_p_min[ind]) / 2)

    else:
        raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

    return input_array


def B10_FKM_time_series_conversion(Positions, Actions):
    input_features, state_transitions = [], []
    next_inst_pos = Positions[2, :]
    current_inst_pos = Positions[1, :]
    past_inst_pos = Positions[0, :]

    current_inst_tau = Actions[1, :]
    past_inst_tau = Actions[0, :]
    for i in range(2, Positions.shape[0] - 1):
        temp_inp_features = np.hstack((past_inst_pos, current_inst_pos, past_inst_tau, current_inst_tau))

        input_features.append([temp_inp_features])  # inputs
        state_transitions.append([next_inst_pos])  # to predict

        past_inst_tau = current_inst_tau.copy()
        current_inst_tau = Actions[i, :]

        past_inst_pos = Positions[i - 1, :]
        current_inst_pos = Positions[i, :]
        next_inst_pos = Positions[i + 1, :]

    input_features = (np.array(input_features)).reshape(len(input_features), (Positions.shape[1] * 2) +
                                                        (Actions.shape[1] * 2))
    state_transitions = (np.array(state_transitions)).reshape(len(state_transitions), Positions.shape[1])
    return input_features, state_transitions


