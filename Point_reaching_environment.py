import time

import numpy as np
import math
import tensorflow as tf
from tensorflow.python.platform import gfile
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from matplotlib import style
import matplotlib as mpl
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from scipy.io import loadmat


class Custom_point_reaching(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(Custom_point_reaching, self).__init__()
        self.action_space_n = 9
        self.observation_space_n = 12

        self.action_space = spaces.Box(
            low=np.array([-0.2] * self.action_space_n),
            high=np.array([0.2] * self.action_space_n),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * self.observation_space_n),
            high=np.array([1.0] * self.observation_space_n),
            dtype=np.float32
        )
        self.time_tracker = np.float64(0.0)
        self.current_step = 0
        self.viewer = None

        # Load normalization parameters
        file_name = "LSTM_model/normalization_parameters_dict.mat"
        normalization_dict = loadmat(file_name)
        self.total_act_min = np.squeeze(normalization_dict["actuations_min"])
        self.total_act_max = np.squeeze(normalization_dict["actuations_max"])
        self.total_frame_min = np.squeeze(normalization_dict["frame_min"])
        self.total_frame_max = np.squeeze(normalization_dict["frame_max"])

        # Load trained robot model
        # Load the frozen FDM model for faster predictions
        model_location = "LSTM_model/Bellow_Dynamics_Graph.pb"
        with gfile.GFile(model_location, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read(-1))

        with tf.Graph().as_default() as graph1:
            tf.import_graph_def(graph_def)

        self.sess1 = tf.compat.v1.Session(graph=graph1)
        self.FKM_op_tensor = self.sess1.graph.get_tensor_by_name('import/Identity:0')

        self.target_goal = np.asarray([-50.0, -80.0, -455.0])

        self.target_goal_n = self.normalize_it(
            input_data=self.target_goal, total_p_min=self.total_frame_min[6:], total_p_max=self.total_frame_max[6:]
        )

        self.past_inst_tau = None
        self.cur_inst_pos = None
        self.past_inst_pos = None

        self.episode_length = 100
        self.global_rendering_cell = []
        self.global_actions_cell = []

        self.global_count = 0
        self.rendering_counter = 1

    def Threshold_check(self, current_action):
        new_action = np.round(self.past_inst_tau + current_action, 4)
        new_action = np.clip(new_action, -1.0, 1.0)
        return new_action

    def compute_reward(self, achieved_goal, desired_goal, _info):
        distance_penalty = np.linalg.norm(achieved_goal - desired_goal)
        reward = -distance_penalty

        # per-step penalty
        reward += -2.0
        return float(reward)

    def step(self, action):
        new_action = self.Threshold_check(action)

        tau_pos_history = (np.hstack([
            self.past_inst_pos,
            self.cur_inst_pos,
            self.past_inst_tau,
            new_action
        ])).reshape(1, 1, 36)

        state_transition_n = np.squeeze(self.sess1.run(self.FKM_op_tensor, {'import/x:0': tau_pos_history}))
        state_transition_un = self.un_normalize_it(
            state_transition_n, total_p_min=self.total_frame_min, total_p_max=self.total_frame_max
        )
        self.global_rendering_cell.append([state_transition_un])
        self.global_actions_cell.append([new_action])

        # variable re-setting for next step
        self.past_inst_pos = self.cur_inst_pos.copy()
        self.cur_inst_pos = state_transition_un.copy()
        self.past_inst_tau = new_action.copy()

        # Before reward function definition, un-normalize the state
        robot_ee = state_transition_un[6:]

        reward = 0
        done = False

        # If NaN is detected then episode needs to be terminated
        invalid_values_condition = np.isnan(state_transition_n)
        if invalid_values_condition.any():
            done = True
            print('This pos was observed: ', state_transition_n)
        else:
            reward = self.compute_reward(
                achieved_goal=robot_ee, desired_goal=self.target_goal, _info=None
            )

            achieved_distance = abs(reward + 2.0)
            if achieved_distance <= 5.0:
                reward = (10.0 - achieved_distance) * 2.0
                done = True

            if self.current_step == self.episode_length - 1:  # Episode length
                done = True   # ran the whole episode without success

        self.current_step += 1
        self.time_tracker += time.time()
        self.global_count += 1
        # if self.global_count % 200000 == 0:
        #     self.render()
        if done:
            self.render()

        local_error_n = (state_transition_n[6:] - self.target_goal_n) / 2.0
        observation = np.concatenate([state_transition_n, local_error_n])
        # print('new observation: ', observation)
        return observation, reward, done, {"ctime": self.time_tracker}

    def reset(self):
        # initialize the robot i.e., resting position of the robot
        initial_state_un = self.get_initial_state()
        robot_ee = initial_state_un[6:]
        initial_state_n = self.normalize_it(
            initial_state_un, total_p_min=self.total_frame_min, total_p_max=self.total_frame_max
        )

        self.current_step = 0
        self.past_inst_pos = initial_state_n.copy()
        self.cur_inst_pos = initial_state_n.copy()
        self.past_inst_tau = np.asarray([-1.0] * self.action_space_n)

        self.time_tracker = np.float64(0.0)
        self.global_rendering_cell = []
        self.global_actions_cell = []
        self.global_rendering_cell.append([initial_state_un])
        self.global_actions_cell.append([self.past_inst_tau])

        local_error_n = (initial_state_n[6:] - self.target_goal_n) / 2.0
        observation = np.concatenate([initial_state_n, local_error_n])
        # print('at reset: ', observation)
        return observation

    def get_initial_state(self):
        EE1_temp = np.asarray([np.random.uniform(-6.5, -4.0),
                               np.random.uniform(-2.0, 2.0),
                               np.random.uniform(-146.0, -145.0)])
        EE2_temp = np.asarray([np.random.uniform(-20.0, -17.0),
                               np.random.uniform(-15.0, -12.0),
                               np.random.uniform(-281.0, -279.0)])
        EE3_temp = np.asarray([np.random.uniform(-15.0, -10.0),
                               np.random.uniform(-22.0, -17.0),
                               np.random.uniform(-396.0, -393.5)])

        init_state = np.concatenate([EE1_temp, EE2_temp, EE3_temp])
        init_state = np.round(init_state, 4)

        return init_state

    def normalize_it(self, input_data, total_p_min, total_p_max, scaling_factor=1):
        if input_data.ndim == 1:
            input_array = np.zeros((1, len(input_data)))
            for ind in range(len(input_data)):
                input_array[0, ind] = 1 + ((2 / (total_p_max[ind] - total_p_min[ind])) *
                                           (input_data[ind] - total_p_max[ind]))
                input_array[0, ind] = input_array[0, ind] * scaling_factor

            input_array = np.squeeze(input_array)

        elif input_data.ndim == 2:
            input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
            for ind in range(input_data.shape[1]):
                input_array[:, ind] = 1 + (
                        (2 / (total_p_max[ind] - total_p_min[ind])) * (input_data[:, ind] - total_p_max[ind]))
                input_array[:, ind] = input_array[:, ind] * scaling_factor

        else:
            raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

        return input_array

    def un_normalize_it(self, input_data, total_p_min, total_p_max, scaling_factor=1):
        if input_data.ndim == 1:
            input_array = np.zeros((1, len(input_data)))
            for ind in range(len(input_data)):
                input_data[ind] = input_data[ind] / scaling_factor
                input_array[0, ind] = total_p_max[ind] + (
                        (input_data[ind] - 1) * (total_p_max[ind] - total_p_min[ind]) / 2)

            input_array = np.squeeze(input_array)

        elif input_data.ndim == 2:
            input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
            for ind in range(input_data.shape[1]):
                input_data[:, ind] = input_data[:, ind] / scaling_factor
                input_array[:, ind] = total_p_max[ind] + ((input_data[:, ind] - 1) *
                                                          (total_p_max[ind] - total_p_min[ind]) / 2)

        else:
            raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

        return input_array

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        # Ensure interactive backend is set
        mpl.use('TkAgg')
        from matplotlib import animation

        print("Rendering the training outcome....")
        # Ensure correct shapes
        self.global_rendering_cell = np.array(self.global_rendering_cell).reshape(-1, 3, 3)
        self.global_actions_cell = np.array(self.global_actions_cell).reshape(-1, 9)

        history_len = self.global_rendering_cell.shape[0]
        print('Rendering cell size: ', self.global_rendering_cell.shape)

        # 1. PRE-CALCULATE Data and Limits
        full_rewards = []
        full_rmses = []
        for j in range(history_len):
            dist = np.linalg.norm(self.target_goal - self.global_rendering_cell[j, 2, :])
            full_rmses.append(np.sqrt(dist))
            full_rewards.append(self.compute_reward(self.target_goal, self.global_rendering_cell[j, 2, :], _info=None))

        full_rewards = np.array(full_rewards)
        full_rmses = np.array(full_rmses)

        # 2. Setup Figure and GridSpec (6 rows, 3 columns)
        fig1 = plt.figure(figsize=(18, 12))
        gs = fig1.add_gridspec(6, 3)

        # Left side: Reward, RMSE, and 3D Robot
        ax_reward = fig1.add_subplot(gs[0:2, 0])
        ax_RMSE = fig1.add_subplot(gs[0:2, 1])
        ax1 = fig1.add_subplot(gs[2:, 0:2], projection='3d')

        ax1.scatter(self.target_goal[0], self.target_goal[1], self.target_goal[2], marker='*', c='m', s=100)

        # Right side: 3 Action subplots
        ax_act_plots = [
            fig1.add_subplot(gs[0:2, 2]),  # Actions 1-3
            fig1.add_subplot(gs[2:4, 2]),  # Actions 4-6
            fig1.add_subplot(gs[4:6, 2])  # Actions 7-9
        ]

        # 3. Initialize Line Objects (Static Plotting)
        # 3D Robot & Traces
        robot1, = ax1.plot3D([], [], [], 'yo-', lw=4, label="Robot")
        trace1, = ax1.plot3D([], [], [], 'r-', lw=1, alpha=0.3)
        trace2, = ax1.plot3D([], [], [], 'g-', lw=1, alpha=0.3)
        trace3, = ax1.plot3D([], [], [], 'b-', lw=1, alpha=0.3)

        # Reward & RMSE
        line_reward, = ax_reward.plot([], [], 'r-', lw=2)
        line_rmse, = ax_RMSE.plot([], [], 'b-', lw=2)

        # 9 Actions (3 per subplot)
        action_lines = []
        colors = ['red', 'green', 'blue']  # Colors for the 3 lines in each plot
        for p_idx, ax in enumerate(ax_act_plots):
            ax.set_xlim([0, history_len])
            # Set Y limits based on the data in this group
            start, end = p_idx * 3, (p_idx + 1) * 3
            data_min = np.min(self.global_actions_cell[:, start:end])
            data_max = np.max(self.global_actions_cell[:, start:end])
            ax.set_ylim([data_min - 0.1, data_max + 0.1])
            ax.set_title(f"Actions {start + 1}-{end}")

            for l_idx in range(3):
                ln, = ax.plot([], [], color=colors[l_idx], lw=1.5, label=f"Act {start + l_idx + 1}")
                action_lines.append(ln)

        # Set labels and static limits for main plots
        ax1.set_xlim3d([-200, 200]);
        ax1.set_ylim3d([-200, 200]);
        ax1.set_zlim3d([-490, 10])
        ax_RMSE.set_xlim([0, history_len]);
        ax_RMSE.set_ylim([0, np.max(full_rmses) * 1.1])
        ax_reward.set_xlim([0, history_len]);
        ax_reward.set_ylim([np.min(full_rewards) - 1, np.max(full_rewards) + 1])
        ax_RMSE.set_title("RMSE Over Time");
        ax_reward.set_title("Reward Over Time")

        def animate(i):
            t_axis = np.arange(i)

            # Update 3D Robot
            px = [0.0] + list(self.global_rendering_cell[i, :, 0])
            py = [0.0] + list(self.global_rendering_cell[i, :, 1])
            pz = [0.0] + list(self.global_rendering_cell[i, :, 2])
            robot1.set_data(px, py)
            robot1.set_3d_properties(pz)

            # Update 3D Traces
            trace1.set_data(self.global_rendering_cell[:i, 0, 0], self.global_rendering_cell[:i, 0, 1])
            trace1.set_3d_properties(self.global_rendering_cell[:i, 0, 2])
            trace2.set_data(self.global_rendering_cell[:i, 1, 0], self.global_rendering_cell[:i, 1, 1])
            trace2.set_3d_properties(self.global_rendering_cell[:i, 1, 2])
            trace3.set_data(self.global_rendering_cell[:i, 2, 0], self.global_rendering_cell[:i, 2, 1])
            trace3.set_3d_properties(self.global_rendering_cell[:i, 2, 2])

            # Update Reward/RMSE
            line_reward.set_data(t_axis, full_rewards[:i])
            line_rmse.set_data(t_axis, full_rmses[:i])

            # Update all 9 Actions
            for idx in range(9):
                action_lines[idx].set_data(t_axis, self.global_actions_cell[:i, idx])

            return [robot1, trace1, trace2, trace3, line_reward, line_rmse] + action_lines

        # Run Animation
        ani = animation.FuncAnimation(fig1, animate, frames=history_len, interval=50, blit=True)
        plt.tight_layout()
        plt.show()
        return

