#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision squares around it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
import copy
import random
import math
import curses
import time

# 3rd party modules
import gym
import numpy as np
from gym import spaces


class PredatorCaptureEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self, args):
        self.__version__ = "0.0.1"
        self.start_time = time.time()
        # TODO: better config handling
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.PREDATOR_CAPTURE_CLASS = 0
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.ON_PREY_BUT_NOT_CAPTURE_REWARD = -0.025
        self.episode_over = False
        self.action_blind = True
        self.episode_eval_counter = 0

        self.args = args

        self.multi_agent_init(self.args)


    # def init_curses(self):
    #     self.stdscr = curses.initscr()
    #     curses.start_color()
    #     curses.use_default_colors()
    #     curses.init_pair(1, curses.COLOR_RED, -1)
    #     curses.init_pair(2, curses.COLOR_YELLOW, -1)
    #     curses.init_pair(3, curses.COLOR_CYAN, -1)
    #     curses.init_pair(4, curses.COLOR_GREEN, -1)
    #     curses.init_pair(5, curses.COLOR_WHITE, -1)
    #     curses.init_pair(6, curses.COLOR_BLUE, -1)

    # def init_args(self, parser):
    #     env = parser.add_argument_group('Prey Predator task')
    #     env.add_argument('--nenemies', type=int, default=1,
    #                      help="Total number of preys in play")
    #     env.add_argument('--dim', type=int, default=5,
    #                      help="Dimension of box")
    #     env.add_argument('--vision', type=int, default=2,
    #                      help="Vision of predator")
    #     env.add_argument('--moving_prey', action="store_true", default=False,
    #                      help="Whether prey is fixed or moving")
    #     env.add_argument('--no_stay', action="store_true", default=False,
    #                      help="Whether predators have an action to stay in place")
    #     parser.add_argument('--mode', default='mixed', type=str,
    #                     help='cooperative|competitive|mixed (default: mixed)')
    #     env.add_argument('--enemy_comm', action="store_true", default=False,
    #                      help="Whether prey can communicate.")
    #     env.add_argument('--nfriendly_P', type=int, default=2,
    #                         help="Total number of friendly perception agents in play")
    #     env.add_argument('--nfriendly_A', type=int, default=1,
    #                         help="Total number of friendly action agents in play")
    #     env.add_argument('--tensor_obs', action="store_true", default=False,
    #                      help="Do you want a tensor observation")
    #     env.add_argument('--second_reward_scheme', action="store_true", default=False,
    #                      help="Do you want a partial reward for capturing and partial for getting to it?")
    #     env.add_argument('--A_vision', type=int, default=-1,
    #                         help="Vision of A agents. If -1, defaults to blind")

    def multi_agent_init(self, args):

        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'moving_prey', 'mode', 'enemy_comm', 'nfriendly_P', 'nfriendly_A', 'tensor_obs']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.npredator = args.nfriendly_P
        self.npredator_capture = args.nfriendly_A
        self.nprey = args.nenemies
        self.predator_capture_index = self.npredator
        self.captured_prey_index = self.npredator + self.npredator_capture
        self.dims = dims = (self.dim, self.dim)
        self.stay = not args.no_stay
        self.tensor_obs = args.tensor_obs
        self.second_reward_scheme = args.second_reward_scheme
        if args.A_vision != -1:
            # if args.A_vision != 0:
            #     raise NotImplementedError
            self.A_agents_have_vision = True
            self.A_vision = args.A_vision
            self.action_blind = False
        else:
            self.A_agents_have_vision = False

        agents = self.npredator + self.npredator_capture
        self.n_agents = agents

        if args.moving_prey:
            raise NotImplementedError
            # TODO

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        if self.stay:
            self.naction = 6
        else:
            self.naction = 5

        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        self.PREDATOR_CAPTURE_CLASS += self.BASE
        self.state_len = (self.nfriendly_P + self.nfriendly_A + 1) # Change this if you change state_len in main_copy.py

        # self.vocab_size = self.BASE + self.state_len
        # 4 is 4 types of classes!
        self.num_classes = 4
        self.vocab_size = self.BASE + self.num_classes

        # Observation for each agent will be vision * vision ndarray
        # if self.tensor_obs:
        #     self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self.dim, self.dim, 4])
        #     if self.vision == 0:
        #         self.feature_map = [np.zeros((1, 1, 3)), np.zeros((self.dim, self.dim, 1))]
        #     elif self.vision == 1:
        #         self.feature_map = [np.zeros((3, 3, 3)), np.zeros((self.dim, self.dim, 1))]
        #     else:
        #         assert "vision of 2 unsupported"

        #     self.true_feature_map = np.zeros((self.dim, self.dim, 3 + self.nfriendly_P + self.nfriendly_A))

        # else:
        #     self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1), dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        # self.action_space = spaces.MultiDiscrete([self.naction])
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        tmp_obs = self.reset()

        for agent in range(agents):
            self.action_space.append(spaces.Discrete(self.naction))

            # Observation for each agent will be vision * vision ndarray
            # self.observation_space.append(spaces.Box(low=0, high=1, shape=self.vocab_size * ((2 * self.vision) + 1) * ((2 * self.vision) + 1), dtype=int))
            # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size
        self.observation_space = [spaces.Box(low=0, high=1, shape=(tmp_obs[n].shape[0], ), dtype=np.float32)
                                  for n in range(self.n_agents)]
        share_obs_dim = tmp_obs[0].shape[0] * self.n_agents
        self.share_observation_space = [spaces.Box(low=0, high=1, shape=(share_obs_dim, ), dtype=np.float32)
                                  for n in range(self.n_agents)]

        return

    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        # self.start_time = time.time()
        if self.episode_over:
            raise RuntimeError("Episode is done")

        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        for i, a in enumerate(action):
            self._take_action(i, a)

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        self.episode_over = False
        if self.tensor_obs:
            self.obs = self._get_tensor_obs()
        else:
            self.obs = self._get_obs()

        debug = {
            'predator_locs' : self.predator_loc,
            'predator_capture_locs' : self.predator_capture_loc,
            'prey_locs' : self.prey_loc,
        }

        return self.obs, self._get_reward(), [self.episode_over for _ in range(self.n_agents)], [debug for _ in range(self.n_agents)]

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.start_time = time.time()
        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator + self.npredator_capture)
        self.captured_prey = np.zeros(self.npredator_capture)

        # Locations
        locs = self._get_cordinates()
        self.predator_loc = locs[:self.npredator]
        self.predator_capture_loc = locs[self.predator_capture_index:self.captured_prey_index]
        self.prey_loc = locs[self.captured_prey_index:]

        # TODO: check vocab size, may be incorrect
        # Test
        # self.predator_loc[0] = self.predator_loc[1]
        self._set_grid()

        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        if self.tensor_obs:
            self.obs = self._get_tensor_obs()
        else:
            self.obs = self._get_obs()

        return self.obs

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims), (self.npredator + self.npredator_capture + self.nprey),
                               replace=False)
        # if self.args.eval:
        #     with open('/home/rohanpaleja/PycharmProjects/HetGAT_MARL_Communication/test/IC3Net/initial_starts_for_test_time.txt', 'r') as my_file:
        #         new_idx = my_file.read()
        #         import ast
        #         idx = ast.literal_eval(new_idx.split('\n')[self.episode_eval_counter])
        #         idx = np.array(idx)
        #     self.episode_eval_counter += 1
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.grid[self.prey_loc[:,0], self.prey_loc[:,1]] = self.prey_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)


    def _get_obs(self):
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1

        for i, p in enumerate(self.predator_capture_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CAPTURE_CLASS] += 1

        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x].flatten())

        for p in self.predator_capture_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])
            # x,y,z = self.bool_base_grid[slice_y, slice_x].shape
            # obs.append(np.arange(x*y*z).reshape(x,y,z))

            if self.action_blind:
                # [-1] is because we just appended it above
                obs[-1][:, :, self.BASE:] = np.full(shape=obs[-1][:, :, self.BASE:].shape, fill_value=-1)
            elif self.A_agents_have_vision:
                slice_A = slice(self.vision - self.A_vision, self.vision + self.A_vision + 1)

                A_vision_obs = obs[-1][slice_A, slice_A].copy()
                obs[-1] = np.full(shape=obs[-1].shape, fill_value=-1)
                # translates all other observations and states into -1s except the one that is "active"
                obs[-1][slice_A, slice_A] = A_vision_obs
            
            obs[-1] = obs[-1].flatten()

        if self.enemy_comm:
            for p in self.prey_loc:
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                obs.append(self.bool_base_grid[slice_y, slice_x].flatten())

        obs = np.stack(obs)

        return obs

    def _get_tensor_obs(self):
        """
        Notes: axis is initialized from the top corner. For example, (1,0) (one down, 0 right) refers to
        array([[0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        """
        true_feature_map = self.true_feature_map.copy()

        for i in ['predator', 'predator_capture', 'prey', 'self']:
            if i == 'predator':
                for p in self.predator_loc:
                    true_feature_map[p[0], p[1], 0] += 1
            elif i== 'predator_capture':
                for p in self.predator_capture_loc:
                    true_feature_map[p[0], p[1], 1] += 1
            elif i == 'prey':
                for p in self.prey_loc:
                    true_feature_map[p[0], p[1], 2] += 1
            elif i == 'self':
                all_predator_locs = np.vstack([self.predator_loc, self.predator_capture_loc])
                for k, p in enumerate(all_predator_locs):
                    true_feature_map[p[0], p[1], 3 + k] = 1

        obs = []
        agent_counter = 0

        for p in self.predator_loc:
            sub_obs = copy.deepcopy(self.feature_map)

            if self.vision == 0:
                y_start = p[0] - self.vision
                y_end = p[0] + self.vision
                x_start = p[1] - self.vision
                x_end = p[1] + self.vision

                sub_obs[0][:,:,0] = true_feature_map[y_start, x_start, 0]
                sub_obs[0][:,:,1] = true_feature_map[y_start, x_start, 1]
                sub_obs[0][:,:,2] = true_feature_map[y_start, x_start, 2]
                sub_obs[1] = true_feature_map[:, :, 3 + agent_counter]
            elif self.vision == 1:
                y_start = p[0] + 1 - self.vision
                y_end = p[0] + self.vision + 1 + 1
                x_start = p[1] + 1 - self.vision
                x_end = p[1] + 1 + self.vision + 1

                padded_feature_map_0 = np.pad(true_feature_map[:,:,0], 1, constant_values=-1)
                padded_feature_map_1 = np.pad(true_feature_map[:,:,1], 1, constant_values=-1)
                padded_feature_map_2 = np.pad(true_feature_map[:,:,2], 1, constant_values=-1)

                sub_obs[0][0] = padded_feature_map_0[y_start:y_end, x_start:x_end]
                sub_obs[0][1] = padded_feature_map_1[y_start:y_end, x_start:x_end]
                sub_obs[0][2] = padded_feature_map_2[y_start:y_end, x_start:x_end]
                sub_obs[1] = true_feature_map[:,:, 3 + agent_counter]
            else:
                raise ValueError

            agent_counter += 1
            obs.append(sub_obs)

        # 29 are concatenation of one-hot vectors including the information for the prey and predator, and also the position information of the grid
        # 10:14
        # There is also one indicating if this grid is a result of padding. There can be paddings at the edge of the grids when vision >= 1
        for p in self.predator_capture_loc:
            sub_obs = self.feature_map.copy()
            sub_obs[1] = true_feature_map[:, :, 3 + agent_counter]
            agent_counter += 1
            obs.append(sub_obs)

        # if self.enemy_comm:
        #     for p in self.prey_loc:
        #         slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
        #         slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
        #         obs.append(self.bool_base_grid[slice_y, slice_x])

        obs = np.stack(obs)

        return obs


    def _take_action(self, idx, act):
        # if idx == 2:
        #     print(act, self.predator_capture_loc, self.prey_loc)
        if isinstance(act, np.ndarray):
            assert sum(act) == 1
            act = list(act).index(1)
        # prey action
        if idx >= self.npredator + self.npredator_capture:
            # fixed prey
            if not self.moving_prey:
                return
            else:
                raise NotImplementedError

        if not self.stay:
            raise NotImplementedError

        # STAY action
        if act==4:
            return

        # Predator Capture action
        if idx >= self.predator_capture_index:
            # TODO: fix sink
            if self.reached_prey[idx] == 1 and self.captured_prey[idx - self.npredator]:
                return
            if self.reached_prey[idx] == 1 and act in [0,1,2,3,4]:
                return

            # UP
            if act == 0 and self.grid[max(0,
                                          self.predator_capture_loc[idx - self.npredator][0] + self.vision - 1),
                                      self.predator_capture_loc[idx - self.npredator][1] + self.vision] != self.OUTSIDE_CLASS:
                self.predator_capture_loc[idx - self.npredator][0] = max(0, self.predator_capture_loc[idx - self.npredator][0] - 1)

            # RIGHT
            elif act == 1 and self.grid[self.predator_capture_loc[idx - self.npredator][0] + self.vision,
                                        min(self.dims[1] - 1,
                                            self.predator_capture_loc[idx - self.npredator][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
                self.predator_capture_loc[idx - self.npredator][1] = min(self.dims[1] - 1,
                                                self.predator_capture_loc[idx - self.npredator][1] + 1)

            # DOWN
            elif act == 2 and self.grid[min(self.dims[0] - 1,
                                            self.predator_capture_loc[idx - self.npredator][0] + self.vision + 1),
                                        self.predator_capture_loc[idx - self.npredator][1] + self.vision] != self.OUTSIDE_CLASS:
                self.predator_capture_loc[idx - self.npredator][0] = min(self.dims[0] - 1,
                                                self.predator_capture_loc[idx - self.npredator][0] + 1)

            # LEFT
            elif act == 3 and self.grid[self.predator_capture_loc[idx - self.npredator][0] + self.vision,
                                        max(0,
                                            self.predator_capture_loc[idx - self.npredator][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
                self.predator_capture_loc[idx - self.npredator][1] = max(0, self.predator_capture_loc[idx - self.npredator][1] - 1)

            elif act == 5:
                if (self.predator_capture_loc[idx - self.npredator] == self.prey_loc).all():
                    self.captured_prey[idx - self.npredator] = True

        else:
            if self.reached_prey[idx] == 1:
                return
            # UP
            if act==0 and self.grid[max(0,
                                    self.predator_loc[idx][0] + self.vision - 1),
                                    self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0]-1)

            # RIGHT
            elif act==1 and self.grid[self.predator_loc[idx][0] + self.vision,
                                    min(self.dims[1] -1,
                                        self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][1] = min(self.dims[1]-1,
                                                self.predator_loc[idx][1]+1)

            # DOWN
            elif act==2 and self.grid[min(self.dims[0]-1,
                                        self.predator_loc[idx][0] + self.vision + 1),
                                        self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][0] = min(self.dims[0]-1,
                                                self.predator_loc[idx][0]+1)

            # LEFT
            elif act==3 and self.grid[self.predator_loc[idx][0] + self.vision,
                                        max(0,
                                        self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1]-1)

            # Predator agents cannot take capture.
            elif act==5:
                return

    # TODO: fix reward
    def _get_reward(self):
        n = self.npredator + self.npredator_capture if not self.enemy_comm else self.npredator + self.npredator_capture + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)

        # find if predator is on prey
        # TODO: May be able to np.where pred_loc and pred_cap_loc seperately
        all_predator_locs = np.vstack([self.predator_loc, self.predator_capture_loc])
        on_prey_val = np.zeros((all_predator_locs.shape[0]), dtype=bool)

        for prey in self.prey_loc:
            on_prey_i = np.all(all_predator_locs==prey, axis=1)
            on_prey_val = np.any([on_prey_val, on_prey_i], axis=0)

        on_prey = np.where(on_prey_val)[0]      # indices of predators on prey
        nb_predator_on_prey = on_prey.size

        if self.mode == 'cooperative':
            reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'competitive':
            if nb_predator_on_prey:
                reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
        elif self.mode == 'mixed':
            reward[on_prey] = self.PREY_REWARD
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        self.reached_prey[on_prey] = 1

        # Take care of capture reward seperately
        if not self.second_reward_scheme:
            which_action_agents_have_not_captured_prey = np.where(self.captured_prey == 0)
            proper_action_agent_indexes = np.array(which_action_agents_have_not_captured_prey) + self.nfriendly_P
            reward[proper_action_agent_indexes] = self.TIMESTEP_PENALTY # self.TIMESTEP_PENALTY
        else:
            which_action_agents_have_not_captured_prey_but_are_on_prey = np.intersect1d(np.where(self.captured_prey == 0) , np.where(self.reached_prey[self.nfriendly_P:] == 1))

            proper_action_agent_indexes = np.array(which_action_agents_have_not_captured_prey_but_are_on_prey) + self.nfriendly_P
            reward[proper_action_agent_indexes] = self.ON_PREY_BUT_NOT_CAPTURE_REWARD  # self.TIMESTEP_PENALTY
            # if len(np.intersect1d(np.where(self.captured_prey == 0) , np.where(self.reached_prey[self.nfriendly_P:] == 1))) > 0:
            #     print(reward)


        if np.all(self.reached_prey == 1) and np.all(self.captured_prey == 1) and self.mode == 'mixed':
            self.episode_over = True
            # print('success episode!!!')

        # # Prey reward
        # if nb_predator_on_prey == 0:
        #     reward[self.npredator:] = self.TIMESTEP_PENALTY
        # else:
        #     # TODO: discuss & finalise
        #     reward[self.npredator:] = 0

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.npredator+self.npredator_capture and self.episode_over:
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0
        # print(reward)
        # print('reward', reward, 'all_predator_locs', all_predator_locs, 'prey_loc', self.prey_loc)
        return np.expand_dims(reward, axis=1)

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _onehot_initialization(self, a):
        ncols = self.vocab_size

        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def render(self, mode='human', close=False):
        self.init_curses()
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for p in self.predator_capture_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'A'
            else:
                grid[p[0]][p[1]] = 'A'

        for p, captured in zip(self.prey_loc, self.captured_prey):
            sym = 'G' if not captured else 'C'
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + sym
            else:
                grid[p[0]][p[1]] = sym

        # print(grid)

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'C' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'G' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    elif 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(5))
                    elif 'A' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(6))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()
