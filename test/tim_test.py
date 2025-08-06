# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

import argparse
import gymnasium as gym
import pygame
import sys
import time

from collections import OrderedDict
from pygame import KEYDOWN, QUIT, K_ESCAPE, K_LEFT, K_UP, K_RIGHT
from pyquaticus.config import ACTION_MAP
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from tim_policy import hybrid_agent
from pyquaticus import pyquaticus_v0
import numpy as np
import torch
from collections import deque
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std
import matplotlib.pyplot as plt

RENDER_MODE = "human"

runtime = 120  # seconds

import os


class Mem_Buffer:
    def __init__(self, max_size=10000, obs_dim=61, action_dim=5, discrete=False):
        self.max_size = max_size
        self.states = np.zeros((max_size, obs_dim))
        self.states_ = np.zeros((max_size, obs_dim))
        self.actions = np.zeros(
            (max_size, action_dim), dtype=np.int64 if discrete else np.float32
        )
        self.rewards = np.zeros((max_size))
        self.dones = np.zeros((max_size))
        self.current_step = 0
        self.max_step = 0

    def add(self, state, action, reward, state_, done):
        self.states[self.current_step] = state
        self.states_[self.current_step] = state_
        self.actions[self.current_step] = action
        self.rewards[self.current_step] = reward
        self.dones[self.current_step] = done
        self.current_step += 1
        if self.current_step >= self.max_size:
            self.max_step = self.max_size
            self.current_step = 0
        if self.max_step < self.max_size:
            self.max_step = self.current_step

    def get(self, index):
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.states_[index],
            self.dones[index],
        )

    def sample(self, batch_size):
        idx = np.random.choice(self.max_step, batch_size, replace=False)
        return self.get(idx)

    def __len__(self):
        return self.max_step

    def __str__(self):
        s = f"Mem_Buffer: {self.max_step} / {self.max_size}\n"
        s += f"states: {self.states[:self.max_step]}\n"
        s += f"states_: {self.states_[:self.max_step]}\n"
        s += f"actions: {self.actions[:self.max_step]}\n"
        s += f"rewards: {self.rewards[:self.max_step]}\n"
        s += f"dones: {self.dones[:self.max_step]}\n"
        return s

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, "states.npy"), self.states)
        np.save(os.path.join(path, "states_.npy"), self.states_)
        np.save(os.path.join(path, "actions.npy"), self.actions)
        np.save(os.path.join(path, "rewards.npy"), self.rewards)
        np.save(os.path.join(path, "dones.npy"), self.dones)
        # save current step and max step
        with open(os.path.join(path, "steps.txt"), "wb") as f:
            f.write(f"{self.current_step},{self.max_step},{self.max_size}".encode())

    def load(self, path):
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            return
        self.states = np.load(os.path.join(path, "states.npy"))
        self.states_ = np.load(os.path.join(path, "states_.npy"))
        self.actions = np.load(os.path.join(path, "actions.npy"))
        self.rewards = np.load(os.path.join(path, "rewards.npy"))
        self.dones = np.load(os.path.join(path, "dones.npy"))
        with open(os.path.join(path, "steps.txt"), "rb") as f:
            lines = f.readlines()
            for line in lines:
                self.current_step, self.max_step, self.max_size = [
                    int(x) for x in line.decode().split(",")
                ]


class KeyTest:

    def __init__(self, env, quittable=True):
        """
        Args:
            env: the pyquaticus environment
        """
        pygame.init()
        self.obs, self.info = env.reset(return_info=True)
        self.env = env
        self.manager_losses = []
        self.policy_losses = []
        self.human_losses = []

        self.red_team = []

        self.manager_mem_buffer = Mem_Buffer(
            max_size=100000,
            obs_dim=61,
            action_dim=7,
            discrete=True,
        )

        self.policy_mem_buffer = Mem_Buffer(
            max_size=100000,
            obs_dim=61,
            action_dim=2,
            discrete=True,
        )

        self.human_mem_buffer = Mem_Buffer(
            max_size=10000,
            obs_dim=61,
            action_dim=2,
            discrete=True,
        )
        self.human_mem_buffer.load("./human_transitions/")
        self._get_red_team(env, [])
        print("Red team: ", self.enemy_types)
        self.blue_team = []
        for player in env.agents_of_team[Team.BLUE_TEAM]:
            self.blue_team.append(hybrid_agent(test=False, anum=0))

        self.quittable = quittable
        self.no_op_action = 16
        straight = 4
        left = 6
        right = 2
        straightleft = 5
        straightright = 3

        self.blue_keys_to_action = {
            0: [0, 0],
            K_UP: [3, 0.0],
            K_LEFT: [0, -70],
            K_RIGHT: [0, 70],
            K_UP + K_LEFT: [3, -70],
            K_UP + K_RIGHT: [3, 70],
        }

        self.blue_agent_ids = []  # = env.agents_of_team[Team.BLUE_TEAM][0].id
        self.red_agent_ids = []  # = env.agents_of_team[Team.RED_TEAM][0].id
        for player in env.agents_of_team[Team.BLUE_TEAM]:
            self.blue_agent_ids.append(player.id)
        for player in env.agents_of_team[Team.RED_TEAM]:
            self.red_agent_ids.append(player.id)

    def _get_red_team(self, env, enemy_types):
        for i, player in enumerate(env.agents_of_team[Team.RED_TEAM]):
            etype = np.random.randint(0, 6)
            if etype == 0:
                enemy_types.append("BaseAttacker")
                self.red_team.append(
                    BaseAttacker(
                        env.agents_of_team[Team.RED_TEAM][0].id,
                        Team.RED_TEAM,
                        env,
                        mode="competition_medium",
                        continuous=True,
                    )
                )
            elif etype == 1:
                enemy_types.append("BaseDefender")
                self.red_team.append(
                    BaseDefender(
                        env.agents_of_team[Team.RED_TEAM][0].id,
                        Team.RED_TEAM,
                        env,
                        mode="competition_medium",
                        continuous=True,
                    )
                )
            elif etype == 2:
                enemy_types.append("Heuristic_CTF_Agent")
                self.red_team.append(
                    Heuristic_CTF_Agent(
                        env.agents_of_team[Team.RED_TEAM][0].id,
                        Team.RED_TEAM,
                        env,
                        mode="hard",
                        continuous=True,
                    )
                )
            elif etype == 3:
                enemy_types.append("hybrid_agent_defender")
                self.red_team.append(hybrid_agent(policy_type="defender"))
            elif etype == 4:
                enemy_types.append("hybrid_agent_attacker")
                self.red_team.append(hybrid_agent(policy_type="attacker"))
            elif etype == 5:
                enemy_types.append("hybrid_agent_dqn")
                self.red_team.append(
                    hybrid_agent(
                        policy_type="dqn",
                        path=f"./models copy/blue_agent_{i}",
                        test=True,
                    )
                )
        self.enemy_types = enemy_types

    def _reset_reward_items(self):
        self.num_oob = np.zeros(6)
        self.num_grabs = np.zeros(6)
        self.num_caps = np.zeros(6)

    def save_transitions(
        self, obs, obs_, action_dict, rewards, terminated, truncated, debug=False
    ):
        # obs: current observation
        # obs_: next observation
        # action_dict: action taken
        # rewards: reward received
        # terminated: whether the episode is done
        # truncated: whether the episode is truncated
        if False:
            for i, enemy_id in enumerate(self.red_agent_ids):
                if enemy_id in action_dict:
                    if debug:
                        print(f"Saving transition for Enemy Agent {enemy_id}")
                        print(f"  obs: {obs[enemy_id].shape}")
                        print(f"  obs_: {obs_[enemy_id].shape}")
                        print(f"  action: {action_dict[enemy_id]}")
                        print(f"  reward: {rewards[enemy_id]}")
                        print(f"  terminated: {terminated[enemy_id]}")
                        print(f"  truncated: {truncated[enemy_id]}")
                        print(f"  done: {terminated[enemy_id] or truncated[enemy_id]}")
                    self.policy_mem_buffer.add(
                        obs[enemy_id],
                        np.array(action_dict[enemy_id]),
                        rewards[enemy_id],
                        obs_[enemy_id],
                        terminated[enemy_id] or truncated[enemy_id],
                    )
        for i, player_id in enumerate(self.blue_agent_ids):
            if player_id in action_dict:
                if player_id == self.blue_agent_ids[self.key_agent]:
                    if debug:
                        print(f"Saving transition for Key Agent {player_id}")
                        print(f"  obs: {obs[player_id].shape}")
                        print(f"  obs_: {obs_[player_id].shape}")
                        print(f"  action: {action_dict[player_id]}")
                        print(f"  reward: {rewards[player_id]}")
                        print(f"  terminated: {terminated[player_id]}")
                        print(f"  truncated: {truncated[player_id]}")
                        print(
                            f"  done: {terminated[player_id] or truncated[player_id]}"
                        )
                    self.human_mem_buffer.add(
                        obs[player_id],
                        np.array(action_dict[player_id]),
                        rewards[player_id],
                        obs_[player_id],
                        terminated[player_id] or truncated[player_id],
                    )
                else:
                    if debug:
                        print(f"Saving transition for DQN Agent {player_id}")
                        print(f"  obs: {obs[player_id].shape}")
                        print(f"  obs_: {obs_[player_id].shape}")
                        print(f"  action: {action_dict[player_id]}")
                        print(f"  reward: {rewards[player_id]}")
                        print(f"  terminated: {terminated[player_id]}")
                        print(f"  truncated: {truncated[player_id]}")
                        print(
                            f"  done: {terminated[player_id] or truncated[player_id]}"
                        )
                    self.policy_mem_buffer.add(
                        obs[player_id],
                        np.array(action_dict[player_id]),
                        rewards[player_id],
                        obs_[player_id],
                        terminated[player_id] or truncated[player_id],
                    )
                    if debug:
                        print(f"Saving manager transition for DQN Agent {player_id}")
                        print(f"  action: {self.blue_team[i].manager_act}")
                    self.manager_mem_buffer.add(
                        obs[player_id],
                        self.blue_team[i].manager_act,
                        rewards[player_id],
                        obs_[player_id],
                        terminated[player_id] or truncated[player_id],
                    )

    def manager_rl(self):
        if self.manager_mem_buffer.current_step < 2048:
            return
        states, actions, rewards, states_, terminated = self.manager_mem_buffer.sample(
            256
        )
        for i, player_id in enumerate(self.blue_agent_ids):
            dqloss, cqloss = self.blue_team[i].manager.reinforcement_learn(
                discrete_actions=[torch.from_numpy(actions).to(torch.int64).to("cuda")],
                continuous_actions=[None],
                obs=[torch.from_numpy(states).to("cuda")],
                obs_=[torch.from_numpy(states_).to("cuda")],
                global_rewards=torch.from_numpy(rewards).to("cuda"),
                terminated=torch.from_numpy(terminated).to("cuda"),
                agent_num=0,
                critic_only=False,
                debug=False,
            )
            # if i == 0:
            self.manager_losses.append(dqloss)

    def policy_rl(self):
        if self.policy_mem_buffer.current_step < 2048:
            return
        states, actions, rewards, states_, terminated = self.policy_mem_buffer.sample(
            256
        )
        for i, player_id in enumerate(self.blue_agent_ids):
            dqloss, cqloss = self.blue_team[i].policy_one.reinforcement_learn(
                discrete_actions=[None],
                continuous_actions=[torch.from_numpy(actions).to("cuda")],
                obs=[torch.from_numpy(states).to("cuda")],
                obs_=[torch.from_numpy(states_).to("cuda")],
                global_rewards=torch.from_numpy(rewards).to("cuda"),
                terminated=torch.from_numpy(terminated).to("cuda"),
                agent_num=0,
                critic_only=False,
                debug=False,
            )
            # if i == 0:
            self.policy_losses.append(cqloss)

    def human_immitation(self):
        if self.human_mem_buffer.current_step < 2048:
            return
        states, actions, rewards, states_, terminated = self.human_mem_buffer.sample(
            256
        )

        for i, player_id in enumerate(self.blue_agent_ids):
            # dqloss, cqloss = self.blue_team[i].policy_one.reinforcement_learn(
            #     discrete_actions=[None],
            #     continuous_actions=[torch.from_numpy(actions).to("cuda")],
            #     obs=[torch.from_numpy(states).to("cuda")],
            #     obs_=[torch.from_numpy(states_).to("cuda")],
            #     global_rewards=torch.from_numpy(rewards).to("cuda"),
            #     terminated=torch.from_numpy(terminated).to("cuda"),
            #     agent_num=0,
            #     critic_only=False,
            #     debug=False,
            # )
            # if i == 0:
            # self.human_losses.append(cqloss)
            # print(f"observations: {states.shape}, actions: {actions.shape}")
            dqloss, cqloss = self.blue_team[i].policy_one.imitation_learn(
                torch.from_numpy(states).to("cuda"),
                continuous_actions=torch.from_numpy(actions).to("cuda"),
                discrete_actions=None,
            )
            self.human_losses.append(cqloss)

    def begin(self):
        self.key_agent = -1
        step = 0
        reward_tots = [0, 0, 0, 0, 0, 0]
        rolling_rewards = []
        prev_checkpoint = -100
        highest_avg = 0
        for i in range(6):
            rolling_rewards.append(deque(maxlen=10))
        while True:
            action_dict = self.process_event(self.quittable)
            self.obs_, rewards, terminated, truncated, self.info = self.env.step(
                action_dict
            )
            for i, player_id in enumerate(self.blue_agent_ids):
                reward_tots[i] += rewards[player_id]
            for i, player_id in enumerate(self.red_agent_ids):
                reward_tots[i + 3] += rewards[player_id]
            self.save_transitions(
                self.obs, self.obs_, action_dict, rewards, terminated, truncated
            )
            self.manager_rl()
            self.policy_rl()
            # self.human_immitation()
            # print(self.obs)
            self.obs = self.obs_
            for k in terminated:
                if terminated[k] == True or truncated[k] == True:
                    # time.sleep(1.0)
                    self.obs, self.info = self.env.reset(return_info=True)
                    self.red_team = []
                    self._get_red_team(env, [])
                    self.key_agent = -1  # np.random.randint(0, 3)
                    for ri in range(6):
                        rolling_rewards[ri].append(reward_tots[ri])
                    reward_tots = [0, 0, 0, 0, 0, 0]

                    avg_rewards = [0, 0, 0, 0, 0, 0]
                    for ri in range(6):
                        avg_rewards[ri] = sum(rolling_rewards[ri]) / len(
                            rolling_rewards[ri]
                        )
                    for i in range(3):
                        if (
                            avg_rewards[0] + avg_rewards[1] + avg_rewards[2]
                            > highest_avg
                        ):
                            highest_avg = (
                                avg_rewards[0] + avg_rewards[1] + avg_rewards[2]
                            )
                            # Save blue agent models
                            self.blue_team[i].policy_one.save(
                                f"./models/blue_agent_{i}_model/"
                            )
                            self.blue_team[i].manager.save(
                                f"./models/blue_agent_{i}_manager/"
                            )
                    # plt.plot(self.manager_losses, label="Manager Loss")
                    # plt.plot(self.policy_losses, label="Policy Loss")
                    # plt.plot(self.human_losses, label="Human Loss")
                    # plt.legend()
                    # plt.xlabel("Step")
                    # plt.ylabel("Loss")
                    # plt.title("Losses")
                    # plt.show()
                    print("Avg rewards: ", avg_rewards)
                    # print("Red team: ", self.enemy_types)
                    # print("Key agent: ", self.key_agent)
                    s = "n"
                    # s = input("Save human transitions?")
                    if s == "y":
                        self.human_mem_buffer.save("./human_transitions/")
                    else:
                        self.human_mem_buffer.load("./human_transitions/")
                    break
            step += 1

    def process_event(self, quittable):

        if quittable:
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    self.env.close()
                    sys.exit()

        action_dict = OrderedDict(
            [(player_id, self.no_op_action) for player_id in self.env.players]
        )
        is_key_pressed = pygame.key.get_pressed()
        self.red_actions = []
        # red policy
        for i in range(3):
            if self.red_agent_ids[i] in action_dict:
                # rta = self.red_team[i]
                # rta: BaseAttacker
                # rta.compute_action()
                # print(f" computing red action for {i} {self.red_agent_ids[i]}")
                if self.enemy_types[i] in ["BaseAttacker", "BaseDefender"]:
                    action_dict[self.red_agent_ids[i]] = self.red_team[
                        i
                    ].compute_action(self.obs[self.red_agent_ids[i]], self.info)
                elif self.enemy_types[i] in ["Heuristic_CTF_Agent"]:
                    action_dict[self.red_agent_ids[i]] = self.red_team[
                        i
                    ].compute_action(self.obs[self.red_agent_ids[i]], self.info)
                else:
                    action_dict[self.red_agent_ids[i]] = self.red_team[
                        i
                    ].compute_action(
                        self.red_agent_ids[i],
                        full_obs_normalized=self.obs,
                        full_obs=self.obs,
                        global_state=None,
                    )
                self.red_actions.append(action_dict[self.red_agent_ids[i]])
        # red_action = self.red_policy.compute_action(self.obs, self.info)
        # action_dict[self.red_agent_id] = red_action

        # blue keys
        # blue_keys = (
        #     K_RIGHT * is_key_pressed[K_RIGHT]
        #     + K_LEFT
        #     * is_key_pressed[K_LEFT]
        #     * (is_key_pressed[K_LEFT] - is_key_pressed[K_RIGHT])
        #     + K_UP * is_key_pressed[K_UP]
        # )
        # blue_action = self.blue_keys_to_action[blue_keys]

        for i in range(3):
            if self.blue_agent_ids[i] in action_dict:
                if i == self.key_agent:
                    blue_keys = (
                        K_RIGHT * is_key_pressed[K_RIGHT]
                        + K_LEFT
                        * is_key_pressed[K_LEFT]
                        * (is_key_pressed[K_LEFT] - is_key_pressed[K_RIGHT])
                        + K_UP * is_key_pressed[K_UP]
                    )
                    blue_action = self.blue_keys_to_action[blue_keys]
                    action_dict[self.blue_agent_ids[i]] = blue_action
                else:
                    action_dict[self.blue_agent_ids[i]] = self.blue_team[
                        i
                    ].compute_action(self.blue_agent_ids[i], self.obs, self.obs, None)
                # print(self.blue_team[i].manager_act)
        # print(action_dict)
        # print(env.action_space(self.blue_agent_id))
        # action_dict[self.blue_agent_id] = blue_action
        return action_dict

    def caps_and_grabs(
        agent_id: str,
        team: Team,
        agents: list,
        agent_inds_of_team: dict,
        state: dict,
        prev_state: dict,
        env_size: np.ndarray,
        agent_radius: np.ndarray,
        catch_radius: float,
        scrimmage_coords: np.ndarray,
        max_speeds: list,
        tagging_cooldown: float,
    ):
        reward = 0.0
        # print(f"agent_id: {agent_id} team: {team}")
        prev_num_oob = float(prev_state["agent_oob"][agents.index(agent_id)])
        num_oob = float(state["agent_oob"][agents.index(agent_id)])
        if num_oob > prev_num_oob:
            reward += -0.5
        for i, t in enumerate(state["grabs"]):
            prev_num_grabs = prev_state["grabs"][i]
            num_grabs = state["grabs"][i]
            if num_grabs > prev_num_grabs:
                reward += 1.5 if i == team.value else -1.0

            prev_num_caps = prev_state["captures"][i]
            num_caps = state["captures"][i]
            if num_caps > prev_num_caps:
                reward += 10.0 if i == team.value else -10.0
        # if reward > 0.01 or reward < -0.01:
        #     # print(f"i [{i}] team: {team} agent_id: {agent_id}")
        #     print(f"agent_id: {agent_id} team: {team}{team.value}")
        #     print(f"prev_num_oob: {prev_num_oob} num_oob: {num_oob}")
        #     print(f"prev_num_grabs: {prev_state['grabs']} num_grabs: {state['grabs']}")
        #     print(
        #         f"prev_num_caps: {prev_state['captures']} num_caps: {state['captures']}"
        #     )
        #     print(f"reward: {reward}")
        #     input()
        return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a 3v3 policy in a 3v3 PyQuaticus environment"
    )
    parser.add_argument("--render", help="Enable rendering", action="store_true")
    reward_config = {
        "agent_0": KeyTest.caps_and_grabs,
        "agent_1": KeyTest.caps_and_grabs,
        "agent_2": KeyTest.caps_and_grabs,
        "agent_3": KeyTest.caps_and_grabs,
        "agent_4": KeyTest.caps_and_grabs,
        "agent_5": KeyTest.caps_and_grabs,
    }  # Example Reward Config
    # Competitors: reward_config should be updated to reflect how you want to reward your learning agent

    args = parser.parse_args()

    RENDER_MODE = (
        "human" if args.render else None
    )  # set to 'human' if you want rendered output

    config_dict = config_dict_std
    config_dict["sim_speedup_factor"] = 10
    config_dict["max_score"] = 3
    config_dict["max_time"] = 240
    config_dict["tagging_cooldown"] = 60
    config_dict["tag_on_oob"] = True
    config_dict["render_agent_ids"] = True
    env = pyquaticus_v0.PyQuaticusEnv(
        render_mode=RENDER_MODE,
        team_size=3,
        config_dict=config_dict,
        reward_config=reward_config,
    )
    kt = KeyTest(env)
    kt.begin()
