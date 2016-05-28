from __future__ import division
import random
import operator
from collections import defaultdict
from collections import Counter
import time

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import numpy as np


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    num_trial = 100
    success_count = 0
    total_net_reward = 0 # For counting total reward
    update_counter = 0 # For counting total steps
    num_reset = -1 # For getting the trial number

    trial_meta_info = [] # For monitoring what happens in each trial

    epsilon = 1
    gamma = 0.9
    random_reward = [0]

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.q_dict = defaultdict(lambda: (0, 0)) # element of q_dict is (next_waypoint, tl, oa_oc, oa_lt, oa_rt, act): [q_value, t]

        self.net_reward = 0

        self.num_step = 0 # Number of steps for each trial; get reset each time a new trial begins
        self.penalty = False # Noting if any penalty incurred, default False

    def max_q(self, next_waypoint, traffic_light, other_agents):
        # start = time.time()
        max_q = ''
        q_compare_dict = {}

        # Populate the q_dict
        for act in set(Environment.valid_actions):
            _ = self.q_dict[(next_waypoint, traffic_light, other_agents[0], other_agents[1], other_agents[2], act)]
            q_compare_dict[(next_waypoint, traffic_light, other_agents[0], other_agents[1], other_agents[2], act)] \
            = self.q_dict[(next_waypoint, traffic_light, other_agents[0], other_agents[1], other_agents[2], act)]

        try:
            max(q_compare_dict.iteritems(), key=lambda x:x[1])
        except ValueError:
            print("Wrong Q Value in Q Compare Dict!")
        else:
            key, qAndT = max(q_compare_dict.iteritems(), key=lambda x:x[1])
            return key[-1], qAndT[0], qAndT[1]

    def q_update(self, now_waypoint, now_light, now_agents):
        q_temp = self.q_dict[(self.prev_waypoint, self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)]
        q_temp0 = (1 - (1 / (q_temp[1] + 1))) * q_temp[0] + (1 / (q_temp[1] + 1)) * (self.prev_reward + self.gamma * self.max_q(now_waypoint, now_light, now_agents)[1])
        self.q_dict[(self.prev_waypoint, self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)] = (q_temp0, q_temp[1] + 1)

        return (self.q_dict[(self.prev_waypoint, self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)])

    # def _navigator(self, heading, delta):
    #     """Use heading and delta (next loc - current loc) to decide the SmartCab's action. Internal use (by policy) only.
    #     """
    #     valid_deltas = set([-1, 0, 1])

    #     if delta[0] == 0 and delta[1] == 0:
    #         return None
    #     elif delta[0] in valid_deltas and delta[1] in valid_deltas:

    #         heading_3d = [heading[0], heading[1], 0]
    #         delta_3d = [delta[0], delta[1], 0]

    #         direction = np.cross(heading_3d, delta_3d)

    #         if direction[2] == 0:
    #             return 'forward'
    #         elif direction[2] == -1:
    #             return 'left'
    #         elif direction[2] == 1:
    #             return 'right'
    #         else:
    #             raise ValueError, "Navigation system malfuncitoning, man!"
    #     else:
    #         raise ValueError, "Navigator warning: wrong delta!"

    def policy(self, next_waypoint, traffic_light, other_agents):

        return self.max_q(next_waypoint, traffic_light, other_agents)[0]

    def reset(self, destination=None):
        if self.num_step != 0:
            self.trial_meta_info.append({'penalty': self.penalty, 'step': self.num_step, 'net_reward': self.net_reward, 'reward_rate': self.net_reward / self.num_step})

        self.planner.route_to(destination)

        self.net_reward = 0
        self.num_step = 0 # Recalculate the steps for the new trial
        self.penalty = False

        self.num_reset += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        oncoming = inputs['oncoming']
        left = inputs['left']
        right = inputs['right']

        # Update state
        now_waypoint = self.next_waypoint
        now_light = inputs['light'] # get traffic light status
        now_agents = [oncoming, left, right] # get other agents' locations

        # Select action according to your policy
        if self.epsilon > 0.03: # decreasing the possibility of going random
            self.epsilon = self.epsilon - 0.0014

        self.decision = np.random.choice(2, p = [self.epsilon, 1 - self.epsilon]) # decide to go random or with the policy

        print("random decision: {}".format(self.decision))
        if self.decision == 0: # if zero, go random
            action = random.choice(Environment.valid_actions)
        else: # else go with the policy
            action = self.policy(now_waypoint, now_light, now_agents)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Update q_dict with inverse learning rate self.i_alpha
        # q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value

        try:
            self.prev_waypoint
        except AttributeError:
            print("Initializing...no prevs exist.")
        else:
            self.q_update(now_waypoint, now_light, now_agents)

        self.prev_waypoint = now_waypoint
        self.prev_light = now_light
        self.prev_agents = now_agents
        self.prev_action = action
        self.prev_reward = reward

        self.net_reward += reward
        self.num_step += 1
        if reward < 0:
            self.penalty = True

        self.total_net_reward += reward
        self.update_counter += 1
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

# def coord_convert(num_list):
#     if num_list[0] != 0 and num_list[0] != 8:
#         num_list[0] = num_list[0] % 8

#         if num_list[1] != 0 and num_list[1] != 6:
#             num_list[1] = num_list[1] % 6
#         elif num_list[1] == 6:
#             pass
#         else:
#             num_list[1] = num_list[1] + 6
#     elif num_list[0] == 8:
#         if num_list[1] != 0 and num_list[1] != 6:
#             num_list[1] = num_list[1] % 6
#         elif num_list[1] == 6:
#             pass
#         else:
#             num_list[1] = num_list[1] + 6

#     else:
#         num_list[0] = num_list[0] + 8

#         if num_list[1] != 0 and num_list[1] != 6:
#             num_list[1] = num_list[1] % 6
#         elif num_list[1] == 6:
#             pass
#         else:
#             num_list[1] = num_list[1] + 6

#     return num_list

# def delta_convert(num_list):
#     """Converts displacement vector to the correct form
#     """
#     if num_list[0] > 1:
#         num_list[0] = num_list[0] - 8
#         if num_list[1] > 1:
#             num_list[1] = num_list[1] - 6
#         elif num_list[1] < -1:
#             num_list[1] = num_list[1] + 6
#         else:
#             pass
#     elif num_list[0] < -1:
#         num_list[0] = num_list[0] + 8
#         if num_list[1] > 1:
#             num_list[1] = num_list[1] - 6
#         elif num_list[1] < -1:
#             num_list[1] = num_list[1] + 6
#         else:
#             pass
#     else:
#         if num_list[1] > 1:
#             num_list[1] = num_list[1] - 6
#         elif num_list[1] < -1:
#             num_list[1] = num_list[1] + 6
#         else:
#             pass

#     return num_list


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.5)  # reduce update_delay to speed up simulation
    sim.run(n_trials=a.num_trial)  # press Esc or close pygame window to quit

    run_log = open('perf.txt', 'a')
    if a.decision == 0:
        print("Random mode")
        run_log.write("Random mode\n")
    else:
        run_log.write("Smart-to-be mode\n")
        print("Smart-to-be mode")
    print("{} trials run. Success Count: {}\n".format(a.num_trial, a.success_count))
    print("Success Rate: {}".format(a.success_count / a.num_trial))
    print("Total Net Reward: {}".format(a.total_net_reward))
    print("Average Net Reward: {}".format(a.total_net_reward / a.num_trial))
    print("Finished {0} trials in {1} steps!".format(a.num_trial, a.update_counter))

    penalty_id = []

    for x, y in enumerate(a.trial_meta_info):
        if y['penalty'] == True:
            penalty_id.append(x)

    reward_rates = []

    for x in a.trial_meta_info:
        reward_rates.append(x['reward_rate'])

    reward_rates_rounded = [(round(x * 10) / 10) for x in reward_rates]

    print("Number of Penalized Trial(s): {}".format(len(penalty_id)))
    print("Penalized Trial(s): {}".format(penalty_id))
    print("Average Reward Rate: {}".format(sum(reward_rates) / len(reward_rates)))
    print("Max Reward Rate: {}".format(max(reward_rates)))
    print("Min Reward Rate: {}".format(min(reward_rates)))
    print("Range of Reward Rate: {}".format(max(reward_rates) - min(reward_rates)))

    reward_rate_counts = Counter(reward_rates_rounded)

    print("Mode of Reward Rate: {}".format(reward_rate_counts.most_common(1)))

    run_log.write("{} trials run. Success Count: {}\n".format(a.num_trial, a.success_count))
    run_log.write("Success Rate: {}\n".format(a.success_count / a.num_trial))
    run_log.write("Total Net Reward: {}\n".format(a.total_net_reward))
    run_log.write("Average Net Reward: {}\n".format(a.total_net_reward / a.num_trial))
    run_log.write("Finished {0} trials in {1} steps!\n\n\n".format(a.num_trial, a.update_counter))
    # run_log.write(a.trial_meta_info)
    run_log.close()


if __name__ == '__main__':
    run()
