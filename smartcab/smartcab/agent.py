from __future__ import division
import random
import operator
from collections import defaultdict
import time

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import numpy as np


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    num_trial = 10
    success_count = 0

    r90_matrix = np.matrix([[0, -1], [1, 0]]) # Rotate CCW
    rn90_matrix = np.matrix([[0, 1], [-1, 0]]) # Rotate CW

    # random_reward = [-2, -1, 0, 1, 2]
    random_reward = [0]

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.epsilon = 1
        self.gamma = 0.9
        self.i_alpha = 2
        self.q_dict = defaultdict(lambda: np.random.choice(self.random_reward)) # element of q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value
        self.net_reward = 0
        self.total_net_reward = 0

    def max_q(self, location, heading, traffic_light, other_agents):
        # start = time.time()
        max_q = ''
        q_compare_dict = {}

        # Populate the q_dict
        for act in set(Environment.valid_actions):
            _ = self.q_dict[(location[0], location[1], heading[0], heading[1], traffic_light, other_agents[0], other_agents[1], other_agents[2], act)]
            q_compare_dict[(location[0], location[1], heading[0], heading[1], traffic_light, other_agents[0], other_agents[1], other_agents[2], act)] = self.q_dict[(location[0], location[1], heading[0], heading[1], traffic_light, other_agents[0], other_agents[1], other_agents[2], act)]
        # for loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act in self.q_dict: # <- DEFAULTDICT doesn't work here!!!!!!
        #     if (loc_x == location[0] and loc_y == location[1] and hd_x == heading[0] and hd_y == heading[1] and tl == traffic_light and oa_oc == other_agents[0] and oa_lt == other_agents[1] and oa_rt == other_agents[2]):
        #         q_compare_dict[(loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act)] = self.q_dict[(loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act)]
        #         # print("q_compare_dict (loop): {}".format(q_compare_dict))

        try:
            max(q_compare_dict.iteritems(), key=lambda x:x[1])
        except ValueError:
            pass
            # print("No Qs for the state, yet...")
        else:
            key, q_value = max(q_compare_dict.iteritems(), key=lambda x:x[1])
            return key[-1], q_value

    def q_update(self, now_loc, now_heading, now_light, now_agents):
        q_temp = self.q_dict[(self.prev_loc[0], self.prev_loc[1], self.prev_heading[0], self.prev_heading[1], self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)]
        q_temp = (1 - (1 / self.i_alpha)) * q_temp + (1 / self.i_alpha) * (self.prev_reward + self.gamma * self.max_q(now_loc, now_heading, now_light, now_agents)[1])
        self.q_dict[(self.prev_loc[0], self.prev_loc[1], self.prev_heading[0], self.prev_heading[1], self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)] = q_temp
        self.i_alpha = self.i_alpha + 1
        return (self.q_dict[(self.prev_loc[0], self.prev_loc[1], self.prev_heading[0], self.prev_heading[1],
            self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)])

    def _navigator(self, heading, delta):
        """Use heading and delta (next loc - current loc) to decide the SmartCab's action. Internal use (by policy) only.
        """
        valid_deltas = set([-1, 0, 1])

        if delta[0] == 0 and delta[1] == 0:
            return None
        elif delta[0] in valid_deltas and delta[1] in valid_deltas:

            heading_3d = [heading[0], heading[1], 0]
            delta_3d = [delta[0], delta[1], 0]

            direction = np.cross(heading_3d, delta_3d)

            if direction[2] == 0:
                return 'forward'
            elif direction[2] == -1:
                return 'left'
            elif direction[2] == 1:
                return 'right'
            else:
                raise ValueError, "Navigation system malfuncitoning, man!"
        else:
            raise ValueError, "Navigator warning: wrong delta!"

    def policy(self, location, heading, traffic_light, other_agents):
        # start = time.time()
        # print("Start calculating policy...")
        # print("Location and Heading: {0}, {1}".format(location, heading))

        next_headings = [np.matrix(heading), np.dot(self.r90_matrix, heading), np.dot(self.rn90_matrix, heading)]
        # print("Next Headings: {}".format(next_headings))


        next_locs = [x + np.matrix(location) for x in next_headings]
        next_locs.append(np.matrix(location))

        # print("NEXT LOCS!: {}".format(next_locs))
        temp_next_loc = []
        # print("Next Locations (RAW): {}".format(next_locs))

        for m in next_locs:
            temp_next_loc.append(np.matrix(coord_convert([m[0, 0], m[0, 1]])))

        next_locs = [y for x in temp_next_loc for y in x.tolist()]
        # print("Next Locations (list): {}".format(next_locs))

        valid_actions = set([None, 'forward', 'right', 'left'])

        max_q = ''
        q_compare_dict = {}

        # print("Test MaxQ: {}".format(self.max_q([3, 4], (-1, 0), 'red', ['forward', None, None])))
        for loc in next_locs:
            # print("Next loc (in loop): {}".format(loc))
            delta_arr = np.array(loc) - np.array(location)
            if delta_arr[0] == 0 and delta_arr[1] == 0:
                # print("Staying where you are, heading: {}".format(heading))
                nnext_headings = [heading, tuple(np.dot(self.r90_matrix, np.array(heading)).tolist()[0]), tuple(np.dot(self.rn90_matrix, np.array(heading)).tolist()[0])]
            else:
                nnext_headings = [tuple(delta_arr.tolist()), tuple(np.dot(self.r90_matrix, delta_arr).tolist()[0]), tuple(np.dot(self.rn90_matrix, delta_arr).tolist()[0])]
            # print("NNext headings (list): {}".format(nnext_headings))
            for hd in nnext_headings:
                for tl in set(['green', 'red']):
                    for oa_oc in valid_actions:
                        for oa_lt in valid_actions:
                            for oa_rt in valid_actions:
                                q_compare_dict[(loc[0], loc[1], hd[0], hd[1], tl, oa_oc, oa_lt, oa_rt)] = self.max_q(loc, (hd[0], hd[1]), tl, [oa_oc, oa_lt, oa_rt])[1]
                                # print("Next Location and Utility: ({0}, {1})".format(loc, q_compare_dict[(loc[0], loc[1], hd[0], hd[1], tl, oa_oc, oa_lt, oa_rt)]))
        # print("Length of Q_compare_dict: {}".format(len(q_compare_dict)))
        # print("Q_Counter: {}".format(q_counter))
        key, q_value = max(q_compare_dict.iteritems(), key=lambda x:x[1])

        # print("KEY and MaxQ!: {0}, {1}".format(key, q_value))

        delta = [(key[0] - location[0]), (key[1] - location[1])]
        delta[0], delta[1] = delta_convert([delta[0], delta[1]])

        # print("delta[0] = {}".format(delta[0]))
        # print("delta[1] = {}".format(delta[1]))

        # end = time.time()
        # print("Got policy in {} sec.".format(end - start))
        # print("Going: {}".format(self._navigator(heading, delta)))
        return self._navigator(heading, delta)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.epsilon = 1
        self.i_alpha = 1
        self.net_reward = 0
        # self.q_dict = defaultdict(lambda: np.random.choice(self.random_reward))

        # Retain traffic light rules!
        for loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act in self.q_dict:
            if tl != 'red':
                self.q_dict[(loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act)] = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        oncoming = inputs['oncoming']
        left = inputs['left']
        right = inputs['right']

        # TODO: Update state
        now_loc = self.env.agent_states[self]['location'] # get SmartCab's location components
        now_heading = self.env.agent_states[self]['heading'] # get SmartCab's heading components
        now_light = inputs['light'] # get traffic light status
        now_agents = [oncoming, left, right] # get other agents' locations

        # TODO: Select action according to your policy
        if self.epsilon > 0.2: # decreasing the possibility of going random
            self.epsilon = self.epsilon - 0.05

        decision = np.random.choice(2, p = [self.epsilon, 1 - self.epsilon]) # decide to go random or with the policy
        print("random decision: {}".format(decision))
        if decision == 0: # if zero, go random
            action = random.choice(Environment.valid_actions[1:])
        else: # else go with the policy
            action = self.policy(now_loc, now_heading, now_light, now_agents)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Update q_dict with inverse learning rate self.i_alpha
        # q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value
        alpha = 1 / self.i_alpha

        try:
            self.prev_loc
        except AttributeError:
            print("Initializing...no prevs exist.")
        else:
            self.q_update(now_loc, now_heading, now_light, now_agents)

        self.prev_loc = now_loc
        self.prev_heading = now_heading
        self.prev_light = now_light
        self.prev_agents = now_agents
        self.prev_action = action
        self.prev_reward = reward

        self.net_reward += reward
        self.total_net_reward += reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def coord_convert(num_list):
    if num_list[0] != 0 and num_list[0] != 8:
        num_list[0] = num_list[0] % 8

        if num_list[1] != 0 and num_list[1] != 6:
            num_list[1] = num_list[1] % 6
        elif num_list[1] == 6:
            pass
        else:
            num_list[1] = num_list[1] + 6
    elif num_list[0] == 8:
        if num_list[1] != 0 and num_list[1] != 6:
            num_list[1] = num_list[1] % 6
        elif num_list[1] == 6:
            pass
        else:
            num_list[1] = num_list[1] + 6

    else:
        num_list[0] = num_list[0] + 8

        if num_list[1] != 0 and num_list[1] != 6:
            num_list[1] = num_list[1] % 6
        elif num_list[1] == 6:
            pass
        else:
            num_list[1] = num_list[1] + 6

    return num_list

def delta_convert(num_list):
    """Converts displacement vector to the correct form
    """
    if num_list[0] > 1:
        num_list[0] = num_list[0] - 8
        if num_list[1] > 1:
            num_list[1] = num_list[1] - 6
        elif num_list[1] < -1:
            num_list[1] = num_list[1] + 6
        else:
            pass
    elif num_list[0] < -1:
        num_list[0] = num_list[0] + 8
        if num_list[1] > 1:
            num_list[1] = num_list[1] - 6
        elif num_list[1] < -1:
            num_list[1] = num_list[1] + 6
        else:
            pass
    else:
        if num_list[1] > 1:
            num_list[1] = num_list[1] - 6
        elif num_list[1] < -1:
            num_list[1] = num_list[1] + 6
        else:
            pass

    return num_list


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=2.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=a.num_trial)  # press Esc or close pygame window to quit
    print("{} trials run. Sucess Count: {}".format(a.num_trial, a.success_count))
    print("Sucess Rate: {}".format(a.success_count / a.num_trial))
    print("Total Net Reward: {}".format(a.total_net_reward))
    print("Average Net Reward: {}".format(a.total_net_reward / a.num_trial))


if __name__ == '__main__':
    run()
