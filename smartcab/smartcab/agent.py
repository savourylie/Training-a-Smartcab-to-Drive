from __future__ import division
import random
import operator
from collections import defaultdict

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import numpy as np



class LearningAgent(Agent): ## TODO: REMOVE EVERY MATRIX FROM THIS CLASS!!!!!!!!!
    """An agent that learns to drive in the smartcab world."""

    r90_matrix = np.matrix([[0, -1], [1, 0]]) # Rotate CCW
    rn90_matrix = np.matrix([[0, 1], [-1, 0]]) # Rotate CW

    random_reward = [-2, -1, 0, 1, 2]

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.epsilon = 1
        self.gamma = 0.9
        self.i_alpha = 2
        self.q_dict = defaultdict(lambda: np.random.choice(self.random_reward)) # element of q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value

    def max_q(self, location, heading, traffic_light, other_agents):
        max_q = ''
        q_compare_dict = {}

        # Populate the q_dict
        for act in set(Environment.valid_actions):
            _ = self.q_dict[(location[0], location[1], heading[0], heading[1], traffic_light, other_agents[0], other_agents[1], other_agents[2], act)]

        for loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act in self.q_dict: # <- DEFAULTDICT doesn't work here!!!!!!
            if (loc_x == location[0] and loc_y == location[1] and hd_x == heading[0] and hd_y == heading[1] and tl == traffic_light and oa_oc == other_agents[0] and oa_lt == other_agents[1] and oa_rt == other_agents[2]):
                "YES!!"
                q_compare_dict[(loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act)] = self.q_dict[(loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act)]
                # print("q_compare_dict (loop): {}".format(q_compare_dict))

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
        # print("q_temp: {}".format(q_temp))
        # print("prev_reward: {}".format(self.prev_reward))
        # print("max_q: {}".format(self.max_q(now_loc, now_heading, now_light, now_agents)))
        # print("Gamma: {}".format(self.gamma))
        # print("Alpha: {}".format(1 / self.i_alpha))
        q_temp = (1 - (1 / self.i_alpha)) * q_temp + (1 / self.i_alpha) * (self.prev_reward + self.gamma * self.max_q(now_loc, now_heading, now_light, now_agents)[1])
        self.q_dict[(self.prev_loc[0], self.prev_loc[1], self.prev_heading[0], self.prev_heading[1], self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)] = q_temp
        self.i_alpha = self.i_alpha + 1
        return (self.q_dict[(self.prev_loc[0], self.prev_loc[1], self.prev_heading[0], self.prev_heading[1],
            self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)])

    def _navigator(self, heading, delta):
        """Use heading and delta (next loc - current loc) to decide the SmartCab's action. Internal use (by policy) only.
        """
        if delta[0] == 0 and delta[1] == 0:
            return None

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

    def policy(self, location, heading, traffic_light, other_agents):
        print("Location: {}".format(location))
        print("Heading: {}".format(heading))

        next_headings = [np.matrix(heading), np.dot(self.r90_matrix, heading), np.dot(self.rn90_matrix, heading)]
        print("Next Headings: {}".format(next_headings))


        next_locs = [x + np.matrix(location) for x in next_headings]
        next_locs.append(np.matrix(location))

        print("NEXT LOCS!: {}".format(next_locs))
        temp_next_loc = []
        print("Next Locations (RAW): {}".format(next_locs))

        for m in next_locs:
            temp_next_loc.append(np.matrix(coord_convert([m[0, 0], m[0, 1]])))

        next_locs = temp_next_loc
        print("Next Locations: {}".format(next_locs))

        ## TODO: FIX THE COORDINATE PROBLEMS FOR nnext_headings!!!!
        nnext_headings1 = [(loc - np.matrix(location)).T for loc in next_locs]
        nnext_headings2 = [np.dot(self.r90_matrix, (loc - np.matrix(location)).T) for loc in next_locs]
        nnext_headings3 = [np.dot(self.rn90_matrix, (loc - np.matrix(location)).T) for loc in next_locs]
        nnext_headings = nnext_headings1 + nnext_headings2 + nnext_headings3
        temp_nnext_headings = []
        print("NNext headings (RAW): {}".format(nnext_headings))

        for m in nnext_headings:
            # print(m)
            temp_nnext_headings.append(np.matrix(delta_convert([m[0, 0], m[1, 0]])))

        print(temp_nnext_headings)
        # nnext_headings = temp_nnext_headings

        valid_actions = set([None, 'forward', 'right', 'left'])

        max_q = ''
        q_compare_dict = {}

        # print("q_dict: {}".format([x for x in self.q_dict.values()]))
        # print("q_dict: {}".format([x for x in self.q_dict]))
        print("Test MaxQ: {}".format(self.max_q([3, 4], (-1, 0), 'red', ['forward', None, None])))
        for loc in next_locs:
            print("Next loc (in loop): {}".format(loc))
            for hd in nnext_headings:
                # print("NNext headings: {}".format(hd))
                for tl in set(['green', 'red']):
                    for oa_oc in valid_actions:
                        for oa_lt in valid_actions:
                            for oa_rt in valid_actions:
                                # if self.max_q([loc[0, 0], loc[0, 1]], (hd[0, 0], hd[1, 0]), tl, [oa_oc, oa_lt, oa_rt]) is not None:
                                q_compare_dict[(loc[0, 0], loc[0, 1], hd[0, 0], hd[1, 0], tl, oa_oc, oa_lt, oa_rt)] = self.max_q([loc[0, 0], loc[0, 1]], (hd[0, 0], hd[1, 0]), tl, [oa_oc, oa_lt, oa_rt])[1]
        # print("q_compare_dict: {}".format(q_compare_dict))
        key, q_value = max(q_compare_dict.iteritems(), key=lambda x:x[1])

        print("Very useful KEY!: {}".format(key))
        print("Useless MaxQ: {}".format(q_value))

        delta = [(key[0] - location[0]), (key[1] - location[1])]
        delta[0], delta[1] = delta_convert([delta[0], delta[1]])

        print("delta[0] = {}".format(delta[0]))
        print("delta[1] = {}".format(delta[1]))

        return self._navigator(heading, delta)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

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
        if self.epsilon > 0.05: # decreasing the possibility of going random
            self.epsilon = self.epsilon - 0.05

        decision = np.random.choice(2, p = [self.epsilon, 1 - self.epsilon]) # decide to go random or with the policy

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
        except NameError:
            print("Initializing...no prevs exist.")
        else:
            self.q_update(now_loc, now_heading, now_light, now_agents)

        self.prev_loc = now_loc
        self.prev_heading = now_heading
        self.prev_light = now_light
        self.prev_agents = now_agents
        self.prev_action = action
        self.prev_reward = reward

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
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
