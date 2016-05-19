from __future__ import division
import random
import operator
from collections import defaultdict

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import numpy as np



class LearningAgent(Agent):
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
        self.i_alpha = 1
        self.q_dict = defaultdict(lambda: np.random.choice(self.random_reward)) # element of q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value

    def q_learning(self, location, heading, traffic_light, other_agents):
        max_q = ''
        q_compare_dict = {}

        for loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act in self.q_dict:
            if loc_x == location[0] & loc_y == location[1] & hd_x == heading[0] & hd_y == heading[1] & tl == traffic_light & oa_oc == other_agents[0] & oa_lt == other_agents[1] & oa_rt == other_agents[2]:
                q_compare_dict[(loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act)] = self.q_dict[(loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act)]

        _, q_value = max(q_compare_dict.iteritems(), key=lambda x:x[1])

        return q_value

    def policy(self, location, heading, traffic_light, other_agents):
        next_headings = set([heading, np.dot(self.r90_matrix, heading), np.dot(self.rn90_matrix, heading)])
        next_locs = set([x + heading for x in next_headings])
        valid_actions = set(['forward', 'right', 'left'])

        max_q = ''
        q_compare_dict = {}

        for loc in next_locs:
            for hd in set([loc - location, np.dot(self.r90_matrix, loc - location), np.dot(self.rn90_matrix, loc - location)]):
                for tl in set(['green', 'red']):
                    for oa_oc in valid_actions:
                        for oc_lt in valid_actions:
                            for oa_rt in valid_actions:
                                for act in set(Environment.valid_actions):
                                    q_compare_dict[(loc[0], loc[1], hd[0], hd[1], tl, oa_oc, oa_lt, oa_rt, act)] = self.q_dict[(loc[0], loc[1], hd[0], hd[1], tl, oa_oc, oa_lt, oa_rt, act)]

        key, q_value = max(q_compare_dict.iteritems(), key=lambda x:x[1])

        delta = (key[0] - location[0], key[1] - location[1])

        if delta[0] != 0: # Going EW
            if delta[0] * heading[0] == 1:
                action = 'forward'
            elif delta[0] * heading[1] == -1:
                action = 'right'
            elif delta[0] * heading[1] == 1:
                action = 'left'
        else: # Going NS
            if delta[1] * heading[1] == 1:
                action = 'forward'
            elif delta[1] * heading[0] == -1:
                action = 'left'
            elif delta[1] * heading[0] == 1:
                action = 'right'

        return action

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
            q_update = q_dict[(self.prev_loc[0], self.prev_loc[1], self.prev_heading[0], self.prev_heading[1], self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)]
            q_update = (1 - alpha) * q_update + (alpha * (self.gamma * self.q_learning(now_loc, now_heading, now_light, now_agents)))
            q_dict[(self.prev_loc[0], self.prev_loc[1], self.prev_heading[0], self.prev_heading[1], self.prev_light, self.prev_agents[0], self.prev_agents[1], self.prev_agents[2], self.prev_action)] = q_update
            self.i_alpha = self.i_alpha + 1

        self.prev_loc = now_loc
        self.prev_heading = now_heading
        self.prev_light = now_light
        self.prev_agents = now_agents
        self.prev_action = action
        self.prev_reward = reward


        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


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
