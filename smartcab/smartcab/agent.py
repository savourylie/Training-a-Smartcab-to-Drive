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
    num_trial = 500

    total_net_reward = 0 # For counting total reward
    update_counter = 0 # For counting total steps
    trial_counter = 0 # For getting the trial number
    random_rounds = 100

    trial_meta_info = {} # For monitoring what happens in each trial

    epsilon = 1
    gamma = 0.8
    random_reward = [0]

    random_counter = 0
    policy_counter = 0

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.q_dict = defaultdict(lambda: (0, 0)) # element of q_dict is (next_waypoint, tl, oa_oc, oa_lt, oa_rt, act): [q_value, t]

        self.net_reward = 0

        self.num_step = 0 # Number of steps for each trial; get reset each time a new trial begins
        self.penalty = False # Noting if any penalty incurred, default False
        self.fail = False # Noting if it has reached the destination
        self.status = ()

    def get_state(self):

        return self.status

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

    def policy(self, next_waypoint, traffic_light, other_agents):

        return self.max_q(next_waypoint, traffic_light, other_agents)[0]

    def reset(self, destination=None):
        self.planner.route_to(destination)

        if self.epsilon - 1/self.random_rounds > 0:
            self.random_counter += 1
            self.epsilon = self.epsilon - 1/self.random_rounds
        else:
            self.epsilon = 0
            self.policy_counter += 1

        self.net_reward = 0
        self.num_step = 0 # Recalculate the steps for the new trial
        self.penalty = False
        self.fail = False

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

        self.status = (('next_waypoint', now_waypoint), ('traffic_light', now_light), ('oncoming', oncoming), ('left', left), ('right', right))

        # ETL layer for reducing dimentionality
        oncoming = left = right = 'None'

        now_agents = [oncoming, left, right] # get other agents' locations

        self.decision = np.random.choice(2, p = [self.epsilon, 1 - self.epsilon]) # decide to go random or with the policy
        # self.decision = 0 # Force random mode

        # print("random decision: {}".format(self.decision))
        if self.decision == 0: # if zero, go random
            action = random.choice(Environment.valid_actions)
        else: # else go with the policy
            action = self.policy(now_waypoint, now_light, now_agents)

        # Execute action and get reward
        reward = self.env.act(self, action)

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

        if self.trial_counter >= self.random_rounds: # Only count steps in the testing phase.
            self.update_counter += 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.3)  # reduce update_delay to speed up simulation
    sim.run(n_trials=a.num_trial)  # press Esc or close pygame window to quit

    run_log = open('perf.txt', 'a')

    print("# Dummy Agents: {}".format(e.num_dummies))
    print("Gamma: {}".format(a.gamma))

    fail_id = []

    for x, y in enumerate(a.trial_meta_info):
        if a.trial_meta_info[y]['fail'] == True:
            fail_id.append(y)

    print("{0} trials run. Random rounds: {1}, Test-driving {2} trials. Success Count: {3}".format(len(a.trial_meta_info), a.random_counter, a.policy_counter, (a.policy_counter - len([x for x in fail_id if x >= a.random_rounds]))))


    print("Failed Trials: {}".format([x for x in fail_id if x >= a.random_rounds]))
    print("Success Rate: {}".format((a.policy_counter - len([x for x in fail_id if x >= a.random_rounds])) / a.policy_counter))

    steps = []

    for x, y in enumerate(a.trial_meta_info):
        steps.append(a.trial_meta_info[y]['distance'])

    avgd = sum(steps[a.random_rounds:]) / len(steps[a.random_rounds:])
    avgs = a.update_counter / (a.num_trial - a.random_rounds)

    print("Avg. distance {}".format(avgd))
    print("Avg. steps: {}".format(avgs))
    print("D/S: {}".format(avgd/avgs))

    penalty_id = []

    for x, y in enumerate(a.trial_meta_info):
        if a.trial_meta_info[y]['penalty'] == True:
            penalty_id.append(y)

    reward_rates = []

    for x, y in enumerate(a.trial_meta_info):
        reward_rates.append(a.trial_meta_info[y]['reward_rate'])

    reward_rates_rounded = [(round(x * 10) / 10) for x in reward_rates[a.random_rounds:]]

    print("Number of Penalized Trial(s): {}".format(len([x for x in penalty_id if x >= a.random_rounds])))
    print("Penalized Trial(s): {}".format([x for x in penalty_id if x >= a.random_rounds]))

    try:
        max(penalty_id[a.random_rounds:])
    except ValueError:
        pass
    else:
        print("Last Penalized Trial: {}".format(max(penalty_id[a.random_rounds:])))

    # Only counting the statistics of data where the agent follows the policy completely
    print("Average Reward Rate: {}".format(sum(reward_rates[a.random_rounds:]) / len(reward_rates[a.random_rounds:])))
    print("SD of Reward Rate: {}".format(np.std(reward_rates[a.random_rounds:])))

    print("Max Reward Rate: {}".format(max(reward_rates[a.random_rounds:])))
    print("Min Reward Rate: {}".format(min(reward_rates[a.random_rounds:])))
    print("Range of Reward Rate: {}".format(max(reward_rates[a.random_rounds:]) - min(reward_rates[a.random_rounds:])))

    reward_rate_counts = Counter(reward_rates_rounded)

    print("Mode of Reward Rate: {}".format(reward_rate_counts.most_common(1)))

    print("Has shit happened? {}".format(e.shitshappened))
    print("a.trial_meta_info: {}".format(len(a.trial_meta_info)))
    print(a.trial_meta_info)

    run_log.write("# Dummy Agents: {}".format(e.num_dummies))
    run_log.write("Gamma: {}\n".format(a.gamma))
    run_log.write("{0} trials run. Random rounds: {1}, Test-driving {2} trials. Success Count: {3}\n".format(len(a.trial_meta_info), a.random_counter, a.policy_counter, (a.policy_counter - len([x for x in fail_id if x >= a.random_rounds]))))
    run_log.write("Failed Trials: {}\n".format([x for x in fail_id if x >= a.random_rounds]))
    run_log.write("Success Rate: {}\n".format((a.policy_counter - len([x for x in fail_id if x >= a.random_rounds])) / a.policy_counter))

    run_log.write("Avg. distance {}\n".format(avgd))
    run_log.write("Avg. steps: {}\n".format(avgs))
    run_log.write("D/S: {}\n".format(avgd/avgs))

    run_log.write("Number of Penalized Trial(s): {}\n".format(len([x for x in penalty_id if x >= a.random_rounds])))
    run_log.write("Penalized Trial(s): {}\n".format([x for x in penalty_id if x >= a.random_rounds]))

    # Only counting the statistics of data where the agent follows the policy completely
    run_log.write("Average Reward Rate: {}\n".format(sum(reward_rates[a.random_rounds:]) / len(reward_rates[a.random_rounds:])))
    run_log.write("SD of Reward Rate: {}\n".format(np.std(reward_rates[a.random_rounds:])))

    run_log.write("Max Reward Rate: {}\n".format(max(reward_rates[a.random_rounds:])))
    run_log.write("Min Reward Rate: {}\n".format(min(reward_rates[a.random_rounds:])))
    run_log.write("Range of Reward Rate: {}\n".format(max(reward_rates[a.random_rounds:]) - min(reward_rates[a.random_rounds:])))
    run_log.write("Mode of Reward Rate: {}\n\n".format(reward_rate_counts.most_common(1)))


if __name__ == '__main__':
    run()

