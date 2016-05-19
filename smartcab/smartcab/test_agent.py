from __future__ import division
import random
import operator
from collections import defaultdict

import numpy as np

from environment import Agent, Environment
from planner import RoutePlanner
from agent import LearningAgent

class TestAgent:
	r90_matrix = np.matrix([[0, -1], [1, 0]]) # Rotate CCW
    rn90_matrix = np.matrix([[0, 1], [-1, 0]]) # Rotate CW

    random_reward = [-2, -1, 0, 1, 2]

	def setup_method(self, method):
		print("Setting up METHOD {0}".format(method.__name__))
		self.e = Environment()
		self.agent = self.e.create_agent(LearningAgent)
		self.e.set_primary_agent(self.agent, enforce_deadline=False)
		self.q_dict = defaultdict(lambda: np.random.choice(self.random_reward))

	def test_q_learning(self):
		q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value
		print("Testing Q-Learning...")
		print(self.agent)



