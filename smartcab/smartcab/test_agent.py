
from agent import LearningAgent
from agent import coord_convert
from agent import delta_convert
from environment import Environment
import numpy as np
from numbers import Number

class TestAgent:
	def setup_method(self, method):
		print("Setting up METHOD {0}".format(method.__name__))
		self.e = Environment()
		self.agent = self.e.create_agent(LearningAgent)
		self.e.set_primary_agent(self.agent, enforce_deadline=False)

		self.q_dict = self.agent.q_dict

	def test_q_dict(self):
		# q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value
		print("Testing q_dict...")
		assert self.agent.q_dict[0] in [-2, -1, 0, 1, 2]
		assert self.agent.q_dict[28] in [-2, -1, 0, 1, 2]
		assert self.agent.q_dict[100] not in [3, 5, -100]

	def test_q_learning_output_type(self):
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12
		assert isinstance(self.agent.q_learning(np.array([5, 6]), np.array([1, 0]), 'green', ['forward', 'right', 'left'])[1], Number)

	def test_q_learning_output_value(self):
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12

		assert self.agent.q_learning(np.array([5, 6]), np.array([1, 0]), 'green', ['forward', 'right', 'left'])[1] == 18

	def test__navigation(self):
		assert self.agent._navigation((1, 0), [0, -1]) == 'left'
		assert self.agent._navigation((1, 0), [0, 1]) == 'right'
		assert self.agent._navigation((1, 0), [1, 0]) == 'forward'
		assert self.agent._navigation((0, 1), [0, 1]) == 'forward'
		assert self.agent._navigation((0, 1), [1, 0]) == 'left'
		assert self.agent._navigation((0, 1), [-1, 0]) == 'right'
		assert self.agent._navigation((0, 1), [0, 0]) == None
		assert self.agent._navigation((0, -1), [0, 0]) == None
		assert self.agent._navigation((1, 0), [0, 0]) == None
		assert self.agent._navigation((-1, 0), [0, 0]) == None
		assert self.agent._navigation((-1, 0), [-1, 0]) == 'forward'
		assert self.agent._navigation((-1, 0), [0, 1]) == 'left'
		assert self.agent._navigation((-1, 0), [0, -1]) == 'right'
		assert self.agent._navigation((0, -1), [0, -1]) == 'forward'
		assert self.agent._navigation((0, -1), [-1, 0]) == 'left'
		assert self.agent._navigation((0, -1), [1, 0]) == 'right'

	def test_policy_output_type(self):
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
		self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12

		assert type(self.agent.policy(np.array([5, 6]), np.array([1, 0]), 'green', ['forward', 'right', 'left'])) == type('string')

	# def test_policy_output_value(self):
	# 	self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18
	# 	self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
	# 	self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
	# 	self.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12

	# 	assert self.agent.policy(np.array([5, 6]), np.array([1, 0]), 'green', ['forward', 'right', 'left']) == 18

def test_coord_convert():
	assert coord_convert([8, 5]) == [8, 5]
	assert coord_convert([2, 6]) == [2, 6]
	assert coord_convert([8, 6]) == [8, 6]
	assert coord_convert([1, 2]) == [1, 2]
	assert coord_convert([9, 5]) == [1, 5]
	assert coord_convert([1, 7]) == [1, 1]
	assert coord_convert([0, 3]) == [8, 3]
	assert coord_convert([0, 0]) == [8, 6]
	assert coord_convert([2, 0]) == [2, 6]

def test_delta_convert():
	assert delta_convert([0, -5]) == [0, 1]
	assert delta_convert([-7, 0]) == [1, 0]
	assert delta_convert([7, 0]) == [-1, 0]
	assert delta_convert([7, 0]) == [-1, 0]
	assert delta_convert([0, 5]) == [0, -1]



