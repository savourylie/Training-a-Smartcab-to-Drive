
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

	def test_q_dict(self):
		# q_dict is (loc_x, loc_y, hd_x, hd_y, tl, oa_oc, oa_lt, oa_rt, act): q_value
		print("Testing q_dict...")
		assert self.agent.q_dict[0] in [-2, -1, 0, 1, 2]
		assert self.agent.q_dict[28] in [-2, -1, 0, 1, 2]
		assert self.agent.q_dict[100] not in [3, 5, -100]

	def test_max_q_output_type(self):
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12
		assert isinstance(self.agent.max_q([5, 6], [1, 0], 'green', ['forward', 'right', 'left'])[1], Number)

	def test_max_q_output_value(self):
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12

		assert self.agent.max_q([5, 6], [1, 0], 'green', ['forward', 'right', 'left'])[1] == 18

	def test_q_update_output_value(self):
		# Prev state/action 1
		self.agent.prev_reward = 5
		self.agent.prev_loc = [4, 6]
		self.agent.prev_heading = (1, 0)
		self.agent.prev_light = 'green'
		self.agent.prev_agents = ['right', 'left', 'forward']
		self.agent.prev_action = 'forward'

		self.agent.q_dict[(self.agent.prev_loc[0], self.agent.prev_loc[1], self.agent.prev_heading[0], self.agent.prev_heading[1], self.agent.prev_light, self.agent.prev_agents[0], self.agent.prev_agents[1], self.agent.prev_agents[2], self.agent.prev_action)] = 2

		# Now state 1
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18

		# Possibilities 1
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12

		assert self.agent.q_update([5, 6], [1, 0], 'green', ['forward', 'right', 'left']) \
		== 11.6

		self.agent.i_alpha = 5
		self.agent.gamma = 0.9

		# Prev state 2
		self.agent.prev_reward = 2
		self.agent.prev_loc = [8, 6]
		self.agent.prev_heading = (0, 1)
		self.agent.prev_light = 'green'
		self.agent.prev_agents = [None, None, 'right']
		self.agent.prev_action = 'left'

		self.agent.q_dict[(self.agent.prev_loc[0], self.agent.prev_loc[1], self.agent.prev_heading[0], self.agent.prev_heading[1], self.agent.prev_light, self.agent.prev_agents[0], self.agent.prev_agents[1], self.agent.prev_agents[2], self.agent.prev_action)] = -1

		# Now state 2
		now_loc = [1, 6]
		now_heading = (1, 0)
		now_light = 'red'
		now_agents = [None, 'forward', None]

		# Possibilities 2
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], 'forward')] = 3
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], 'right')] = 2
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], 'left')] = 0
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], None)] = -1

		assert self.agent.q_update(now_loc, now_heading, now_light, now_agents) \
		== 0.14

		self.agent.i_alpha = 20
		self.agent.gamma = 0.9

		# Prev state 3
		self.agent.prev_reward = 5
		self.agent.prev_loc = [4, 4]
		self.agent.prev_heading = (-1, 0)
		self.agent.prev_light = 'red'
		self.agent.prev_agents = [None, 'right', 'forward']
		self.agent.prev_action = 'forward'

		self.agent.q_dict[(self.agent.prev_loc[0], self.agent.prev_loc[1], self.agent.prev_heading[0], self.agent.prev_heading[1], self.agent.prev_light, self.agent.prev_agents[0], self.agent.prev_agents[1], self.agent.prev_agents[2], self.agent.prev_action)] = -2.5

		# Now state 3
		now_loc = [3, 4]
		now_heading = (-1, 0)
		now_light = 'red'
		now_agents = ['forward', 'forward', None]

		# Possibilities 3
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], 'forward')] = -1
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], 'right')] = 2
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], 'left')] = -2
		self.agent.q_dict[(now_loc[0], now_loc[1], now_heading[0], now_heading[1], now_light, now_agents[0], now_agents[1], now_agents[2], None)] = -1

		assert self.agent.q_update(now_loc, now_heading, now_light, now_agents) \
		== -2.035

	def test__navigator(self):
		assert self.agent._navigator((1, 0), [0, -1]) == 'left'
		assert self.agent._navigator((1, 0), [0, 1]) == 'right'
		assert self.agent._navigator((1, 0), [1, 0]) == 'forward'
		assert self.agent._navigator((0, 1), [0, 1]) == 'forward'
		assert self.agent._navigator((0, 1), [1, 0]) == 'left'
		assert self.agent._navigator((0, 1), [-1, 0]) == 'right'
		assert self.agent._navigator((0, 1), [0, 0]) == None
		assert self.agent._navigator((0, -1), [0, 0]) == None
		assert self.agent._navigator((1, 0), [0, 0]) == None
		assert self.agent._navigator((-1, 0), [0, 0]) == None
		assert self.agent._navigator((-1, 0), [-1, 0]) == 'forward'
		assert self.agent._navigator((-1, 0), [0, 1]) == 'left'
		assert self.agent._navigator((-1, 0), [0, -1]) == 'right'
		assert self.agent._navigator((0, -1), [0, -1]) == 'forward'
		assert self.agent._navigator((0, -1), [-1, 0]) == 'left'
		assert self.agent._navigator((0, -1), [1, 0]) == 'right'

	def test_policy_output_type(self):
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'right')] = 18
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'forward')] = -2
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', None)] = 0
		self.agent.q_dict[(5, 6, 1, 0, 'green', 'forward', 'right', 'left', 'left')] = 12

		assert type(self.agent.policy(np.array([5, 6]), np.array([1, 0]), 'green', ['forward', 'right', 'left'])) == type('string') or type(None)

	def test_policy_output_value(self):
		# Now state 1
		now_loc = [3, 4]
		now_heading = (-1, 0)
		now_light = 'red'
		now_agents = ['forward', 'forward', None]

		# Next state 1-0
		next_loc10 = [3, 4]
		next_heading10 = (-1, 0)
		next_light10 = 'red'
		next_agents10 = ['forward', None, None]
		next_action10 = 'right'

		# Next state 1-1-1
		next_loc11 = [3, 3]
		next_heading11 = (0, -1)
		next_light11 = 'red'
		next_agents11 = ['forward', 'forward', None]
		next_action11 = None

		# Next state 1-1-2
		next_loc12 = [3, 3]
		next_heading12 = (0, -1)
		next_light12 = 'green'
		next_agents12 = ['forward', None, None]
		next_action12 = 'forward'

		# Next state 1-2-1
		next_loc21 = [2, 4]
		next_heading21 = (-1, 0)
		next_light21 = 'green'
		next_agents21 = ['left', 'forward', None]
		next_action21 = 'left'

		# Next state 1-2-2
		next_loc22 = [2, 4]
		next_heading22 = (-1, 0)
		next_light22 = 'red'
		next_agents22 = ['left', 'forward', 'left']
		next_action22 = 'right'

		# Next state 1-2-3
		next_loc23 = [2, 4]
		next_heading23 = (-1, 0)
		next_light23 = 'green'
		next_agents23 = ['forward', 'forward', 'forward']
		next_action23 = 'right'

		# Next state 1-3
		next_loc3 = [3, 5]
		next_heading3 = (0, 1)
		next_light3 = 'red'
		next_agents3 = ['forward', None, None]
		next_action3 = 'right'

		# Possibilities 1
		self.agent.q_dict[(next_loc11[0], next_loc11[1], next_heading11[0], next_heading11[1], next_light11, next_agents11[0], next_agents11[1], next_agents11[2], next_action11)] = 3
		self.agent.q_dict[(next_loc12[0], next_loc11[1], next_heading12[0], next_heading12[1], next_light12, next_agents12[0], next_agents12[1], next_agents12[2], next_action12)] = 1
		self.agent.q_dict[(next_loc21[0], next_loc21[1], next_heading21[0], next_heading21[1], next_light21, next_agents21[0], next_agents21[1], next_agents21[2], next_action21)] = 0
		self.agent.q_dict[(next_loc22[0], next_loc22[1], next_heading22[0], next_heading22[1], next_light22, next_agents22[0], next_agents22[1], next_agents22[2], next_action22)] = -1.5
		self.agent.q_dict[(next_loc23[0], next_loc23[1], next_heading23[0], next_heading23[1], next_light23, next_agents23[0], next_agents23[1], next_agents23[2], next_action23)] = -1.5
		self.agent.q_dict[(next_loc3[0], next_loc3[1], next_heading3[0], next_heading3[1], next_light3, next_agents3[0], next_agents3[1], next_agents3[2], next_action3)] = -3

		assert self.agent.policy(now_loc, now_heading, now_light, now_agents) == 'right'

		self.agent.q_dict[(next_loc3[0], next_loc3[1], next_heading3[0], next_heading3[1], next_light3, next_agents3[0], next_agents3[1], next_agents3[2], next_action3)] = 10

		assert self.agent.policy(now_loc, now_heading, now_light, now_agents) == 'left'

		self.agent.q_dict[(next_loc22[0], next_loc22[1], next_heading22[0], next_heading22[1], next_light22, next_agents22[0], next_agents22[1], next_agents22[2], next_action22)] = 15

		assert self.agent.policy(now_loc, now_heading, now_light, now_agents) == 'forward'

		self.agent.q_dict[(next_loc10[0], next_loc10[1], next_heading10[0], next_heading10[1], next_light10, next_agents10[0], next_agents10[1], next_agents10[2], next_action10)] = 20

		assert self.agent.policy(now_loc, now_heading, now_light, now_agents) == None

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



