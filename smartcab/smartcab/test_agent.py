
from agent import LearningAgent
# from agent import coord_convert
# from agent import delta_convert
from environment import Environment
import numpy as np
from numbers import Number

# Test failing due to the newly added ETL process and change of gamma.

class TestAgent:
	def setup_method(self, method):
		print("Setting up METHOD {0}".format(method.__name__))
		self.e = Environment()
		self.agent = self.e.create_agent(LearningAgent)
		self.e.set_primary_agent(self.agent, enforce_deadline=False)

	def test_q_dict(self):
		# q_dict is (next_way_point, tl, oa_oc, oa_lt, oa_rt, act): [q_value, t]
		print("Testing q_dict...")
		assert self.agent.q_dict['forward', 'green', 'forward', None, 'right', 'forward'] == (0, 0)
		assert self.agent.q_dict['right', 'red', 'right', None, 'left', None] == (0, 0)
		assert self.agent.q_dict['left', 'red', 'forward', 'left', None, 'left'] == (0, 0)
		assert self.agent.q_dict[None, 'green', None, 'right', 'forward', 'right'] == (0, 0)

	def test_max_q_output_type(self):
		self.agent.q_dict[('forward', 'green', 'forward', 'right', 'left', 'right')] = (4, 0)
		self.agent.q_dict[('forward', 'green', 'forward', 'right', 'left', 'forward')] = (-2, 1)
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', None)] = (2, 0)
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', 'left')] = (0, 1)
		self.agent.q_dict[('right', 'green', 'forward', 'right', 'left', 'left')] = (2, 0)

		assert isinstance(self.agent.max_q('forward', 'green', ['forward', 'right', 'left'])[1], Number)
		assert isinstance(self.agent.max_q('forward', 'green', ['forward', 'right', 'left'])[2], Number)

	def test_max_q_output_value(self): # return (action, q_value, t)
		self.agent.q_dict[('forward', 'green', 'forward', 'right', 'left', 'right')] = (4, 0)
		self.agent.q_dict[('forward', 'green', 'forward', 'right', 'left', 'forward')] = (-2, 1)
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', None)] = (2, 3)
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', 'left')] = (0, 1)
		self.agent.q_dict[('right', 'green', None, 'right', None, 'left')] = (1, 0)

		assert self.agent.max_q('forward', 'green', ['forward', 'right', 'left'])[0] == 'right'
		assert self.agent.max_q('forward', 'green', ['forward', 'right', 'left'])[1] == 4
		assert self.agent.max_q('forward', 'green', ['forward', 'right', 'left'])[2] == 0

		assert self.agent.max_q('left', 'green', ['forward', 'right', 'left'])[0] == None
		assert self.agent.max_q('left', 'green', ['forward', 'right', 'left'])[1] == 2
		assert self.agent.max_q('left', 'green', ['forward', 'right', 'left'])[2] == 3

		assert self.agent.max_q('right', 'green', [None, 'right', None])[0] == 'left'
		assert self.agent.max_q('right', 'green', [None, 'right', None])[1] == 1
		assert self.agent.max_q('right', 'green', [None, 'right', None])[2] == 0

	def test_q_update_output_value(self):
		# Prev state/action 1
		self.agent.prev_reward = 5
		self.agent.prev_waypoint = 'forward'
		self.agent.prev_light = 'green'
		self.agent.prev_agents = ['right', 'left', 'forward']
		self.agent.prev_action = 'forward'

		self.agent.q_dict[(self.agent.prev_waypoint, self.agent.prev_light, self.agent.prev_agents[0], self.agent.prev_agents[1], self.agent.prev_agents[2], self.agent.prev_action)] = (2, 0)

		# Now state 1
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', 'right')] = [3, 4]

		# Possibilities 1
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', 'forward')] = [-2, 3]
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', None)] = [0, 1]
		self.agent.q_dict[('left', 'green', 'forward', 'right', 'left', 'left')] = [12, 8]

		assert self.agent.q_update('left', 'green', ['forward', 'right', 'left']) \
		== (15.8, 1)

		# Prev state 2
		self.agent.prev_reward = 2
		self.agent.prev_waypoint = 'right'
		self.agent.prev_light = 'green'
		self.agent.prev_agents = [None, None, 'right']
		self.agent.prev_action = 'left'

		self.agent.q_dict[(self.agent.prev_waypoint, self.agent.prev_light, self.agent.prev_agents[0], self.agent.prev_agents[1], self.agent.prev_agents[2], self.agent.prev_action)] = (-1, 3)

		# Now state 2
		now_waypoint = 'forward'
		now_light = 'red'
		now_agents = [None, 'forward', None]

		# Possibilities 2
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'forward')] = (3, 1)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'right')] = (2, 2)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'left')] = (0, 0)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], None)] = (-1, 1)

		result = self.agent.q_update(now_waypoint, now_light, now_agents)

		assert abs(result[0] - 0.425) < 0.00001
		assert result[1] == 4

		# Prev state 3
		self.agent.prev_reward = 5
		self.agent.prev_waypoint = 'left'
		self.agent.prev_light = 'red'
		self.agent.prev_agents = [None, 'right', 'forward']
		self.agent.prev_action = 'forward'

		self.agent.q_dict[(self.agent.prev_waypoint, self.agent.prev_light, self.agent.prev_agents[0], self.agent.prev_agents[1], self.agent.prev_agents[2], self.agent.prev_action)] = (-2.5, 5)

		# Now state 3
		now_waypoint = 'forward'
		now_light = 'red'
		now_agents = ['forward', 'forward', None]

		# Possibilities 3
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'forward')] = (-1, 0)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'right')] = (2, 4)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'left')] = (-2, 2)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], None)] = (-1, 1)

		result = self.agent.q_update(now_waypoint, now_light, now_agents)

		assert abs(result[0] + 0.95) < 0.00001
		assert result[1] == 6

	def test_policy_output_type(self):
		self.agent.q_dict[('right', 'green', 'forward', 'right', 'left', 'right')] = (18, 1)
		self.agent.q_dict[('right', 'green', 'forward', 'right', 'left', 'forward')] = (-2, 0)
		self.agent.q_dict[('right', 'green', 'forward', 'right', 'left', None)] = (0, 2)
		self.agent.q_dict[('right', 'green', 'forward', 'right', 'left', 'left')] = (12, 5)

		assert type(self.agent.policy('right', 'green', ['forward', 'right', 'left'])) == type('string') or type(None)

	def test_policy_output_value(self):
		# Now state 1
		now_waypoint = 'forward'
		now_light = 'red'
		now_agents = ['forward', 'forward', None]

		# Possibilities 1
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'forward')] = (3, 1)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'right')] = (2, 0)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'left')] = (1, 2)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], None)] = (5, 1)

		assert self.agent.policy(now_waypoint, now_light, now_agents) == None

		# Now state 2
		now_waypoint = 'forward'
		now_light = 'green'
		now_agents = ['left', 'forward', 'left']

		# Possibilities 2
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'forward')] = (-1, 1)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'right')] = (-2, 0)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], 'left')] = (-1, 2)
		self.agent.q_dict[(now_waypoint, now_light, now_agents[0], now_agents[1], now_agents[2], None)] = (-5, 1)

		assert self.agent.policy(now_waypoint, now_light, now_agents) == 'left' or self.agent.policy(now_waypoint, now_light, now_agents) == 'forward'