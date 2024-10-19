# ML
Reinforcement Learning (RL) has emerged as a pivotal area within machine learning, enabling agents to learn optimal behaviors through interactions with their environments. Two fundamental algorithms in this domain are Q-learning and SARSA (State-Action-Reward-State-Action). Understanding these algorithms is crucial for anyone looking to delve into RL, as they form the basis for more complex methods used in various applications, from robotics to game playing.
Overview of Q-learning and SARSA
Q-learning
Q-learning is an off-policy algorithm that aims to learn the value of the optimal policy regardless of the agent's actions. It updates its Q-values based on the maximum expected future rewards, which means it looks ahead to determine the best possible action in the next state. This characteristic allows Q-learning to be more aggressive in exploring the environment, often leading to faster convergence towards an optimal policy.
The update rule for Q-learning can be expressed mathematically as:
Q
(
s
t
,
a
t
)
←
Q
(
s
t
,
a
t
)
+
α
(
r
t
+
γ
max
⁡
a
Q
(
s
t
+
1
,
a
)
−
Q
(
s
t
,
a
t
)
)
Q(s 
t
​
 ,a 
t
​
 )←Q(s 
t
​
 ,a 
t
​
 )+α(r 
t
​
 +γ 
a
max
​
 Q(s 
t+1
​
 ,a)−Q(s 
t
​
 ,a 
t
​
 ))
where:
s
t
s 
t
​
  is the current state,
a
t
a 
t
​
  is the action taken,
r
t
r 
t
​
  is the reward received,
s
t
+
1
s 
t+1
​
  is the next state,
α
α is the learning rate,
γ
γ is the discount factor.
SARSA
In contrast, SARSA is an on-policy algorithm that updates its Q-values based on the actions actually taken by the agent. This means that SARSA learns about the policy it is currently following rather than aiming for an optimal policy directly. The update rule for SARSA is given by:
Q
(
s
t
,
a
t
)
←
Q
(
s
t
,
a
t
)
+
α
(
r
t
+
γ
Q
(
s
t
+
1
,
a
t
+
1
)
−
Q
(
s
t
,
a
t
)
)
Q(s 
t
​
 ,a 
t
​
 )←Q(s 
t
​
 ,a 
t
​
 )+α(r 
t
​
 +γQ(s 
t+1
​
 ,a 
t+1
​
 )−Q(s 
t
​
 ,a 
t
​
 ))
where 
a
t
+
1
a 
t+1
​
  is the action taken in the next state according to the current policy. This approach makes SARSA more stable and conservative compared to Q-learning.
Key Differences Between Q-learning and SARSA
Exploration vs. Exploitation
One of the primary differences between these two algorithms lies in their exploration strategies. Q-learning tends to explore more aggressively since it always updates its values based on the maximum possible reward from future states. This can lead to faster learning but may also result in suboptimal policies if not managed properly.
On the other hand, SARSA balances exploration and exploitation more conservatively by updating its values based on actions taken according to its current policy. This makes it particularly useful in environments where safety and stability are paramount.
Update Rules
The update rules reflect their differing philosophies:
Q-learning uses a greedy approach by focusing solely on maximizing future rewards.
SARSA, however, incorporates the actual actions taken into its updates, making it sensitive to the current policy's performance.
Convergence Rates
In practice, both algorithms can converge to optimal policies under certain conditions; however, they do so at different rates. Q-learning often converges faster due to its aggressive exploration but may oscillate or diverge in unstable environments. Conversely, SARSA tends to have a more stable learning process due to its on-policy nature but may take longer to reach optimality.
Practical Implementation
To illustrate these concepts further, let’s consider practical implementations of both algorithms using Python and OpenAI’s Gym library. We will use a simple environment like CartPole.
Environment Setup
python
import gym
import numpy as np

# Create CartPole environment
env = gym.make('CartPole-v1')

Q-learning Agent Implementation
python
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state

SARSA Agent Implementation
python
class SarsaAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        next_action = self.choose_action(next_state)
        td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                action = next_action

Conclusion
In conclusion, both Q-learning and SARSA are foundational algorithms in reinforcement learning with distinct characteristics that make them suitable for different scenarios. While Q-learning excels in aggressive exploration and faster convergence towards optimal policies in dynamic environments like gaming or robotics, SARSA provides stability and safety in uncertain environments such as healthcare or traffic management. Understanding these differences helps practitioners choose the right algorithm based on their specific needs and constraints.
By implementing these algorithms and experimenting with various parameters and environments, one can gain deeper insights into their workings and applications within reinforcement learning frameworks.
