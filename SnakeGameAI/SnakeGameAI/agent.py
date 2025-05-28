import torch
import random
import numpy as np
from collections import deque
from env import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
  def __init__(self):
    self.n_games = 0
    self.epsilon = 0 #randomness
    self.discount = 0.9
    self.memory = deque(maxlen=MAX_MEMORY) #popleft
    self.model = Linear_QNet(11, 256, 3)
    self.trainer = QTrainer(self.model, lr = LR, discount=self.discount)

  def get_state(self, env):
    head = env.snake[0]

    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = env.direction == Direction.LEFT
    dir_r = env.direction == Direction.RIGHT
    dir_u = env.direction == Direction.UP
    dir_d = env.direction == Direction.DOWN

    state = [
      #Danger Straight
      (dir_r and env.is_collision(point_r)) or
      (dir_l and env.is_collision(point_l)) or
      (dir_u and env.is_collision(point_u)) or
      (dir_d and env.is_collision(point_d)),

      #Danger Right
      (dir_u and env.is_collision(point_r)) or
      (dir_d and env.is_collision(point_l)) or
      (dir_l and env.is_collision(point_u)) or
      (dir_r and env.is_collision(point_d)),

      #Danger Left
      (dir_d and env.is_collision(point_r)) or
      (dir_u and env.is_collision(point_l)) or
      (dir_r and env.is_collision(point_u)) or
      (dir_l and env.is_collision(point_d)),

      #Move direction
      dir_l,
      dir_r,
      dir_u,
      dir_d,

      #Food location
      env.food.x < env.head.x, #food left
      env.food.x > env.head.x, #food right
      env.food.y < env.head.y,  #food up
      env.food.y > env.food.y  #food down
    ]
    return np.array(state, dtype=int)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append([state, action, reward, next_state, done]) #popleft if MAX_MEMORY is reached

  def train_long_memory(self):
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    else:
      mini_sample = self.memory
    
    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, dones)

  def traing_short_memory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)


  def get_action(self, state):  
    self.epsilon = 80 - self.n_games
    final_move = [0,0,0]
    if random.randint(0,200) < self.epsilon: # random action (decay percentage of random action at each iteration)
      index = random.randint(0,2)
      final_move[index] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      assert prediction.shape[0] == 3, f"Output size mismatch: expected 3, got {prediction.shape[0]}"
      index = torch.argmax(prediction).item()
      final_move[index] = 1

    return final_move

def train():
  plot_scores = []
  plot_mean_scores = []
  total_score = 0
  record = 0
  agent = Agent()
  env = SnakeGameAI()

  while True:
    state_old = agent.get_state(env)
    #get move
    final_move = agent.get_action(state_old)
    #perform move and get new state
    reward, done, score = env.play_step(final_move)
    state_new = agent.get_state(env)

    agent.traing_short_memory(state_old, final_move, reward, state_new, done)

    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
      # train long memory
      env.reset()
      agent.n_games += 1
      agent.train_long_memory()

      if score > record:
        record = score
        agent.model.save()
      
      print('Game: ', agent.n_games, '| Score: ', score, '| Record: ', record)

      plot_scores.append(score)
      total_score += score
      mean_score = total_score / agent.n_games
      plot_mean_scores.append(mean_score)
      plot(plot_scores, plot_mean_scores)




if __name__ == '__main__':
  train()