from maspy import *
from maspy.learning import *
import pygame
import random
from itertools import product

pygame.init()
font = pygame.font.SysFont('arial', 25)

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20

class Map(Environment):
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.max_col = w // BLOCK_SIZE
        self.max_row = h // BLOCK_SIZE
        self.score = 0
        
        # Criar percepts
        self.create(Percept("head_location", (self.max_col, self.max_row), cartesian))
        self.create(Percept("food_location", (self.max_col, self.max_row), cartesian))
        #self.create(Percept("snake_body", [(10, 15), (10, 14), (10, 13)], listed))
        self.create(Percept("snake_body", (self.max_col,self.max_row,2)), cartesian)

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.change(Percept("head_location"), (10, 15))
        self.change(Percept("food_location"), (16, 12))
        self.change(Percept("snake_body"), [(10, 15,1), (10, 14,1), (10, 13,1)])
        
        self.score = 0
        self.frame_iteration = 0
    
    def place_food(self, state):
        while True:
            new_food = (random.randint(0, self.max_col - 1), random.randint(0, self.max_row - 1))
            if new_food not in state["snake_body"]:
                break
        self.change(Percept("food_location"), new_food)
    
    def play_step(self, state, action):
        reward = 0
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        #state = self.get_current_state()
        new_state, reward, done = self.move_transition(state, action)
        
        if not done:
            self.apply_action(new_state)
        else:
            self.reset()
            return reward, done, self.score
        
        self._update_ui(state)
        self.clock.tick(SPEED)
        return state, reward, done
    
    def get_current_state(self):
        return {
            "head_location": self.get(Percept("head_location", (Any, Any))),
            "snake_body": self.get(Percept("snake_body", [(Any, Any)])),
            "food_location": self.get(Percept("food_location", (Any, Any)))
        }
    
    def apply_action(self, new_state):
        self.change(Percept("head_location"), new_state["head_location"])
        self.change(Percept("snake_body"), new_state["snake_body"])
    
    def move_transition(self, state, direction):
        snake_body = state["snake_body"][:]
        new_head = self.calculate_new_position(state["head_location"], direction)
        snake_body.insert(0, new_head)
        
        reward, done = -0.1, False
        
        if self.is_collision(new_head, snake_body):
            return state, -10, True
        elif new_head == state["food_location"]:
            reward = 10
            self.place_food(state)
        else:
            snake_body.pop()
        
        state["head_location"] = new_head
        state["snake_body"] = snake_body
        return state, reward, done
    
    def calculate_new_position(self, head, direction):
        x, y = head
        if direction == "UP":
            return (x, y - 1)
        elif direction == "DOWN":
            return (x, y + 1)
        elif direction == "LEFT":
            return (x - 1, y)
        elif direction == "RIGHT":
            return (x + 1, y)
        return head
    
    def is_collision(self, head, snake_body):
        return (
            head[0] < 0 or head[0] >= self.max_col or
            head[1] < 0 or head[1] >= self.max_row or
            head in snake_body[1:]
        )
    
    def _update_ui(self, state):
        self.display.fill(BLACK)
        for pt in state["snake_body"]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0] * BLOCK_SIZE, pt[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0] * BLOCK_SIZE + 4, pt[1] * BLOCK_SIZE + 4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(state["food_location"][0] * BLOCK_SIZE, state["food_location"][1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    @action(listed, ("UP", "DOWN", "LEFT", "RIGHT"), play_step)
    def move(self, agt, direction):
        state = self.get_current_state()
        new_state, reward, done = self.move_transition(state, direction)
        if not done:
            self.apply_action(new_state)
        return reward, done