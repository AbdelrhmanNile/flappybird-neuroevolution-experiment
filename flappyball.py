import pygame, random
from gakeras import Brain, Population, Individual
from tensorflow import keras
import time
import multiprocessing as mp
import numpy as np

pygame.init()
clock = pygame.time.Clock()

# Dimensions of the window
WIDTH, HEIGHT = 1080, 700

# Colors
BG = (29, 31, 33)
ACC_COLOR = (175, 175, 175)
counter = 0
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(BG)
g = 0.5  # Gravity
game_font = pygame.font.Font("freesansbold.ttf", 28)


def build_model():
    inputss = keras.layers.Input(shape=[5])
    x = keras.layers.Dense(
        10,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        name="dense1",
        input_shape=[5],
    )(inputss)
    x = keras.layers.Dense(
        8,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
    )(x)
    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
    )(x)
    return keras.Model(inputs=[inputss], outputs=[outputs])


def my_model():
    return keras.Sequential(
        [
            keras.layers.Dense(10, activation="relu", input_shape=[5]),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


# fitness function
def fitness_bird(bird):
    fitness = bird.score ** 2
    return fitness


# agent class
class Bird(Individual):
    def __init__(self, brain):
        super().__init__(brain)
        self.shape = pygame.Rect(100, HEIGHT / 2 - 15, 30, 30)
        self.vel_y = np.random.randint(-10, 10)
        self.dead = False
        self.color = [
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        ]

    def up(self):
        self.vel_y = -8

    def move(self):
        if not self.dead:
            self.vel_y += 0.5 * g
            self.shape.y += self.vel_y
            if self.shape.bottom >= HEIGHT:
                self.shape.y = HEIGHT - 30
                self.vel_y *= -0.7

            if self.shape.top <= 0:
                self.shape.y = 0
                self.vel_y *= -1

    def draw(self):
        if not self.dead:
            self.shape.y += self.vel_y
            self.vel_y += 0.5 * g
            pygame.draw.ellipse(
                screen,
                self.color,
                self.shape,
            )

    def reward(self):
        if not self.dead:
            self.score += 1

    def think(self, pipes):
        if not self.dead:
            if len(pipes) > 0:
                pipe = pipes[0]
                top = pipe[0].bottom / HEIGHT
                bottom = pipe[1].top / HEIGHT
                pipe_x = pipe[0].x / WIDTH
                inputs = [[self.shape.y / HEIGHT, top, bottom, pipe_x, self.vel_y]]
                output = self.brain.predict(inputs)
                if output[0] > 0.5:
                    self.up()


popul = Population(
    Individual_class=Bird,
    keras_functional_model=build_model,
    fitness_function=fitness_bird,
    population_size=20,
)

# Pipes
pipes = []
prev_x = WIDTH
pipe_height = 200  # The gap between the two pipes
pipe_vel = 3
pipe_intervals = (
    6000  # Defining the interval at which new pipes will be generated (in milliseconds)
)
# not used because the blocking call to the model affects the timing
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, pipe_intervals)


def generate_pipes():
    x = random.randint(50, 450)
    pipe1 = pygame.Rect(prev_x, 0, 25, x)
    pipe2 = pygame.Rect(
        prev_x, pipe1.bottom + pipe_height, 25, HEIGHT - pipe_height - x
    )
    return pipe1, pipe2


def start_game():
    global counter
    counter = 0
    SCORE = 0
    pipes.clear()
    while True:
        counter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        if counter % 70 == 0:
            pipe1, pipe2 = generate_pipes()
            pipes.append([pipe1, pipe2])
        for bird in pool.population:
            bird.move()
            # model inference and decision making happens here
            bird.think(pipes)
        screen.fill(BG)
        for bird in pool.population:
            bird.draw()
        for pipe_pair in pipes:
            for pipe in pipe_pair:
                pygame.draw.rect(screen, ACC_COLOR, pipe)
                pipe.x -= 3
                for bird in pool.population:
                    if pipe.colliderect(bird.shape):
                        bird.dead = True
                        if pool.all_dead():
                            pool.evolve(0.5, 0.1)
                            start_game()
                    elif (
                        pipe_pair[0].x < bird.shape.x and pipe_pair[1].x < bird.shape.x
                    ):
                        bird.reward()

                    if pipe_pair[0].x < bird.shape.x:
                        pipes.pop(0)
        pygame.display.update()
        clock.tick(60)


start_game()
