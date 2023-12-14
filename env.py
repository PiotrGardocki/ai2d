import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import pygame
import numpy as np

from objects_manager import ObjectsManager
from player import Player, PlayerAction
from missile import Missile


class SpaceInvaders2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    window_size = 720
    fps_limit = 60

    def __init__(self, render_mode=None):
        shape = (self.window_size, self.window_size, 3)
        self.observation_space = spaces.Box(0, 255, shape, np.uint8)
        self.action_space = spaces.Discrete(3)

        self.__action_mapping = {
            0: PlayerAction.STAY,
            1: PlayerAction.LEFT,
            2: PlayerAction.RIGHT,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.__window = None
        self.__clock = None
        self.__canvas = None

        self.__enemies_spawn_interval = 1.5
        self.__time_since_last_spawn = 0
        self.__enemies_to_spawn = 3

    @staticmethod
    def create_human_game():
        env = SpaceInvaders2Env("human")
        env.reset()
        return env

    def __get_state(self):
        return pygame.surfarray.pixels3d(self.__canvas).transpose(1, 0, 2)

    def __get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pygame.init()

        if self.render_mode == "human":
            self.__window = pygame.display.set_mode((self.window_size, self.window_size))
            self.__clock = pygame.time.Clock()

        self.__running = True
        self.__dt = 0
        self.__time_since_last_spawn = self.__enemies_spawn_interval

        player = Player((self.window_size, self.window_size))
        objects = ObjectsManager(pygame.Rect(0, 0, self.window_size, self.window_size))
        objects.add_player(player)
        self.__objects = objects

        self.__render_frame()

        observation = self.__get_state()
        info = self.__get_info()

        return observation, info

    def __spawn_random_enemies(self):
        left_limit = 10
        right_limit = self.window_size - 10

        for _ in range(self.__enemies_to_spawn):
            x = self.np_random.integers(left_limit, right_limit)
            self.__objects.add_enemy_object(Missile((x, 10)))

    def __frame_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__running = False

        self.__objects.update(self.__dt, action)

        self.__time_since_last_spawn += self.__dt
        if self.__time_since_last_spawn > self.__enemies_spawn_interval:
            self.__time_since_last_spawn -= self.__enemies_spawn_interval
            self.__spawn_random_enemies()

        if self.__objects.does_player_collide():
            self.__running = False

        if self.render_mode == "human":
            self.__dt = self.__clock.tick(self.fps_limit) / 1000
        else:
            self.__dt = 1 / self.fps_limit

    def run_human_game(self):
        assert self.render_mode == "human"

        while self.__running:
            action = PlayerAction.STAY
            keys = pygame.key.get_pressed()

            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = PlayerAction.LEFT
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = PlayerAction.RIGHT

            self.__frame_step(action)
            self.__render_frame()

        self.close()

    def step(self, action):
        if self.__running:
            self.__frame_step(self.__action_mapping[action])

        terminated = not self.__running
        reward = 0 if terminated else 1

        if self.render_mode == "human":
            self.__render_frame()

        observation = self.__get_state()
        info = self.__get_info()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.__render_frame()

    def __render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        self.__canvas = canvas

        canvas.fill("cyan2")
        self.__objects.draw(canvas)

        if self.render_mode == "human":
            self.__window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
        else:
            self.__get_state()

    def close(self):
        self.__objects = None
        self.__window = None
        self.__clock = None
        pygame.quit()


register(
     id="games/SpaceInvaders2",
     entry_point="env:SpaceInvaders2Env",
)
