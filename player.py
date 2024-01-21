import pygame
from enum import Enum


PlayerAction = Enum('PlayerAction', ('STAY', 'LEFT', 'RIGHT'))


class Player:
    speed = 300

    def __init__(self, screen_size) -> None:
        self.rect = pygame.Rect(0, 0, 50, 50)
        self.rect.center = pygame.Vector2(screen_size[0] / 2,
                                          screen_size[1] - self.rect.height)
        self.screen_size = screen_size

    def update(self, dt, action):
        match action:
            case PlayerAction.LEFT:
                self.rect.x -= self.speed * dt
            case PlayerAction.RIGHT:
                self.rect.x += self.speed * dt

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.screen_size[0]:
            self.rect.right = self.screen_size[0]

    def get_rect(self):
        return self.rect

    def draw(self, screen):
        pygame.draw.rect(screen, pygame.Color(0, 0, 0), self.rect)
