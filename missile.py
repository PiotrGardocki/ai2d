import pygame


class Missile:
    speed = 250

    def __init__(self, position) -> None:
        self.rect = pygame.Rect(position, (20, 20))

    def update(self, dt):
        self.rect.y += self.speed * dt

    def get_rect(self):
        return self.rect

    def draw(self, screen):
        pygame.draw.rect(screen, pygame.Color(0, 0, 255), self.rect)
