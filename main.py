import pygame
from player import Player

pygame.init()
screen = pygame.display.set_mode((720, 720))
clock = pygame.time.Clock()
running = True
dt = 0

player = Player(screen.get_size())

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("blue")

    player.update(dt)
    player.draw(screen)

    pygame.display.flip()

    dt = clock.tick(60) / 1000

pygame.quit()
