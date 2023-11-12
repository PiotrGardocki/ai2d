import pygame
from player import Player
from missile import Missile

pygame.init()
screen = pygame.display.set_mode((720, 720))
clock = pygame.time.Clock()
running = True
dt = 0

player = Player(screen.get_size())
missile1 = Missile((10, 10))
missile2 = Missile((40, 10))
missile3 = Missile((70, 10))

objects = [player, missile1, missile2, missile3]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("cyan2")

    for object in objects:
        object.update(dt)
    for object in objects:
        object.draw(screen)

    pygame.display.flip()

    dt = clock.tick(60) / 1000

pygame.quit()
