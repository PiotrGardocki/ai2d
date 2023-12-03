import itertools
from pygame import Rect
from player import PlayerAction


class ObjectsManager():
    def __init__(self, window_rect: Rect) -> None:
        self.__enemy_objects = []
        self.__player = None
        self.__window_rect = window_rect

    def add_player(self, player):
        self.__player = player

    def add_enemy_object(self, enemy_object):
        self.__enemy_objects.append(enemy_object)

    def __all_objects(self):
        return itertools.chain(self.__enemy_objects, (self.__player,))

    def update(self, dt, player_action=PlayerAction.STAY):
        for object in self.__enemy_objects:
            object.update(dt)

        enemies = self.__enemy_objects
        enemies = [e for e in enemies if self.__window_rect.bottom > e.rect.top]
        self.__enemy_objects = enemies

        self.__player.update(dt, player_action)

    def draw(self, screen):
        for object in self.__all_objects():
            object.draw(screen)

    def does_player_collide(self):
        enemies_rects = [enemy.get_rect() for enemy in self.__enemy_objects]

        return self.__player.get_rect().collidelist(enemies_rects) != -1
