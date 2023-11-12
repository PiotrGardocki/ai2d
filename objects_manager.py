import itertools


class ObjectsManager():
    enemy_objects = []
    player = None

    def add_player(self, player):
        self.player = player

    def add_enemy_object(self, enemy_object):
        self.enemy_objects.append(enemy_object)

    def __all_objects(self):
        return itertools.chain(self.enemy_objects, (self.player,))

    def update(self, dt):
        for object in self.__all_objects():
            object.update(dt)

    def draw(self, screen):
        for object in self.__all_objects():
            object.draw(screen)

    def does_player_collide(self):
        enemies_rects = [enemy.get_rect() for enemy in self.enemy_objects]

        return self.player.get_rect().collidelist(enemies_rects) != -1
