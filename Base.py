import pygame
import os


BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','base.png')))

class Base:
    VEL = 5

    def __init__(self, y):
        self.img = BASE_IMG
        self.WIDTH = self.img.get_width()
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    
    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))