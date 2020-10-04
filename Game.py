from GeneticAlgorithm import Population
from Pipe import Pipe
from Base import Base
import numpy as np
import pygame
import os


# Assets
pygame.font.init()
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bg.png')))
STAT_FONT = pygame.font.SysFont("comicsans", 50)


def draw_window(win, base, birds, pipes, score, gen):
    win.blit(BG_IMG, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    text1 = STAT_FONT.render("Score: {}".format(score), 1, (255, 255, 255))
    text2 = STAT_FONT.render("Generation: {}".format(gen), 1, (255, 255, 255))
    win.blit(text1, (500 - 10 -text1.get_width(), 10))
    win.blit(text2, (500 - 10 -text2.get_width(), 30))

    for bird in birds:
        bird.draw(win)

    base.draw(win)
    pygame.display.update()

def main():
    previous = []
    clock = pygame.time.Clock()
    score = 0
    pipes = [Pipe(600)]
    inputs = np.array([
        0,
        pipes[-1].x - pipes[-1].PIPE_TOP.get_width(),
        pipes[-1].top + pipes[-1].PIPE_TOP.get_height(),
        pipes[-1].bottom,
    ])
    population = Population(inputs)
    birds = population.members[:]
    birds[-1].nn.theta = np.array(
        [[ 0.76314658, 0.3645334, 0.99851746, -0.15675087],
        [-0.88392422, -0.38872669, 0.66258692, 0.94892011],
        [-0.26858695, -0.03098821, 0.0439891, -0.21977436],
        [ 0.23092167, 0.57162172, 0.15491268, -0.22485136],
        [ 0.50117003, 0.04667388, -0.08010465, 0.11969299],
        [-0.83630789, 0.97021094, 0.21263142, 0.35092167]]
    )
    base = Base(730)
    win = pygame.display.set_mode((500, 800))
    run = True

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if len(birds) == 0:
            population.previous = previous[:]
            previous = []
            pipes = [Pipe(600)]
            inputs = np.array([
                0,
                pipes[-1].x - pipes[-1].PIPE_TOP.get_width(),
                pipes[-1].top + pipes[-1].PIPE_TOP.get_height(),
                pipes[-1].bottom,
            ])
            population.reseed(inputs)
            birds = population.members[:]
            base = Base(730)
            population.generation += 1
            score = 0

        rem_birds = []
        
        for bird in birds:
            if bird.x > pipes[0].x:
                closest_pipe = pipes[-1]
            else:
                closest_pipe = pipes[0]
            closest_pipe = min(pipes, key=lambda pipe: abs(pipe.x - bird.x))
            # closest_pipe = pipes[-1]
            inputs = np.array([
                bird.y - closest_pipe.middle,
                closest_pipe.x - bird.x,
                closest_pipe.top + closest_pipe.PIPE_TOP.get_height(),
                closest_pipe.bottom,
            ])
            bird.nn.X = inputs
            bird.move(closest_pipe.middle)
            
            if bird.y + bird.img.get_height() > 730 or bird.y + bird.img.get_height() < 0:
                # bird.alive -= 100
                rem_birds.append(bird)

        base.move()

        add_pipe = False
        rem_pipes = []
        for pipe in pipes:
            for bird in birds:
                if pipe.collide(bird):
                    rem_birds.append(bird)

                if not pipe.passed and pipe.x < bird.x:
                    # if bird.x > pipe.x + pipe.PIPE_BOTTOM.get_width():
                    # bird.alive += 100
                    bird.pipes_passed += 1
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem_pipes.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            pipes.append(Pipe(600))

        for r in rem_pipes:
            pipes.remove(r)
        
        for r in rem_birds:
            try:
                previous.append(r)
                birds.remove(r)
            except:
                pass

        draw_window(win, base, birds, pipes, score, population.generation)

    pygame.quit()
    quit()

main()
