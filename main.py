import pygame
#from pygame.locals import *
#import pymunk
# Import pymunk for physics

# Initialize Pygame
pygame.init()

# Set up the game window
window = pygame.display.set_mode((600, 600))

pygame.draw.circle(window, (255, 0, 0), (300, 300), 170, 0)

#update the display
pygame.display.update()

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
pygame.quit()
