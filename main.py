import pygame
#import pymunk
# Import pymunk for physics

# Initialize Pygame
pygame.init()

# Set up the game window
window = pygame.display.set_mode((600, 600))

pygame.draw.circle(window, (255, 0, 0), (300, 300), 170, 0)

#update the display
pygame.display.update()

# Quit Pygame
