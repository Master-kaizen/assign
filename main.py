import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 800, 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Sensei")

WHITE = (240, 240, 240)
FONT = pygame.font.SysFont("arial", 40)

def draw_text(text, pos):
    render = FONT.render(text, True, (0, 0, 0))
    SCREEN.blit(render, pos)

def main():
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        SCREEN.fill(WHITE)
        draw_text("Welcome to Chess Sensei", (150, 350))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


fade_surface = pygame.Surface((800, 800))
fade_surface.fill((0, 0, 0))
for alpha in range(0, 255, 5):
    fade_surface.set_alpha(alpha)
    SCREEN.blit(fade_surface, (0, 0))
    pygame.display.update()
    pygame.time.delay(30)


SCREEN = pygame.display.set_mode((720, 720))  # try smaller, like 720x720


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # draw here
    pygame.display.update()

pygame.quit()



for event in pygame.event.get():
    if event.type == pygame.QUIT:
        running = False





if __name__ == "__main__":
    main()
