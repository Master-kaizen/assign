# ui/menu_screen.py
import pygame
import sys
from UI.chess_board import show_board

pygame.init()

WHITE = (240, 240, 240)
BLACK = (0, 0, 0)
GREY = (180, 180, 180)

FONT = pygame.font.SysFont("arial", 40)
SMALL_FONT = pygame.font.SysFont("arial", 30)

class Button:
    def __init__(self, text, pos, size, screen):
        self.text = text
        self.rect = pygame.Rect(pos, size)
        self.screen = screen
        self.color = GREY
        self.hover_color = (120, 120, 120)

    def draw(self):
        mouse = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(self.screen, color, self.rect, border_radius=10)
        text_surf = SMALL_FONT.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        self.screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


def fade_in(screen):
    fade_surface = pygame.Surface(screen.get_size())
    fade_surface.fill((0, 0, 0))
    for alpha in range(255, -1, -25):
        fade_surface.set_alpha(alpha)
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(20)


def draw_text(screen, text, pos):
    render = FONT.render(text, True, (0, 0, 0))
    screen.blit(render, pos)


def main_menu():
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Sensei")




    fade_in(screen)
    clock = pygame.time.Clock()
    running = True

    two_player_btn = Button("2 Players", (300, 350), (200, 60), screen)
    sensei_btn = Button("Play with Sensei", (270, 440), (260, 60), screen)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if two_player_btn.is_clicked(event):
                print("2 Players selected!")
                show_board(screen)

            elif sensei_btn.is_clicked(event):
                print("Play with Sensei selected!")
                show_board(screen)

        screen.fill(WHITE)
        draw_text(screen, "Welcome to Chess Sensei", (160, 200))
        two_player_btn.draw()
        sensei_btn.draw()
        pygame.display.flip()
        clock.tick(60)
