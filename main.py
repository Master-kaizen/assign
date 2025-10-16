# main.py
import pygame
import sys
from UI.menu_screen import main_menu

if __name__ == "__main__":
    pygame.init()
    main_menu()  # run the game menu
    pygame.quit()
    sys.exit()
