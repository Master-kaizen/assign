# ui/chess_board.py
import pygame
import sys
import io
import cairosvg
from pathlib import Path
from copy import deepcopy

# import chess logic from engine (adjust to 'ENGINE' if your folder is uppercase)
from ENGINE.sensei_engine import (
    legal_moves_for,
    move_leaves_king_in_check,
    get_color,
    in_bounds,
    get_type,
)


# -- helpers to load SVG once
def load_svg_as_surface(svg_path, convert_size=None):
    png_bytes = cairosvg.svg2png(url=str(svg_path))
    surf = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
    if convert_size:
        surf = pygame.transform.smoothscale(surf, convert_size)
    return surf


# -- coordinate helpers
def pos_to_rc(x, y, start_x, start_y, tile_size):
    """mouse x,y -> row,col (row 0 top)"""
    col = (x - start_x) // tile_size
    row = (y - start_y) // tile_size
    if not in_bounds(row, col):
        return None
    return int(row), int(col)


# helper: create a temporary board with the dragged piece at its original source (used before generating legal moves)
def board_matrix_with_drag(board_matrix, dragging_piece, drag_from):
    tmp = deepcopy(board_matrix)
    sr, sc = drag_from
    tmp[sr][sc] = dragging_piece
    return tmp


# ---------------- main interactive function ----------------
def show_board(screen):
    WIDTH, HEIGHT = screen.get_size()
    tile_size = 60
    start_x, start_y = 140, 60
    clock = pygame.time.Clock()

    pieces_dir = Path("assets/pieces")
    piece_names = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    pieces = {}

    for name in piece_names:
        white_path = pieces_dir / f"white_{name}_up.svg"
        black_path = pieces_dir / f"black_{name}_dw.svg"
        pieces[f"white_{name}"] = load_svg_as_surface(white_path, convert_size=(tile_size, tile_size))
        pieces[f"black_{name}"] = load_svg_as_surface(black_path, convert_size=(tile_size, tile_size))

    back_rank = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
    board_matrix = []
    board_matrix.append([f"black_{p}" for p in back_rank])
    board_matrix.append([f"black_pawn" for _ in range(8)])
    for _ in range(4):
        board_matrix.append([None for _ in range(8)])
    board_matrix.append([f"white_pawn" for _ in range(8)])
    board_matrix.append([f"white_{p}" for p in back_rank])

    turn = "white"

    dragging = False
    drag_piece = None
    drag_from = None
    drag_offset = (0, 0)

    while True:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # --- start dragging
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                rc = pos_to_rc(mouse_x, mouse_y, start_x, start_y, tile_size)
                if rc:
                    r, c = rc
                    pk = board_matrix[r][c]
                    if pk and get_color(pk) == turn:
                        dragging = True
                        drag_piece = pk
                        drag_from = (r, c)
                        # offset so the piece draws centered under cursor
                        square_x = start_x + c * tile_size
                        square_y = start_y + r * tile_size
                        drag_offset = (square_x - mouse_x, square_y - mouse_y)
                        # temporarily remove from board for rendering while dragging
                        board_matrix[r][c] = None

            # --- dragging motion: no board change yet (visual only)
            elif event.type == pygame.MOUSEMOTION and dragging:
                pass  # mouse coords used later for drawing

            # --- release / place
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging:
                dragging = False
                rc = pos_to_rc(mouse_x, mouse_y, start_x, start_y, tile_size)
                if not rc:
                    # dropped outside board: snap back
                    sr, sc = drag_from
                    board_matrix[sr][sc] = drag_piece
                    drag_piece = None
                    drag_from = None
                else:
                    er, ec = rc
                    # basic target conditions: not same-color
                    target_key = board_matrix[er][ec]
                    if target_key and get_color(target_key) == get_color(drag_piece):
                        # cannot capture own piece
                        sr, sc = drag_from
                        board_matrix[sr][sc] = drag_piece
                        drag_piece = None
                        drag_from = None
                    else:
                        # Prepare a temp board with the piece at its source so move generation is correct
                        temp_board = board_matrix_with_drag(board_matrix, drag_piece, drag_from)

                        # get legal moves for this piece (uses engine logic)
                        legal = legal_moves_for(drag_piece, drag_from[0], drag_from[1], temp_board)

                        if (er, ec) in legal:
                            # ensure move won't leave own king in check (engine handles simulation)
                            leaves_check = move_leaves_king_in_check(
                                temp_board, drag_from[0], drag_from[1], er, ec
                            )
                            if leaves_check:
                                # illegal because leaves king in check
                                board_matrix[drag_from[0]][drag_from[1]] = drag_piece
                                drag_piece = None
                                drag_from = None
                            else:
                                # commit the move (capture if present)
                                board_matrix[er][ec] = drag_piece
                                print(f"Moved {drag_piece} from {drag_from} to {(er, ec)}")
                                # switch turn
                                turn = "black" if turn == "white" else "white"
                                print("Turn:", turn)
                                drag_piece = None
                                drag_from = None
                        else:
                            # illegal move, snap back
                            sr, sc = drag_from
                            board_matrix[sr][sc] = drag_piece
                            drag_piece = None
                            drag_from = None

        # --- render
        screen.fill((240, 240, 240))
        for row in range(8):
            for col in range(8):
                color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
                rect = pygame.Rect(start_x + col * tile_size,
                                   start_y + row * tile_size,
                                   tile_size, tile_size)
                pygame.draw.rect(screen, color, rect)
                key = board_matrix[row][col]
                if key:
                    surf = pieces.get(key)
                    if surf:
                        screen.blit(surf, rect.topleft)

        # draw dragged piece following cursor (on top)
        if dragging and drag_piece:
            surf = pieces.get(drag_piece)
            if surf:
                draw_x = mouse_x + drag_offset[0]
                draw_y = mouse_y + drag_offset[1]
                screen.blit(surf, (draw_x, draw_y))

        pygame.display.flip()
        clock.tick(60)
