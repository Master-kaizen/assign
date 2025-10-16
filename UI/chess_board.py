# ui/chess_board.py
import pygame
import sys
import io
import cairosvg
from pathlib import Path
from copy import deepcopy
from ENGINE.sensei_engine import legal_moves_for, move_leaves_king_in_check


# -- helpers to load SVG once
def load_svg_as_surface(svg_path, convert_size=None):
    png_bytes = cairosvg.svg2png(url=str(svg_path))
    surf = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
    if convert_size:
        surf = pygame.transform.smoothscale(surf, convert_size)
    return surf


# -- coordinate helpers
def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def pos_to_rc(x, y, start_x, start_y, tile_size):
    """mouse x,y -> row,col (row 0 top)"""
    col = (x - start_x) // tile_size
    row = (y - start_y) // tile_size
    if not in_bounds(row, col):
        return None
    return int(row), int(col)

def get_color(piece_key):
    if not piece_key:
        return None
    return piece_key.split("_")[0]  # "white" or "black"

def get_type(piece_key):
    if not piece_key:
        return None
    return piece_key.split("_")[1]  # "pawn","rook",etc.


# -- attack generation for check detection (pseudo-legal attacks)
def attacked_squares(board_matrix, by_color):
    attacks = set()
    for r in range(8):
        for c in range(8):
            key = board_matrix[r][c]
            if not key:
                continue
            if get_color(key) != by_color:
                continue
            ptype = get_type(key)
            if ptype == "pawn":
                # pawns attack diagonally forward relative to color
                dr = -1 if by_color == "white" else 1
                for dc in (-1, 1):
                    rr, cc = r + dr, c + dc
                    if in_bounds(rr, cc):
                        attacks.add((rr, cc))
            elif ptype == "knight":
                for dr, dc in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
                    rr, cc = r+dr, c+dc
                    if in_bounds(rr, cc):
                        attacks.add((rr, cc))
            elif ptype in ("rook", "queen", "bishop"):
                directions = []
                if ptype in ("rook","queen"):
                    directions += [(1,0),(-1,0),(0,1),(0,-1)]
                if ptype in ("bishop","queen"):
                    directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
                for dr, dc in directions:
                    rr, cc = r+dr, c+dc
                    while in_bounds(rr, cc):
                        attacks.add((rr, cc))
                        if board_matrix[rr][cc] is not None:
                            break
                        rr += dr
                        cc += dc
            elif ptype == "king":
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r+dr, c+dc
                        if in_bounds(rr, cc):
                            attacks.add((rr, cc))
    return attacks


# -- move generation for each piece (returns list of target (r,c) tuples)
def legal_moves_for(piece_key, r, c, board_matrix):
    color = get_color(piece_key)
    ptype = get_type(piece_key)
    moves = []

    if ptype == "pawn":
        direction = -1 if color == "white" else 1
        # single step
        rr = r + direction
        if in_bounds(rr, c) and board_matrix[rr][c] is None:
            moves.append((rr, c))
            # double step from starting rank
            start_row = 6 if color == "white" else 1
            rr2 = r + (2 * direction)
            if r == start_row and in_bounds(rr2, c) and board_matrix[rr2][c] is None:
                moves.append((rr2, c))
        # captures
        for dc in (-1, 1):
            rc, cc = r + direction, c + dc
            if in_bounds(rc, cc) and board_matrix[rc][cc] is not None and get_color(board_matrix[rc][cc]) != color:
                moves.append((rc, cc))

    elif ptype == "knight":
        for dr, dc in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
            rr, cc = r+dr, c+dc
            if in_bounds(rr, cc) and (board_matrix[rr][cc] is None or get_color(board_matrix[rr][cc]) != color):
                moves.append((rr, cc))

    elif ptype in ("rook", "bishop", "queen"):
        directions = []
        if ptype in ("rook","queen"):
            directions += [(1,0),(-1,0),(0,1),(0,-1)]
        if ptype in ("bishop","queen"):
            directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dr, dc in directions:
            rr, cc = r+dr, c+dc
            while in_bounds(rr, cc):
                if board_matrix[rr][cc] is None:
                    moves.append((rr, cc))
                else:
                    if get_color(board_matrix[rr][cc]) != color:
                        moves.append((rr, cc))  # capture
                    break
                rr += dr
                cc += dc

    elif ptype == "king":
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r+dr, c+dc
                if in_bounds(rr, cc) and (board_matrix[rr][cc] is None or get_color(board_matrix[rr][cc]) != color):
                    moves.append((rr, cc))
        # no castling implemented

    return moves


# -- check if moving piece from (sr,sc) to (er,ec) would leave own king in check
def move_leaves_king_in_check(board_matrix, sr, sc, er, ec):
    # simulate move on a deep copy and test if the moving side's king is attacked
    sim = deepcopy(board_matrix)
    moving_piece = sim[sr][sc]
    sim[er][ec] = moving_piece
    sim[sr][sc] = None

    color = get_color(moving_piece)
    # find king pos
    king_pos = None
    for r in range(8):
        for c in range(8):
            k = sim[r][c]
            if k and get_type(k) == "king" and get_color(k) == color:
                king_pos = (r, c)
                break
        if king_pos:
            break
    if not king_pos:
        # shouldn't happen; treat as invalid (king missing)
        return True

    opponent = "white" if color == "black" else "black"
    opp_attacks = attacked_squares(sim, opponent)
    return king_pos in opp_attacks


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
                        # is move in piece's legal moves?
                        legal = legal_moves_for(drag_piece, drag_from[0], drag_from[1], board_matrix_with_drag(board_matrix, drag_piece, drag_from))
                        # But legal_moves_for expects original piece on board; we have it removed.
                        # Create a temp board with the piece at its source so move generation is correct:
                        temp_board = deepcopy(board_matrix)
                        temp_board[drag_from[0]][drag_from[1]] = drag_piece
                        legal = legal_moves_for(drag_piece, drag_from[0], drag_from[1], temp_board)

                        if (er, ec) in legal:
                            # also ensure move won't leave king in check
                            # simulate actual move on a copy of the real board
                            sim_board = deepcopy(board_matrix)
                            # place moving piece at target (overwriting captured piece if exists)
                            sim_board[er][ec] = drag_piece
                            sr, sc = drag_from
                            sim_board[sr][sc] = None
                            if move_leaves_king_in_check(sim_board, er, ec, er, ec):
                                # note: move_leaves_king_in_check expects sr,sc,er,ec on original board;
                                # simpler test below: simulate using the function that handles move simulation
                                # we'll use helper below to test properly
                                if move_leaves_king_in_check(board_matrix_with_drag(board_matrix, drag_piece, drag_from), drag_from[0], drag_from[1], er, ec):
                                    # illegal because leaves king in check
                                    board_matrix[drag_from[0]][drag_from[1]] = drag_piece
                                    drag_piece = None
                                    drag_from = None
                                else:
                                    # commit
                                    board_matrix[er][ec] = drag_piece
                                    print(f"Moved {drag_piece} from {drag_from} to {(er,ec)}")
                                    # switch turn
                                    turn = "black" if turn == "white" else "white"
                                    print("Turn:", turn)
                                    drag_piece = None
                                    drag_from = None
                            else:
                                # commit
                                board_matrix[er][ec] = drag_piece
                                print(f"Moved {drag_piece} from {drag_from} to {(er,ec)}")
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


# helper: create a temporary board with the dragged piece at its original source (used before generating legal moves)
def board_matrix_with_drag(board_matrix, dragging_piece, drag_from):
    tmp = deepcopy(board_matrix)
    sr, sc = drag_from
    tmp[sr][sc] = dragging_piece
    return tmp
