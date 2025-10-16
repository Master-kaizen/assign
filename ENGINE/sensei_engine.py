
# Handles all move generation and validation logic

def is_in_bounds(row, col):
    return 0 <= row < 8 and 0 <= col < 8


def piece_color(piece):
    if not piece:
        return None
    return "white" if piece.startswith("white") else "black"


def opposite(color):
    return "black" if color == "white" else "white"


def legal_moves_for(board, row, col, turn):
    """Return list of valid move coordinates (r, c) for the piece at board[row][col]."""
    piece = board[row][col]
    if not piece or piece_color(piece) != turn:
        return []

    color = piece_color(piece)
    name = piece.split("_")[1]
    moves = []

    directions = {
        "rook": [(1, 0), (-1, 0), (0, 1), (0, -1)],
        "bishop": [(1, 1), (-1, -1), (1, -1), (-1, 1)],
        "queen": [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, -1), (1, -1), (-1, 1)],
        "knight": [(2, 1), (1, 2), (-1, 2), (-2, 1),
                   (-2, -1), (-1, -2), (1, -2), (2, -1)],
        "king": [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (-1, -1), (1, -1), (-1, 1)]
    }

    if name == "pawn":
        direction = -1 if color == "white" else 1
        start_row = 6 if color == "white" else 1

        # Forward move
        if is_in_bounds(row + direction, col) and not board[row + direction][col]:
            moves.append((row + direction, col))

            # Double move from starting rank
            if row == start_row and not board[row + 2 * direction][col]:
                moves.append((row + 2 * direction, col))

        # Captures
        for dc in (-1, 1):
            nr, nc = row + direction, col + dc
            if is_in_bounds(nr, nc) and board[nr][nc]:
                if piece_color(board[nr][nc]) != color:
                    moves.append((nr, nc))

    elif name in ["rook", "bishop", "queen"]:
        for dr, dc in directions[name]:
            nr, nc = row + dr, col + dc
            while is_in_bounds(nr, nc):
                target = board[nr][nc]
                if not target:
                    moves.append((nr, nc))
                elif piece_color(target) != color:
                    moves.append((nr, nc))
                    break
                else:
                    break
                nr += dr
                nc += dc

    elif name == "knight":
        for dr, dc in directions["knight"]:
            nr, nc = row + dr, col + dc
            if is_in_bounds(nr, nc):
                target = board[nr][nc]
                if not target or piece_color(target) != color:
                    moves.append((nr, nc))

    elif name == "king":
        for dr, dc in directions["king"]:
            nr, nc = row + dr, col + dc
            if is_in_bounds(nr, nc):
                target = board[nr][nc]
                if not target or piece_color(target) != color:
                    moves.append((nr, nc))

    return moves


def move_leaves_king_in_check(board, from_pos, to_pos, color):
    """For now placeholder — will handle check logic later."""
    # Copy the board
    import copy
    temp_board = copy.deepcopy(board)

    fr, fc = from_pos
    tr, tc = to_pos
    piece = temp_board[fr][fc]
    temp_board[fr][fc] = None
    temp_board[tr][tc] = piece

    # TODO: Implement actual check detection later
    return False
