from copy import deepcopy

# ---------- Utility helpers ----------

def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def get_color(piece_key):
    if not piece_key:
        return None
    return piece_key.split("_")[0]  # "white" or "black"

def get_type(piece_key):
    if not piece_key:
        return None
    return piece_key.split("_")[1]  # "pawn","rook",etc.

# ---------- Attack generation (for check detection) ----------

def attacked_squares(board_matrix, by_color):
    attacks = set()
    for r in range(8):
        for c in range(8):
            key = board_matrix[r][c]
            if not key or get_color(key) != by_color:
                continue
            ptype = get_type(key)
            if ptype == "pawn":
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

# ---------- Legal move generation ----------

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
        # no castling implemented yet

    return moves

# ---------- Validation: does move leave king in check? ----------

def move_leaves_king_in_check(board_matrix, sr, sc, er, ec):
    sim = deepcopy(board_matrix)
    moving_piece = sim[sr][sc]
    sim[er][ec] = moving_piece
    sim[sr][sc] = None

    color = get_color(moving_piece)
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
        return True  # invalid, king missing

    opponent = "white" if color == "black" else "black"
    opp_attacks = attacked_squares(sim, opponent)
    return king_pos in opp_attacks

# ---------- Helpers for check / checkmate detection ----------

def find_king(board_matrix, color):
    """Return (r,c) of the king of given color, or None if not found."""
    for r in range(8):
        for c in range(8):
            k = board_matrix[r][c]
            if k and get_type(k) == "king" and get_color(k) == color:
                return (r, c)
    return None

def is_in_check(board_matrix, color):
    """
    Return True if the king of `color` is currently attacked.
    Uses attacked_squares by the opponent.
    """
    king_pos = find_king(board_matrix, color)
    if king_pos is None:
        # no king: treat as in check (invalid board)
        return True
    opponent = "white" if color == "black" else "black"
    opp_attacks = attacked_squares(board_matrix, opponent)
    return king_pos in opp_attacks

# ---------- New helpers: simulate move and enumerate all legal moves for color ----------

def simulate_move(board_matrix, sr, sc, er, ec):
    """
    Return a deep-copied board with the move applied.
    This does a simple move/capture (no en-passant, no promotions handling).
    """
    sim = deepcopy(board_matrix)
    moving_piece = sim[sr][sc]
    sim[er][ec] = moving_piece
    sim[sr][sc] = None
    return sim

def all_legal_moves_for_color(board_matrix, color):
    """
    Generator that yields tuples (sr, sc, er, ec, piece_key)
    for every legal move according to piece move generation (not considering king-in-check filter).
    Caller should simulate and test resulting legality (i.e. whether king remains in check).
    """
    for r in range(8):
        for c in range(8):
            key = board_matrix[r][c]
            if not key or get_color(key) != color:
                continue
            moves = legal_moves_for(key, r, c, board_matrix)
            for (er, ec) in moves:
                yield (r, c, er, ec, key)

def is_checkmate(board_matrix, color):
    """
    Checkmate: king is in check and the side has NO legal moves that avoid check.
    We iterate all pieces of `color`, generate legal moves, and simulate them.
    """
    # If king isn't in check, not checkmate
    if not is_in_check(board_matrix, color):
        return False

    # For every legal move for this color
    for sr, sc, er, ec, piece_key in all_legal_moves_for_color(board_matrix, color):
        # simulate the move
        sim = simulate_move(board_matrix, sr, sc, er, ec)

        # find king position after move (the king may have moved)
        kpos = find_king(sim, color)
        if kpos is None:
            # no king after the move => treat as invalid (skip)
            continue

        # if after this move the king is NOT in check, then it's not checkmate
        if not is_in_check(sim, color):
            return False

    # no legal move avoids check -> checkmate
    return True
