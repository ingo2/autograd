package tictactoe

// D4Perms returns the 7 non-identity permutations of the dihedral group D4 for an nxn board.
// Each permutation p satisfies newBoard[i] = origBoard[p[i]], where cell (r, c) maps to r*n+c.
func D4Perms(n int) [][]int {
	mkPerm := func(f func(r, c int) (int, int)) []int {
		p := make([]int, n*n)
		for r := range n {
			for c := range n {
				nr, nc := f(r, c)
				p[r*n+c] = nr*n + nc
			}
		}
		return p
	}
	return [][]int{
		mkPerm(func(r, c int) (int, int) { return c, n - 1 - r }),         // r090
		mkPerm(func(r, c int) (int, int) { return n - 1 - r, n - 1 - c }), // r180
		mkPerm(func(r, c int) (int, int) { return n - 1 - c, r }),         // r270
		mkPerm(func(r, c int) (int, int) { return r, n - 1 - c }),         // fh
		mkPerm(func(r, c int) (int, int) { return n - 1 - r, c }),         // fv
		mkPerm(func(r, c int) (int, int) { return c, r }),                 // d+
		mkPerm(func(r, c int) (int, int) { return n - 1 - c, n - 1 - r }), // d-
	}
}

// boardSideLen returns the side length of a square board.
func boardSideLen(board Board) int {
	n := 1
	for n*n < len(board) {
		n++
	}
	return n
}

// d4CanonicalPerm returns board's D4-canonical representative together with the permutation that
// produced it: rep[cell] = board[perm[cell]] for every cell. perm is the identity permutation when
// board is already its own canonical representative.
func d4CanonicalPerm(board Board) (rep Board, perm []int) {
	n := boardSideLen(board)
	rep = board
	perm = make([]int, n*n)
	for i := range perm {
		perm[i] = i
	}
	for _, p := range D4Perms(n) {
		v := make(Board, n*n)
		for cell := range n * n {
			v[cell] = board[p[cell]]
		}
		if v.LessThan(rep) {
			rep = v
			perm = p
		}
	}
	return rep, perm
}

// CanonicalKey returns the BoardKey of board's D4-canonical representative, so that two boards
// related by a rotation or reflection always produce the same key.
func CanonicalKey(board Board) BoardKey {
	rep, _ := d4CanonicalPerm(board)
	return rep.ToKey()
}

// D4Augment expands boards with the 8 symmetries of the dihedral group D4 (identity, 3 rotations,
// and 4 reflections), transforming both Board and Scores so that each result remains a valid
// AnnotatedBoard for the rotated/reflected position. Boards that are invariant under one or more of
// these symmetries (e.g. an empty board, or a single piece on the center cell) only contribute
// their distinct variants, so the result can be smaller than 8*len(boards).
func D4Augment(boards []AnnotatedBoard) []AnnotatedBoard {
	result := make([]AnnotatedBoard, 0, 8*len(boards))
	if len(boards) == 0 {
		return result
	}

	n := boardSideLen(boards[0].Board)
	perms := D4Perms(n)

	// invSymmetries[i] is the inverse permutation of symmetries[i]. Board cells are transformed via
	// sym (newBoard.Board[cell] = origBoard.Board[sym[cell]], i.e. the piece at old position
	// sym[cell] moves to new position cell). Scores.Pos values move in the opposite direction
	// (old position -> new position), so they need the inverse permutation.
	invPerms := make([][]int, len(perms))
	for i, sym := range perms {
		inv := make([]int, n*n)
		for cell, p := range sym {
			inv[p] = cell
		}
		invPerms[i] = inv
	}

	// seen tracks the canonical Board of every board already added to result, so that boards which
	// are invariant under one or more D4 symmetries (e.g. an empty board, or a single piece on the
	// center cell) don't end up as repeated copies of the same training example.
	seen := make(map[BoardKey]struct{}, 8*len(boards))
	addIfNew := func(b AnnotatedBoard) {
		key := b.Board.ToKey()
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		result = append(result, b)
	}

	for _, origBoard := range boards {
		addIfNew(origBoard)

		for i, sym := range perms {
			newBoard := AnnotatedBoard{
				Player: origBoard.Player, // Player is not affected.
			}
			newBoard.Scores = make([]ScoredMove, len(origBoard.Scores))
			newBoard.Board = make([]Player, n*n)
			// Transform cells.
			for cell := range n * n {
				newBoard.Board[cell] = origBoard.Board[sym[cell]]
			}
			// Transform result scores.
			for si := range origBoard.Scores {
				newBoard.Scores[si] = ScoredMove{
					// Different position, same score.
					Pos:   invPerms[i][origBoard.Scores[si].Pos],
					Score: origBoard.Scores[si].Score,
				}
			}
			addIfNew(newBoard)
		}
	}

	return result
}

// RemoveOverlap returns the entries of boards whose Board value does not appear in any of exclude.
// It is meant to be applied after D4Augment: since the symmetric variants it generates can coincide
// with boards held out for testing or validation, this removes those duplicates from the augmented
// training set.
func RemoveOverlap(boards []AnnotatedBoard, exclude ...[]AnnotatedBoard) []AnnotatedBoard {
	excluded := make(map[BoardKey]struct{})
	for _, set := range exclude {
		for _, b := range set {
			excluded[b.Board.ToKey()] = struct{}{}
		}
	}

	result := make([]AnnotatedBoard, 0, len(boards))
	for _, b := range boards {
		if _, ok := excluded[b.Board.ToKey()]; ok {
			continue
		}
		result = append(result, b)
	}
	return result
}
