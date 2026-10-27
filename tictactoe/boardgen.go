package tictactoe

import (
	"math"
	"slices"
)

func switchPlayer(p Player) Player {
	return 3 - p // X(1) <-> O(2)
}

// FilterBoards removes terminal positions (won or drawn) and forced-move positions (at most one
// empty cell) from boards, returning the remaining entries in a new slice.
func FilterBoards(boards []Board) []Board {
	result := make([]Board, 0, len(boards))
	for _, b := range boards {
		size := int(math.Round(math.Sqrt(float64(len(b)))))
		if _, won := checkWinner(b, size, size); won {
			continue
		}
		if isDraw(b) {
			continue
		}
		empty := 0
		for _, p := range b {
			if p == Empty {
				empty++
			}
		}
		if empty <= 1 {
			continue
		}
		result = append(result, b)
	}
	return result
}

// DeduplicateBoards removes duplicate boards (by content), keeping the last occurrence of each,
// and returns the remaining entries in a new slice preserving relative order.
func DeduplicateBoards(boards []Board) []Board {
	lastIdx := make(map[BoardKey]int, len(boards))
	for i, b := range boards {
		lastIdx[b.ToKey()] = i
	}
	result := make([]Board, 0, len(lastIdx))
	for i, b := range boards {
		if lastIdx[b.ToKey()] == i {
			result = append(result, b)
		}
	}
	return result
}

// SortBoards sorts boards in place in ascending lexicographic order by content.
func SortBoards(boards []Board) {
	slices.SortFunc(boards, func(a, b Board) int {
		return slices.Compare(a, b)
	})
}

// Enumerate3x3Boards enumerates every reachable board position in a standard 3x3 game and returns
// them in canonical form: the side to move is always encoded as 1, the opponent as 2. Terminal
// positions (won or drawn) are included; the empty board is the first entry.
func Enumerate3x3Boards() []Board {
	visited := map[[9]Player]struct{}{}
	var boards []Board

	var dfs func(state [9]Player, player Player)
	dfs = func(state [9]Player, player Player) {
		if _, seen := visited[state]; seen {
			return
		}
		visited[state] = struct{}{}
		b := make(Board, 9)
		if player == X {
			copy(b, state[:])
		} else {
			// Relabel to canonical: O is the mover so O's stones become 1, X's become 2.
			for i, p := range state {
				if p != Empty {
					b[i] = switchPlayer(p)
				}
			}
		}
		boards = append(boards, b)
		if _, won := checkWinner(Board(state[:]), 3, 3); won || isDraw(Board(state[:])) {
			return
		}
		for k := range 9 {
			if state[k] == Empty {
				state[k] = player
				dfs(state, switchPlayer(player))
				state[k] = Empty
			}
		}
	}
	dfs([9]Player{}, X)
	return boards
}
