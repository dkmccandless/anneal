// Package anneal implements simulated annealing optimization of arbitrary data types.
package anneal

import (
	"math"
	"math/rand"
)

var (
	maxi int = 1e6
	ti       = 1.
	tf       = 1e-5
)

// A State can undergo simulated annealing optimization.
type State interface {
	// Energy returns the energy of a State.
	// This is the quantity to be minimized: States with small energy are better than States with large energy.
	Energy() float64

	// Neighbor returns a State in the state space chosen randomly from those adjacent to the current State.
	// The new State must occupy a different location in memory than the current State.
	Neighbor() State
}

// Anneal implements simulated annealing on the input State and returns the best State encountered during the search.
func Anneal(s State) State {
	e := s.Energy()
	sbest, ebest := s, e
	var (
		Tinitial = ebest * ti
		k        = float64(maxi) / math.Log(ti/tf)
	)
	for i := 0; i < maxi; i++ {
		snew := s.Neighbor()
		enew := snew.Energy()
		if enew < e {
			if ebest < enew {
				sbest, ebest = snew, enew
			}
		} else {
			temp := Tinitial * math.Exp(-float64(i)/k)
			if p := math.Exp(-(enew - e) / temp); rand.Float64() > p {
				continue
			}
		}
		s, e = snew, enew
	}
	return sbest
}
