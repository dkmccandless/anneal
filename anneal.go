// Package anneal implements simulated annealing optimization of arbitrary data types.
package anneal

import (
	"math"
	"math/rand"
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

// A Schedule controls the annealing process.
type Schedule struct {
	Iter int     // number of iterations
	Ti   float64 // initial temperature, as a multiple of the input State's energy
	Tf   float64 // final temperature, as a multiple of the input State's energy
}

// NewSchedule returns a pointer to a Schedule populated with default values.
func NewSchedule() *Schedule {
	return &Schedule{
		Iter: 1e6,
		Ti:   1,
		Tf:   1e-5,
	}
}

// Anneal implements simulated annealing on the input State and returns the best State encountered during the search.
// Once per iteration, it calls s.Neighbor() and then calls Energy() on the neighboring State.
// The new State is adopted with probability 1 if its Energy e' is lower than the original State's energy e,
// and with probability exp(-(e'-e)/T) otherwise, where T = Ti * exp(-i/k) is the annealing temperature of the current iteration i,
// and the scale factor k = Iter / ln(Ti/Tf) is the number of iterations required for the temperature to drop by a factor of e.
func Anneal(s State, sch *Schedule) State {
	e := s.Energy()
	sbest, ebest := s, e
	var (
		Tinitial = e * sch.Ti
		k        = float64(sch.Iter) / math.Log(sch.Ti/sch.Tf)
	)
	for i := 0; i < sch.Iter; i++ {
		snew := s.Neighbor()
		enew := snew.Energy()
		if enew < e {
			if ebest < enew {
				sbest, ebest = snew, enew
			}
		} else {
			T := Tinitial * math.Exp(-float64(i)/k)
			if p := math.Exp(-(enew - e) / T); rand.Float64() > p {
				continue
			}
		}
		s, e = snew, enew
	}
	return sbest
}
