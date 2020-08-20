/*
Package anneal implements simulated annealing optimization of arbitrary data types.

Simulated annealing is a probabilistic search heuristic to approximate the global optimum
in a discrete search space by traversing the search space in steps and deciding probabilistically
whether to transition to a new state based on the difference of the "energies" of the current state
and new state as compared with the annealing "temperature", which starts high and decreases
with each successive iteration. The transition decision is thus initially sensitive to large
differences in energy between different areas of the search space, and this sensitivity becomes
finer as the temperature decreases.

The quality of the result depends on the following conditions:

  * Energy() must be defined such that smaller energies are better than larger energies.
  * Neighbor() must randomly select a State from all States that differ from the input State
    by a minimal alteration (i.e., adjacent to the input State in the search space).
    The diameter of the search space must be small: Neighbor() must enable transition
    between any two arbitrary States in a small number of steps.
  * In analogy with the eponymous metallurgical technique, the Schedule parameters must be chosen
    such that the initial temperature is large compared to the difference between the energies
    of typical States, the final temperature is small compared to the difference between
    adjacent states, and the temperature decreases sufficiently slowly that the system is in
    approximate thermodynamic equilibrium at all times.
*/
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
	// Distinct States must not share memory. For example, do not reuse a slice from one State to a neighbor.
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
// The new State is adopted with probability 1 if its energy E' is lower than the original State's energy E,
// and with probability exp(-(E'-E)/T) otherwise, where T = Ti * exp(-i/k) is the annealing temperature of the current iteration i,
// and the scale factor k = Iter / ln(Ti/Tf) is the number of iterations required for the temperature to drop by a factor of e.
func Anneal(s State, sch *Schedule) State {
	e := s.Energy()
	sbest, ebest := s, e
	var (
		T0 = e * sch.Ti
		k  = float64(sch.Iter) / math.Log(sch.Ti/sch.Tf)
	)
	for i := 0; i < sch.Iter; i++ {
		snew := s.Neighbor()
		enew := snew.Energy()
		if enew < e {
			if enew < ebest {
				sbest, ebest = snew, enew
			}
		} else {
			T := T0 * math.Exp(-float64(i)/k)
			if p := math.Exp(-(enew - e) / T); rand.Float64() > p {
				continue
			}
		}
		s, e = snew, enew
	}
	return sbest
}
