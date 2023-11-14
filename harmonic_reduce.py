"""
Adjust a just intonation scale to lie lower in the harmonic series.

Uses the CP-SAT constraint programming solver. For more information on
CP-SAT see:

    - https://developers.google.com/optimization/cp/cp_solver
    - https://github.com/d-krupke/cpsat-primer
"""
import math
from fractions import Fraction

import pandas as pd
from ortools.sat.python import cp_model


def harmonic_reduce(scale, *, max_changes, max_harmonic):
    """
    Adjust scale to lie lower in the harmonic series.

    At most `max_changes` notes are allowed to change. The solver tries
    to make the largest harmonic in the adjusted scale no greater than
    `max_harmonic`. Among scales below `max_harmonic`, the scale closest
    to the original scale is chosen (where closest is in a particular sense
    which is easy to express for the solver).

    Parameters
    ----------
    scale : list of Fraction
        Just ratios of scale to adjust.
    max_changes : int
        Number of notes allowed to be changed.
    max_harmonic : int
        Target maximum harmonic in the adjusted scale.  It may not be possible
        to reach this while changing at most `max_changes` notes; in this case
        the largest harmonic is made as small as possible.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the original scale as ratios and harmonics, the
        adjusted scale as ratios and harmonics, the original and new cent values,
        and the cent differences.

    Examples
    --------
    Adjust a twelve note just chromatic scale, changing at most three notes
    and attempting to get the maximum harmonic under 200.

    >>> from fractions import Fraction as F
    >>> scale = [
    ...     F(16, 15),
    ...     F(9, 8),
    ...     F(6, 5),
    ...     F(5, 4),
    ...     F(4, 3),
    ...     F(45, 32),
    ...     F(3, 2),
    ...     F(8, 5),
    ...     F(5, 3),
    ...     F(9, 5),
    ...     F(15, 8),
    ...     F(2),
    ... ]
    >>> harmonic_reduce(scale, max_changes=3, max_harmonic=200)
        ratio  harmonic new_ratio  new_harmonic    cents  new_cents  cent_diff
    0   16/15       512     16/15            64   111.73     111.73       0.00
    1     9/8       540     67/60            67   203.91     191.04     -12.87
    2     6/5       576       6/5            72   315.64     315.64       0.00
    3     5/4       600       5/4            75   386.31     386.31       0.00
    4     4/3       640       4/3            80   498.04     498.04       0.00
    5   45/32       675       7/5            84   590.22     582.51      -7.71
    6     3/2       720       3/2            90   701.96     701.96       0.00
    7     8/5       768       8/5            96   813.69     813.69       0.00
    8     5/3       800       5/3           100   884.36     884.36       0.00
    9     9/5       864       9/5           108  1017.60    1017.60       0.00
    10   15/8       900     28/15           112  1088.27    1080.56      -7.71
    11      2       960         2           120  1200.00    1200.00       0.00

    The highest harmonic in the original scale is 960, but by changing three
    notes we can bring it down to 120.
    """
    assert 2 in scale

    model = cp_model.CpModel()

    M = 10**12

    x = {s: model.NewIntVar(1, M, f"x[{s}]") for s in scale}
    u = {s: model.NewIntVar(-M, M, f"u[{s}]") for s in scale}

    for s in scale:
        model.Add(u[s] == 2 * x[s] * s.denominator - x[2] * s.numerator)

    zero_constraints = {s: model.Add(u[s] == 0) for s in scale}

    model.Minimize(x[2])

    solver = cp_model.CpSolver()
    solver.Solve(model)

    base_sol = {s: solver.Value(x[s]) for s in scale}

    b = {s: model.NewBoolVar(f"b[{s}]") for s in scale}
    abs_u = {s: model.NewIntVar(0, M, f"abs_u[{s}]") for s in scale}

    for s in scale:
        zero_constraints[s].OnlyEnforceIf(b[s].Not())
        model.AddAbsEquality(abs_u[s], u[s])

    model.Add(sum(b.values()) <= max_changes)
    violation = model.NewIntVar(0, M, "violation")
    model.Add(x[2] <= max_harmonic + violation)
    model.Minimize(10**6 * violation + sum(abs_u.values()))

    solver.Solve(model)

    if solver.Value(violation) != 0:
        print(
            f"Warning: Could not get under {max_harmonic} harmonics with only {max_changes} changes"
        )

    sol = {s: solver.Value(x[s]) for s in scale}

    df = pd.DataFrame(
        {
            "ratio": scale,
            "harmonic": base_sol.values(),
            "new_ratio": [Fraction(2 * sol[s], sol[2]) for s in scale],
            "new_harmonic": sol.values(),
            "cents": [1200 * math.log2(x) for x in scale],
        }
    )
    df["new_cents"] = [1200 * math.log2(x) for x in df.new_ratio]
    df["cent_diff"] = df.new_cents - df.cents
    for col in df.columns:
        if "cent" in col:
            df[col] = df[col].round(2)

    return df
