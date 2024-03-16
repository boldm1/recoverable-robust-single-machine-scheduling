from ortools.linear_solver.pywraplp import Solver


def general_model(p_bar, p_hat, Gamma, Delta, K, additional_z_constraints=True):

    solver = Solver.CreateSolver("SCIP")
    if not solver:
        return

    n = len(p_bar)
    N = [i for i in range(n)]
    K = [k for k in range(K)]

    # Variables
    x = {}
    for i in N:
        for j in N:
            x[i, j] = solver.IntVar(0, 1, "x")

    z = {}
    for i in N:
        for j in N:
            for k in K:
                z[i, j, k] = solver.IntVar(0, 1, "z")

    w = {}
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    w[i, j, l, k] = solver.IntVar(0, 1, "w")

    h = {}
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    h[i, j, l, k] = solver.NumVar(0, solver.infinity(), "h")

    mu = {}
    for k in K:
        mu[k] = solver.NumVar(0, solver.infinity(), "mu")

    pi = solver.NumVar(0, solver.infinity(), "pi")

    rho = {}
    for i in N:
        rho[i] = solver.NumVar(0, solver.infinity(), "rho")

    # Objective function
    objective_terms = []
    for k in K:
        for i in N:
            for j in N:
                for l in N:
                    objective_terms.append(p_bar[i] * (n - j + 1) * h[i, j, l, k])
    for i in N:
        objective_terms.append(rho[i])
    objective_terms.append(Gamma * pi)
    solver.Minimize(solver.Sum(objective_terms))

    # Constraints
    solver.Add(solver.Sum(mu[k] for k in K) == 1)
    for i in N:
        solver.Add(
            pi + rho[i]
            >= solver.Sum(
                p_hat[i] * (n - j + 1) * solver.Sum(h[i, j, l, k] for l in N)
                for j in N
                for k in K
            )
        )
    for j in N:
        for k in K:
            solver.Add(solver.Sum(z[i, j, k] for i in N) == 1)
    if additional_z_constraints:
        for i in N:
            for k in K:
                solver.Add(solver.Sum(z[i, j, k] for j in N) == 1)
    for i in N:
        for j in N:
            for k in K:
                solver.Add(z[i, j, k] == z[j, i, k])
    for k in K:
        solver.Add(solver.Sum(z[i, i, k] for i in N) >= n - 2 * Delta)
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    solver.Add(w[i, j, l, k] <= z[i, l, k])
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    solver.Add(w[i, j, l, k] <= x[l, j])
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    solver.Add(w[i, j, l, k] >= x[l, j] + z[i, l, k] - 1)
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    solver.Add(h[i, j, l, k] <= w[i, j, l, k])
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    solver.Add(h[i, j, l, k] <= mu[k])
    for i in N:
        for j in N:
            for l in N:
                for k in K:
                    solver.Add(h[i, j, l, k] >= mu[k] - (1 - w[i, j, l, k]))
    for j in N:
        solver.Add(solver.Sum(x[i, j] for i in N) == 1)
    for i in N:
        solver.Add(solver.Sum(x[i, j] for j in N) == 1)

    solver.EnableOutput()
    solver.Solve()
    return solver


if __name__ == "__main__":

    Delta = 0
    K = 2
    p_hats = [
        [18, 98, 33, 64, 58, 84, 27, 63, 50, 78],
        [99, 90, 35, 30, 14, 4, 4, 70, 49, 28],
        [93, 68, 98, 64, 30, 30, 29, 59, 3, 72],
        [33, 22, 35, 92, 59, 42, 61, 4, 50, 54],
        [13, 81, 38, 96, 93, 65, 65, 25, 37, 64],
        [87, 49, 45, 69, 99, 31, 93, 11, 22, 69],
        [51, 5, 32, 52, 86, 47, 90, 87, 48, 57],
        [66, 100, 67, 48, 94, 61, 40, 79, 75, 83],
        [22, 30, 99, 70, 30, 66, 74, 59, 85, 78],
        [1, 95, 17, 100, 27, 8, 47, 71, 65, 63],
    ]
    p_bars = [
        [73, 9, 16, 98, 61, 49, 13, 4, 56, 98],
        [1, 58, 93, 76, 41, 3, 84, 2, 88, 55],
        [4, 29, 57, 71, 45, 87, 98, 38, 54, 83],
        [68, 85, 83, 38, 90, 64, 15, 40, 44, 25],
        [24, 93, 16, 43, 92, 55, 86, 39, 76, 65],
        [13, 71, 88, 63, 69, 9, 6, 18, 22, 28],
        [76, 62, 96, 54, 23, 71, 100, 95, 12, 85],
        [14, 21, 51, 63, 4, 6, 91, 76, 51, 22],
        [65, 2, 26, 71, 52, 45, 46, 35, 71, 94],
        [50, 66, 67, 72, 55, 62, 73, 26, 53, 46],
    ]
    with open("with_additional_constraints.txt", "w+") as f:
        f.write("n, Gamma, Delta, obj_val, best_bound, solve_time\n")
    for Gamma in [3, 5, 7]:
        for i, v in enumerate(p_hats):
            sol = general_model(
                p_bars[i], p_hats[i], Gamma, Delta, K, additional_z_constraints=True
            )
            with open("with_additional_constraints.txt", "a") as f:
                f.write(
                    f"{len(v)}, 0, {Gamma}, {sol.Objective().Value()}, {sol.Objective().BestBound()}, {sol.WallTime()/1000}\n"
                )

    with open("without_additional_constraints.txt", "w+") as f:
        f.write("n, Gamma, Delta, obj_val, best_bound, solve_time\n")
    for Gamma in [3, 5, 7]:
        for i, v in enumerate(p_hats):
            sol = general_model(
                p_bars[i], p_hats[i], Gamma, Delta, K, additional_z_constraints=False
            )
            with open("without_additional_constraints.txt", "a+") as f:
                f.write(
                    f"{len(v)}, 0, {Gamma}, {sol.Objective().Value()}, {sol.Objective().BestBound()}, {sol.WallTime()/1000}\n"
                )
