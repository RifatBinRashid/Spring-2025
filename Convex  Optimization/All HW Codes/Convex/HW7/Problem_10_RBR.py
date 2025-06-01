import cvxpy as cvx

# Define variables
x1 = cvx.Variable()
x2 = cvx.Variable()

# Define constraints (inequalities)
constraints = [
    2 * x1 + x2 >= 1,
    x1 + 3 * x2 >= 1,
    x1 >= 0,
    x2 >= 0
]

# Define all five objective functions
objectives = {
    "(a) minimize x1 + x2": cvx.Minimize(x1 + x2),
    "(b) minimize -x1 - x2": cvx.Minimize(-x1 - x2),
    "(c) minimize x1": cvx.Minimize(x1),
    "(d) minimize max{x1, x2}": cvx.Minimize(cvx.maximum(x1, x2)),
    "(e) minimize x1^2 + 9*x2^2": cvx.Minimize(cvx.square(x1) + 9 * cvx.square(x2))
}

# Solve each problem and print results
for label, obj in objectives.items():
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    
    print(f"{label}")
    print(f"  Optimal value: {prob.value}")
    
    # Check for solution
    if x1.value is not None and x2.value is not None:
        print(f"  Optimal x1: {x1.value:.4f}")
        print(f"  Optimal x2: {x2.value:.4f}")
    else:
        print("  Problem is unbounded or infeasible.")
    
    print("-" * 40)
