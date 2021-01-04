from gurobipy import *

def model1(p_bar, p_hat, Gamma, Delta, K, time_limit):

    n = len(p_bar)
    N = [i for i in range(n)]
    K = [k for k in range(K)]

    model = Model("model1")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)
    
    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")
    z = model.addVars([(i,j,k) for i in N for j in N for k in K], vtype=GRB.BINARY, name="z")
    w = model.addVars([(i,j,l,k) for i in N for j in N for l in N for k in K], vtype=GRB.BINARY, name="w")
    h = model.addVars([(i,j,l,k) for i in N for j in N for l in N for k in K], vtype=GRB.CONTINUOUS, lb=0, name="h")
    mu = model.addVars([k for k in K], vtype=GRB.CONTINUOUS, name="mu", lb=0)
    pi = model.addVar(vtype=GRB.CONTINUOUS, name = "pi", lb=0)
    rho = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="rho", lb=0)

    #objective
    model.setObjective(quicksum(quicksum(p_bar[i]*(n-j+1)*quicksum(h[i,j,l,k] for l in N) for i in N for j in N) for k in K) + Gamma*pi + quicksum(rho[i] for i in N), GRB.MINIMIZE)

    #constraints
    model.addConstr(quicksum(mu[k] for k in K) == 1)
    model.addConstrs(pi + rho[i] >= quicksum(quicksum(p_hat[i]*(n-j+1)*quicksum(h[i,j,l,k] for l in N) for j in N) for k in K) for i in N)
    model.addConstrs(quicksum(z[i,j,k] for i in N) == 1 for j in N for k in K)
    model.addConstrs(quicksum(z[i,j,k] for j in N) == 1 for i in N for k in K)
    model.addConstrs(z[i,j,k] == z[j,i,k] for i in N for j in N for k in K)
    model.addConstrs(quicksum(z[i,i,k] for i in N) >= n - 2*Delta for k in K)
    model.addConstrs(w[i,j,l,k] <= z[i,l,k] for i in N for j in N for l in N for k in K)
    model.addConstrs(w[i,j,l,k] <= x[l,j] for i in N for j in N for l in N for k in K)
    model.addConstrs(w[i,j,l,k] >= x[l,j] + z[i,l,k] - 1 for i in N for j in N for l in N for k in K)
    model.addConstrs(h[i,j,l,k] <= w[i,j,l,k] for i in N for j in N for l in N for k in K)
    model.addConstrs(h[i,j,l,k] <= mu[k] for i in N for j in N for l in N for k in K)
    model.addConstrs(h[i,j,l,k] >= mu[k] - (1-w[i,j,l,k]) for i in N for j in N for l in N for k in K)
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)

    model.optimize()

    sol = {'status':model.Status, 'objbound':model.ObjBound, 'objval':model.ObjVal, 'mipgap':model.MIPGap, 'runtime':model.Runtime}
#    model.write("model1.sol")
    return(sol)

def model1_ws(p_bar, p_hat, Gamma, Delta, K, time_limit):

    n = len(p_bar)
    N = [i for i in range(n)]
    K = [k for k in range(K)]

    #solving min_max model to get warmstart solution
    model = Model("min_max")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)

    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")
    pi = model.addVar(vtype=GRB.CONTINUOUS, name = "pi", lb=0)
    rho = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="rho", lb=0)

    #objective
    model.setObjective(quicksum(p_bar[i]*(n+1-j)*x[i,j] for i in N for j in N) + Gamma*pi + quicksum(rho[i] for i in N))
    
    #constraints
    model.addConstrs(pi + rho[i] >= quicksum(p_hat[i]*(n+1-j)*x[i,j] for j in N) for i in N)
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)
    
    model.optimize()
    
    ws_x = [[model.getVarByName("x[{},{}]".format(i,j)).X for j in N] for i in N]
    ws_pi = model.getVarByName("pi").X
    ws_rho = [model.getVarByName("rho[{}]".format(i)).X for i in N]

    #solving model1 with warmstart
    model = Model("model1_ws")
#    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)
    
    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")
    z = model.addVars([(i,j,k) for i in N for j in N for k in K], vtype=GRB.BINARY, name="z")
    w = model.addVars([(i,j,l,k) for i in N for j in N for l in N for k in K], vtype=GRB.BINARY, name="w")
    h = model.addVars([(i,j,l,k) for i in N for j in N for l in N for k in K], vtype=GRB.CONTINUOUS, lb=0, name="h")
    mu = model.addVars([k for k in K], vtype=GRB.CONTINUOUS, name="mu", lb=0)
    pi = model.addVar(vtype=GRB.CONTINUOUS, name = "pi", lb=0)
    rho = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="rho", lb=0)

    #setting warmstart
    pi.start = ws_pi
    for i in N:
        rho[i].start = ws_rho[i]
        for j in N:
            x[i,j].start = ws_x[i][j]

    #objective
    model.setObjective(quicksum(quicksum(p_bar[i]*(n-j+1)*quicksum(h[i,j,l,k] for l in N) for i in N for j in N) for k in K) + Gamma*pi + quicksum(rho[i] for i in N), GRB.MINIMIZE)

    #constraints
    model.addConstr(quicksum(mu[k] for k in K) == 1)
    model.addConstrs(pi + rho[i] >= quicksum(quicksum(p_hat[i]*(n-j+1)*quicksum(h[i,j,l,k] for l in N) for j in N) for k in K) for i in N)
    model.addConstrs(quicksum(z[i,j,k] for i in N) == 1 for j in N for k in K)
    model.addConstrs(quicksum(z[i,j,k] for j in N) == 1 for i in N for k in K)
    model.addConstrs(z[i,j,k] == z[j,i,k] for i in N for j in N for k in K)
    model.addConstrs(quicksum(z[i,i,k] for i in N) >= n - 2*Delta for k in K)
    model.addConstrs(w[i,j,l,k] <= z[i,l,k] for i in N for j in N for l in N for k in K)
    model.addConstrs(w[i,j,l,k] <= x[l,j] for i in N for j in N for l in N for k in K)
    model.addConstrs(w[i,j,l,k] >= x[l,j] + z[i,l,k] - 1 for i in N for j in N for l in N for k in K)
    model.addConstrs(h[i,j,l,k] <= w[i,j,l,k] for i in N for j in N for l in N for k in K)
    model.addConstrs(h[i,j,l,k] <= mu[k] for i in N for j in N for l in N for k in K)
    model.addConstrs(h[i,j,l,k] >= mu[k] - (1-w[i,j,l,k]) for i in N for j in N for l in N for k in K)
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)

    model.optimize()

    sol = {'status':model.Status, 'objbound':model.ObjBound, 'objval':model.ObjVal, 'mipgap':model.MIPGap, 'runtime':model.Runtime}
#    model.write("model1.sol")
    return(sol)

