from gurobipy import *
import random

def model2(p_bar, p_hat, Gamma, Delta, time_limit):

    n = len(p_bar)
    N = [i for i in range(n)]

    model = Model("model2")
#    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)

    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")
    y = model.addVars([(i,j) for i in N for j in N], vtype=GRB.CONTINUOUS, name="y", lb=0, ub=1)
    w = model.addVars([(i,j,l) for i in N for j in N for l in N], vtype=GRB.CONTINUOUS, name="w", lb=0)
    pi = model.addVar(vtype=GRB.CONTINUOUS, name = "pi", lb=0)
    rho = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="rho", lb=0)

    #objective
    model.setObjective(quicksum((n+1)*p_bar[i]*y[i,j] - quicksum(w[i,j,l]*p_bar[i]*l for l in N) for i in N for j in N) + Gamma*pi + quicksum(rho[i] for i in N))

    #constraints
    model.addConstrs(quicksum(y[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(y[i,j] for j in N) == 1 for i in N)
    model.addConstr(quicksum(y[i,i] for i in N) >= n - 2*Delta)
    model.addConstrs(y[i,j] == y[j,i] for i in N for j in N)
    model.addConstrs(pi + rho[i] >= quicksum((n+1)*p_hat[i]*y[i,j] - quicksum(w[i,j,l]*p_hat[i]*l for l in N) for j in N) for i in N)
    model.addConstrs(w[i,j,l] <= x[j,l] for i in N for j in N for l in N)
    model.addConstrs(w[i,j,l] <= y[i,j] for i in N for j in N for l in N)
    model.addConstrs(w[i,j,l] >= y[i,j] - (1-x[j,l]) for i in N for j in N for l in N)
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)

    model.optimize()
    
    sol = {'status':model.Status, 'objbound':model.ObjBound, 'objval':model.ObjVal, 'mipgap':model.MIPGap, 'runtime':model.Runtime}
    return(sol)


def model2_ws(p_bar, p_hat, Gamma, Delta, time_limit):

    n = len(p_bar)
    N = [i for i in range(n)]
    
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

    #solving model2 with warmstart
    model = Model("model2_ws")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)

    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")
    y = model.addVars([(i,j) for i in N for j in N], vtype=GRB.CONTINUOUS, name="y", lb=0, ub=1)
    w = model.addVars([(i,j,l) for i in N for j in N for l in N], vtype=GRB.CONTINUOUS, name="w", lb=0)
    pi = model.addVar(vtype=GRB.CONTINUOUS, name = "pi", lb=0)
    rho = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="rho", lb=0)
    
    #setting warmstart
    pi.start = ws_pi
    for i in N:
        rho[i].start = ws_rho[i]
        for j in N:
            x[i,j].start = ws_x[i][j]

    #objective
    model.setObjective(quicksum((n+1)*p_bar[i]*y[i,j] - quicksum(w[i,j,l]*p_bar[i]*l for l in N) for i in N for j in N) + Gamma*pi + quicksum(rho[i] for i in N))

    #constraints
    model.addConstrs(quicksum(y[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(y[i,j] for j in N) == 1 for i in N)
    model.addConstr(quicksum(y[i,i] for i in N) >= n - 2*Delta)
    model.addConstrs(y[i,j] == y[j,i] for i in N for j in N)
    model.addConstrs(pi + rho[i] >= quicksum((n+1)*p_hat[i]*y[i,j] - quicksum(w[i,j,l]*p_hat[i]*l for l in N) for j in N) for i in N)
    model.addConstrs(w[i,j,l] <= x[j,l] for i in N for j in N for l in N)
    model.addConstrs(w[i,j,l] <= y[i,j] for i in N for j in N for l in N)
    model.addConstrs(w[i,j,l] >= y[i,j] - (1-x[j,l]) for i in N for j in N for l in N)
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)

    model.optimize()

    sol = {'status':model.Status, 'objbound':model.ObjBound, 'objval':model.ObjVal, 'mipgap':model.MIPGap, 'runtime':model.Runtime}
    return(sol)
