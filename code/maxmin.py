
from gurobipy import *

#max-min. best schedule for the worst-case scenario
def max_min(p_bar, p_hat, Gamma, time_limit):
    
    n = len(p_bar)
    N = [i for i in range(n)]

    #get worst-case scenario
    model = Model("max_min")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)

    #variables
    alpha = model.addVars([j for j in N], vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="alpha")
    beta = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="beta")
    delta = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, lb=0, ub=1, name="delta")

    #objective
    model.setObjective(quicksum(alpha[j] for j in N) + quicksum(beta[i] for i in N), GRB.MAXIMIZE)

    #constraints
    model.addConstrs(alpha[j] + beta[i] <= (p_bar[i] + delta[i]*p_hat[i])*(n+1-j) for i in N for j in N)
    model.addConstr(quicksum(delta[i] for i in N) <= Gamma)

    model.optimize()
    
    model.write('wcs.sol')

    #worst-case scenario
    p = []
    for i in N:
        p.append(p_bar[i] + model.getVarByName("delta[{}]".format(i)).X*p_hat[i])

    #getting best solution for worst-case scenario
    model = Model("scenario_soln")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)

    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")

    #objective
    model.setObjective(quicksum(p[i]*(n+1-j)*x[i,j] for i in N for j in N), GRB.MINIMIZE)
    
    #constraints
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)

    model.optimize()
    
    x = [[] for i in N]
    for i in N:
        for j in N:
            x[i].append(model.getVarByName("x[{},{}]".format(i,j)).X)

    #evaluate solution x with adv(x)
    objval = adv(p_bar, p_hat, Gamma, x)

    return(objval)

def adv(p_bar, p_hat, Gamma, x):
    
    n = len(p_bar)
    N = [i for i in range(n)]

    model = Model("adv")
    model.setParam("OutputFlag", 0)
    
    #variables
    delta = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="delta", lb=0, ub=1)

    #objective
    model.setObjective(quicksum((p_bar[i] + delta[i]*p_hat[i])*(n+1-j)*x[i][j] for i in N for j in N), GRB.MAXIMIZE)
    
    #constraints
    model.addConstr(quicksum(delta[i] for i in N) <= Gamma)

    model.optimize()

    return(model.objval)

