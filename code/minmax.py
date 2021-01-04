
from gurobipy import *

#min-max model, i.e. no recourse action. UB
def min_max(p_bar, p_hat, Gamma, time_limit):

    n = len(p_bar)
    N = [i for i in range(n)]
    
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

    sol = {'status':model.Status, 'objbound':model.ObjBound, 'objval':model.ObjVal, 'mipgap':model.MIPGap, 'runtime':model.Runtime}
#    model.write('minmax.sol')
    return(sol)

