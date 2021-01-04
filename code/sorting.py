
from gurobipy import *

def sorting(p_bar, p_hat, l, Gamma):
    
    n = len(p_bar)
    N = [i for i in range(n)]

    #l=0 => sort by nom. values p_bar
    #l=1 => sort by worst-case values p_bar+p_hat
    p = [p_bar[i] + l*p_hat[i] for i in N]

    N_sorted = sorted(N, key=lambda i:p[i])
    print(N_sorted)

    #get N_sorted in terms of x
    x = [[] for i in N]
    for i in N:
        for j in N:
            if N_sorted[j] == i:
                x[i].append(1)
            else:
                x[i].append(0)
    print(x)
    
    #evaluate soln by solving adv(x)        
    objval = adv(p_bar, p_hat, Gamma, x)

    return(objval)
    
def adv(p_bar, p_hat, Gamma, x):
    
    n = len(p_bar)
    N = [i for i in range(n)]

    model = Model("adv")
#    model.setParam("OutputFlag", 0)
    
    #variables
    delta = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="delta", lb=0, ub=1)

    #objective
    model.setObjective(quicksum((p_bar[i] + delta[i]*p_hat[i])*(n+1-j)*x[i][j] for i in N for j in N), GRB.MAXIMIZE)
    
    #constraints
    model.addConstr(quicksum(delta[i] for i in N) <= Gamma)

    model.optimize()

    model.write('sorting_adv.sol')

    return(model.objval)
