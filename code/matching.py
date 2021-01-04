
from gurobipy import *

def model3(p_bar, p_hat, Gamma, Delta, time_limit):
    
    n = len(p_bar)
    N = [i for i in range(n)]

    model = Model("model3")
#    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)
    
    E = [(i,j) for i in N for j in N if j>i]
    
    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")
    y = model.addVars([e for e in E], vtype=GRB.CONTINUOUS, name="y", lb=0)
    pi = model.addVar(vtype=GRB.CONTINUOUS, name = "pi", lb=0)
    rho = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="rho", lb=0)
    u = model.addVars([(i,j,l) for i in N for j in N for l in N], vtype=GRB.CONTINUOUS, name="u", lb=0)
    v = model.addVars([(i,j,l) for i in N for j in N for l in N], vtype=GRB.CONTINUOUS, name="v", lb=0)

    #objective
    model.setObjective(quicksum(p_bar[i]*(n+1-quicksum(x[i,l]*l for l in N)) for i in N) + quicksum(p_bar[e[1]]*quicksum(v[e[0],e[1],l]*l for l in N) - p_bar[e[1]]*quicksum(u[e[0],e[1],l]*l for l in N) - p_bar[e[0]]*quicksum(v[e[0],e[1],l]*l for l in N) + p_bar[e[0]]*quicksum(u[e[0],e[1],l]*l for l in N) for e in E) + Gamma*pi + quicksum(rho[i] for i in N), GRB.MINIMIZE)
    
    #constraints
    model.addConstrs(rho[i] + pi + quicksum(p_hat[e[0]]*(quicksum(v[e[0],e[1],l]*l for l in N)-quicksum(u[e[0],e[1],l]*l for l in N)) for e in E if e[0] == i) - quicksum(p_hat[e[1]]*(quicksum(v[e[0],e[1],l]*l for l in N)-quicksum(u[e[0],e[1],l]*l for l in N)) for e in E if e[1] == i) >= p_hat[i]*(n+1-quicksum(x[i,l]*l for l in N)) for i in N)
    model.addConstrs(quicksum(y[e] for e in E if (e[0] == i) or (e[1] == i)) <= 1 for i in N)
    model.addConstr(quicksum(y[e] for e in E) <= Delta)
    model.addConstrs(u[e[0],e[1],l] <= x[e[0],l] for e in E for l in N)
    model.addConstrs(u[e[0],e[1],l] <= y[e[0],e[1]] for e in E for l in N)
    model.addConstrs(u[e[0],e[1],l] >= y[e[0],e[1]] - (1-x[e[0],l]) for e in E for l in N)
    model.addConstrs(v[e[0],e[1],l] <= x[e[1],l] for e in E for l in N)
    model.addConstrs(v[e[0],e[1],l] <= y[e[0],e[1]] for e in E for l in N)
    model.addConstrs(v[e[0],e[1],l] >= y[e[0],e[1]] - (1-x[e[1],l]) for e in E for l in N)
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)
    
    model.optimize()

    sol = {'status':model.Status, 'objbound':model.ObjBound, 'objval':model.ObjVal, 'mipgap':model.MIPGap, 'runtime':model.Runtime}
    return(sol)

def model3_ws(p_bar, p_hat, Gamma, Delta, time_limit):
    
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

    #solving model3 with warmstart
    model = Model("model3_ws")
#    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)
    
    E = [(i,j) for i in N for j in N if j>i]
    
    #variables
    x = model.addVars([(i,j) for i in N for j in N], vtype=GRB.BINARY, name="x")
    y = model.addVars([e for e in E], vtype=GRB.CONTINUOUS, name="y", lb=0)
    pi = model.addVar(vtype=GRB.CONTINUOUS, name = "pi", lb=0)
    rho = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name="rho", lb=0)
    u = model.addVars([(i,j,l) for i in N for j in N for l in N], vtype=GRB.CONTINUOUS, name="u", lb=0)
    v = model.addVars([(i,j,l) for i in N for j in N for l in N], vtype=GRB.CONTINUOUS, name="v", lb=0)
    
    #setting warmstart
    pi.start = ws_pi
    for i in N:
        rho[i].start = ws_rho[i]
        for j in N:
            x[i,j].start = ws_x[i][j]

    #objective
    model.setObjective(quicksum(p_bar[i]*(n+1-quicksum(x[i,l]*l for l in N)) for i in N) + quicksum(p_bar[e[1]]*quicksum(v[e[0],e[1],l]*l for l in N) - p_bar[e[1]]*quicksum(u[e[0],e[1],l]*l for l in N) - p_bar[e[0]]*quicksum(v[e[0],e[1],l]*l for l in N) + p_bar[e[0]]*quicksum(u[e[0],e[1],l]*l for l in N) for e in E) + Gamma*pi + quicksum(rho[i] for i in N), GRB.MINIMIZE)
    
    #constraints
    model.addConstrs(rho[i] + pi + quicksum(p_hat[e[0]]*(quicksum(v[e[0],e[1],l]*l for l in N)-quicksum(u[e[0],e[1],l]*l for l in N)) for e in E if e[0] == i) - quicksum(p_hat[e[1]]*(quicksum(v[e[0],e[1],l]*l for l in N)-quicksum(u[e[0],e[1],l]*l for l in N)) for e in E if e[1] == i) >= p_hat[i]*(n+1-quicksum(x[i,l]*l for l in N)) for i in N)
    model.addConstrs(quicksum(y[e] for e in E if (e[0] == i) or (e[1] == i)) <= 1 for i in N)
    model.addConstr(quicksum(y[e] for e in E) <= Delta)
    model.addConstrs(u[e[0],e[1],l] <= x[e[0],l] for e in E for l in N)
    model.addConstrs(u[e[0],e[1],l] <= y[e[0],e[1]] for e in E for l in N)
    model.addConstrs(u[e[0],e[1],l] >= y[e[0],e[1]] - (1-x[e[0],l]) for e in E for l in N)
    model.addConstrs(v[e[0],e[1],l] <= x[e[1],l] for e in E for l in N)
    model.addConstrs(v[e[0],e[1],l] <= y[e[0],e[1]] for e in E for l in N)
    model.addConstrs(v[e[0],e[1],l] >= y[e[0],e[1]] - (1-x[e[1],l]) for e in E for l in N)
    model.addConstrs(quicksum(x[i,j] for i in N) == 1 for j in N)
    model.addConstrs(quicksum(x[i,j] for j in N) == 1 for i in N)
    
    model.optimize()

    sol = {'status':model.Status, 'objbound':model.ObjBound, 'objval':model.ObjVal, 'mipgap':model.MIPGap, 'runtime':model.Runtime}
    return(sol)

