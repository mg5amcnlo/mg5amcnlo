import pandas as pd
from pulp import *
import os
import sys

df = pd.read_csv("EWoverestimates/data.csv")

prob = LpProblem("Overestimate", LpMinimize)
x0 = LpVariable("c0", lowBound = 0)
x1 = LpVariable("c1", lowBound = 0)
x2 = LpVariable("c2", lowBound = 0)
x3 = LpVariable("c3", lowBound = 0)

sum = df.sum(axis = 0)/df.shape[0]
#print(sum)

if sum[0] < 1E-10:
    os.remove("EWoverestimates/consts.dat")
    sys.exit()

prob += sum[1]*x0/sum[0] + sum[2]*x1/sum[0] + sum[3]*x2/sum[0] + sum[4]*x3/sum[0] + (x0 + x1 + x2 + x3)/4.
for i in range(df.shape[0]):
    prob += df.at[i,'c0']*x0 + df.at[i,'c1']*x1 + df.at[i,'c2']*x2 + df.at[i,'c3']*x3 >= df.at[i, 'val']

prob.writeLP("EWoverestimates/LinearProgramming.lp")
prob.solve()

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Status:", LpStatus[prob.status])
print("Objective function = ", value(prob.objective))

if value(x0) is None: x0Val = 0
else : x0Val = value(x0)

if value(x1) is None: x1Val = 0
else : x1Val = value(x1)

if value(x2) is None: x2Val = 0
else : x2Val = value(x2)

if value(x3) is None: x3Val = 0
else : x3Val = value(x3)

with open("EWoverestimates/consts.dat", "w") as myfile:
    myfile.write("%s %s %s %s\n" % (x0Val, x1Val, x2Val, x3Val))