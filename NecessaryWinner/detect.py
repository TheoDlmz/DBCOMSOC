# These functions enable to detect the class of a profile of preferences :
# 0 : Class SPLIT + MERGE
# 1 : Class UNILINEAR
# 2 : Class MULTILINEAR
# 3 : Class MERGE
# 4 : Class SPLIT

# TODO : Return for each class the set of voters in this class. (Hybrid)


###

import tools


def detect_sub(Profile,m):
    c = True
    for i in range(len(Profile)):
        if Profile[i] != []:
            P = [[] for i in range(m)]
            C = [0 for i in range(m)]
            for (a,b) in Profile[i]:
                P[b].append(a)
                C[a] += 1
            top = 0
            bot = 0
            for i in range(m):
                if len(P[i]) > 1 or C[i] > 1:
                    return 0,i
                else:
                    if len(P[i]) == 1 and C[i] ==0:
                        top +=1
                    elif len(P[i]) == 0 and C[i] == 1:
                        bot +=1
            if top != bot:
                return 0,i
            elif top != 1:
                c = False
    if c:
        return 1,0 
    else:
        return 2,0 

def detect(Profile,m): 
    v,i = detect_sub(Profile,m)
    if v != 0:
        return v
    else:
        merged = False
        splitted = False
        for j in range(i,len(Profile)):
            V = [[0,0] for i in range(m)]
            for (a,b) in Profile[i]:
                if V[a][0] > 0 and not(splitted):
                    if merged:
                        return 0
                    else:
                        splitted = True
                if V[b][1] > 0 and not(merged):
                    if splitted:
                        return 0
                    else:
                        merged = True
                V[a][0] += 1
                V[b][1] += 1
        if merged:
            return 3
        else:
            return 4