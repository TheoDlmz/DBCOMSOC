##IMPORTS 

from collections import defaultdict 
import numpy as np
import maxflow as mf


###Algorithm for possible winner under plurality


def votes2posiblewinnerB(votes,m):
    n = len(votes)
    matrixRank2 = []
    for i in range(n):
        seen = [1]*m
        v = votes[i]
        for x in v:
            (a,b) =x
            if (seen[b]==1):
                seen[b] = 0
        matrixRank2.append(seen)
    return matrixRank2

def pwStep2(matrixRank2,m):
    #pw2m = [2**i for i in range(m)]
    dico = dict()
    tab = []
    numb = []
    i = 0
    for s in matrixRank2:
        #v = 0
        #for i in range(m):
        #    v += pw2m*s[i]
        if str(s) in dico.keys():
            numb[dico[str(s)]] += 1
        else:
            numb.append(1)
            tab.append(s)
            dico[str(s)] = i
            i+=1
    return tab,numb
        
    
def build_matrixB(g,score,N,M,m,c,query):
    P1 = len(N)
    if query == []:
        query = [1 for i in range(m)]
    size = P1 + m - 1
    nodes = g.add_nodes(size)
    for i in range(P1):
        g.add_tedge(i,N[i],0)
        for j in range(m-1):
            if M[i][j] >0:
                g.add_edge(i,P1+j,N[i],0)
    for i in range(m-1):
        g.add_tedge(P1+i,0,score-query[i])
    return size

def try_heuristic(score,N,M,m):
    n = len(N)
    init = [score-1 for i in range(m-1)]
    for i in range(n):
        j_tab = list(np.argsort(init))
        j_tab.reverse()
        left = N[i]
        for k in range(len(j_tab)):
            j = j_tab[k]
            if M[i][j] == 1:
                suppr = min(init[j],left)
                left -= suppr
                init[j] -= suppr
        if left > 0:
            return False
    return True
    
        
        
        
        

def possibleWinnerB(t,n,m,c,q=[]):
    print(c)
    M= []
    score = 0
    N = []
    maxwanted = 0
    for i in range(len(n)):
        if t[i][c] == 1:
            score += n[i]
        else:
            l = t[i].copy()
            l.pop(c)
            maxwanted += n[i]
            N.append(n[i])
            M.append(l)
    if score>maxwanted:
        return [c]

    #ta = time.time()
    if try_heuristic(score,N,M,m):
        return [c]
    #tb = time.time()
    g = mf.GraphInt()
    size = build_matrixB(g,score,N,M,m,c,[])
    #tc = time.time()
    maxflow = g.maxflow()
    #td = time.time()
    #print(c,tb-ta,tc-tb,td-tc)
    if maxflow >= maxwanted:
        return [c]
    else:
        return []


def isTherePossibleWinnerB(votes,m):
    possible = votes2posiblewinnerB(votes,m)
    t,n = pwStep2(possible,m)
    set = []
    for c in range(m):
        set += possibleWinnerB(t,n,m,c)
    print("The set of possible winner is ",set)
    return set




##Test



def testPW(n,m,dataset):
    print("START")
    s1,s2=0,0
    for i in range(10):
        votes = read_dataset(dataset,n,m,i)
        a = time.time()
        nw = isTherePossibleWinnerB(votes,m)
        b = time.time()
        s2 += b-a
        print(i,"(",(b-a),")")
    print(m," x ",n," : ",s2/(10))
    print("END")