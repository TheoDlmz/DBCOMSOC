q = [['w',1],['items',0,1,5,"1"]]


def isPairwiseDisconnected(q):
    return 0
    
def computeSet(q,i):
    return 0
    
def computeTupleSet(q):
    return 0
    
def possibilitySet(J,votes,n,m):
    possible = listOfFirst(votes,m)
    t,n = aggregate(possible,m)
    for c in J:
        if possibleWinner(t,n,m,c) == [c]:
            return True
    return False

def scoreTuple(dico,T,N,m,c):
    cstr = str(sorted(c))
    if cstr in dico.keys():
        (score,Fset,Fscore) = dico[cstr]
        return score, Fset, Fscore
    lenc = len(c)
    if lenc == 1:
        cand = c[0]
        scorei = 0
        for j in range(len(T)):
            if T[j][cand] == 1:
                scorei += N[j]
        dico[cstr] = (scorei,[],[])
        return scorei,[],[]
    else:
        for i in range(lenc):
            scorei = 0
            xi = c[i]
            Ti = []
            Ni = []
            for j in range(len(T)):
                ok = True
                k = 0
                while k < lenc and ok:
                    if c[k] != xi and T[j][c[k]] == 1:
                        ok = False
                    k += 1
                if ok and T[j][xi] == 1:
                    scorei += N[j]
                else:
                    Ti.append(T[j])
                    Ni.append(N[j])
            ci = c.copy()
            ci.pop(i)
            scoreT,Fset,Fscore = scoreTuple(dico,Ti,Ni,m,ci)
            if scoreT < scorei:
                newF = [0 for k in range(m)]
                for k in c:
                    if k == xi:
                        newF[k] = 1
                    else:
                        newF[k] = -1
                Fset.append(newF)
                Fscore.append(scorei-scoreT)
                dico[cstr] = (scoreT,Fset,Fscore)
                return scoreT,Fset,Fscore
        totalscore = 0
        for j in range(len(T)):
            ok = True
            k = 0
            while k < lenc and ok:
                if T[j][c[k]] == 1:
                    ok = False
                    totalscore += N[j]
                k += 1
        meanscore = totalscore//lenc
        freevoters = totalscore%lenc
        Fset = [0 for i in range(m)]
        for cand in c:
            Fset[cand] = 1
        if freevoters > 0:
            dico[cstr] = (meanscore, [Fset], [freevoters])
            return meanscore, [Fset], [freevoters]
        else:
            dico[cstr] = (meanscore, [], [])
            return meanscore, [], []

def buildMatrixTuple(g,score,Fset,Fscore,T,N,m,c):
    P_empty = []
    N_empty = []
    P_sets = [[] for i in range(len(Fset))]
    N_sets = [[] for i in range(len(Fset))]

    P_sets_len = 0
    for i in range(len(N)):
        notInC= True
        for j in c:
            if T[i][j] == 1:
                notInC = False
        if notInC:
            l = T[i].copy()
            jpop = 0
            c2 = sorted(c)
            c2.reverse()
            for j in c2:
                l.pop(j)
                jpop +=1
            N_empty.append(N[i])
            P_empty.append(l)
        else:
            a = True
            found = -1
            j = 0
            while j < (len(Fset)) and a:
                b = True
                k = 0
                while k < (len(Fset[j])) and b:
                    if (Fset[j][k] == 1 and T[i][k] == 0) or (Fset[j][k] == -1 and T[i][k] == 1):
                        b = False
                    k+=1
                if k == len(Fset[j]) and b:
                    a = False
                    found = j
                j += 1
            if not(a):
                l = T[i].copy()
                for k in c:
                    l.pop(k)
                N_sets[found].append(N[i])
                P_sets[found].append(l)
                P_sets_len += 1
    lenc = len(c)
    lenF = len(Fset)
    lempty = len(P_empty)
    nodes = g.add_nodes(lempty+lenF+P_sets_len+m-lenc)
    maxwanted = 0
    
    for i in range(lempty):
        g.add_tedge(i,N_empty[i],0)
        maxwanted += N_empty[i]
        for j in range(m-lenc):
            if P_empty[i][j] >0:
                g.add_edge(i,lempty+lenF+P_sets_len+j,N_empty[i],0)
    z = 0
    for i in range(lenF):
        g.add_tedge(lempty+i,Fscore[i],0)
        maxwanted += Fscore[i]
        for j in range(len(P_sets[i])):
            g.add_edge(lempty+i,lempty+lenF+z,N_sets[i][j],0)
            for k in range(m-lenc):
                g.add_edge(lempty+lenF+z,lempty+lenF+P_sets_len+k,N_sets[i][j],0)
            z+=1
    for i in range(m-lenc):
        g.add_tedge(lempty+lenF+P_sets_len+i,0,score)
    return maxwanted

def possibilityTuple(dico,T,N,m,c):
    clen = len(c)
    score, Fset, Fscore = scoreTuple(dico,T,N,m,c)
    g = mf.GraphInt()
    maxwanted = buildMatrixTuple(g,score,Fset,Fscore,T,N,m,c)
    #print(maxwanted)
    maxflow = g.maxflow()
    #print(maxflow)
    #print(maxflow,maxwanted,score)
    #print(Fscore,Fset)
    if maxflow >= maxwanted:
        return True
    else:
        return False
    
def possibilityTupleSet(votes,m,C):
    dico =dict()
    for c in C:
        #print("test : ",c)
        possible = listOfFirst(votes,m)
        T,N = aggregate(possible,m)
        if possibilityTuple(dico,T,N,m,c):
            print("Possible winners :",c)
            #return True, c
    return False

#Optimize : ajout dictionnaire pour memoriser les dejas vus

def possibility(q,D):
    return 0
    
def necessity(q,D):
    return 0
    
##

def testMW(n,m,k,phi=0.3,psi=0.5):
    v = create_votes_3(n,m,phi,psi,k)
    print("test 1")
    b = [[i] for i in range(min(m,10))]
    ta = time.time()
    possibilityTupleSet(v,m,b)
    tb = time.time()
    return (ta-tb)/len(b)

##

m = 20
y =  []
for n in [100,1000,10000,100000]:
    y.append(testMW(n,m,1))
##
import matplotlib.pyplot as plt
plt.plot([2,3,4,5],[np.log(-e) for e in y],'k',marker='o')
plt.xlabel("log(#voters)")
plt.ylabel("log(time(s)) per set")
plt.title("Running time with m = 20 candidates")
plt.show()