# Use these implementation of Xia's algo if you think a lot of voters have the same rankings (in case of a CONSENSUS)

##


import tools
import NWBorda

def Order2Up_dico(dico,P,m): 
    Up = [[i] for i in range(m)]
    is_roots = [1 for i in range(m)]
    childs = [[] for i in range(m)]
    parents = [0 for i in range(m)]
    for k in P:
        (n1,n2) = k
        is_roots[n2] = 0
        childs[n1].append(n2)
        parents[n2] += 1
    queue = []
    for i in range(m): 
        if is_roots[i] == 1:
            queue.append(i)
    while queue != []: 
        u = queue.pop() 
        Up[u] = list(set(Up[u]))
        for e in childs[u]: 
            Up[e].extend(Up[u]) 
            parents[e] -= 1
            if parents[e] == 0:
                queue.append(e)
    return Up


def Step1_dico(dico_ranks,dico_models,Profile,m):
    test = len(Profile)//10
    ldown = [0 for i in range(m)]
    U = []
    D = []
    for i in range(test): 
        P = Profile[i]
        s = str(sorted(P))
        if s in dico_ranks.keys():
            dico_ranks[s][0] += 1
            U.append(dico_ranks[s][1])
            D.append(dico_ranks[s][2])
        else:
            u = Order2Up_dico(dico_models,P,m)
            d = Up2Down(u,m,ldown)
            dico_ranks[s] = [1,u,d]
            U.append(u)
            D.append(d)
    if len(dico_ranks) < (test*9/10):
        for i in range(test,len(Profile)): 
            P = Profile[i]
            s = str(sorted(P))
            if s in dico_ranks.keys():
                dico_ranks[s][0] += 1
            else:
                u = Order2Up_dico(dico_models,P,m)
                d = Up2Down(u,m,ldown)
                dico_ranks[s] = [1,u,d]
        return True,[],[],ldown
    else:
        for i in range(test,len(Profile)): 
            P = Profile[i]
            u = Order2Up_dico(dico_models,P,m)
            d = Up2Down(u,m,ldown)
            U.append(u)
            D.append(d)
        return False,U,D,ldown
    
def Step3_borda_dicoc,w,dico,m,l=[]):
    Sw = 0
    Sc = 0
    for key in dico.keys(): 
        [s,U,D] = dico[key]
        if c in U[w]: 
            block_size = intersect(D[c],U[w]) 
            Sc += block_size*s
        else:
            Sw += (m-len(U[w]))*s 
            Sc +=  (len(D[c])-1)*s 
        
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)

def Step2_borda_dico(c,dico,m):
    for w in range(m):
        if c != w:
            if not(Step3_borda_dico(c,w,dico,m)):
                return False
    return True
    
    
def isThereNcW_borda_dico(Profile,m):   
    a = time.time()
    dico_ranks,dico_models = dict(),dict()
    vud,U,D,ldown = Step1_dico(dico_ranks,dico_models,Profile,m)
    b = time.time()
    jtest = 0
    notfound = True
    valmax = np.sum([i+1 for i in range(m)])*len(Profile)/m
    notNW = []
    candNW = []
    max = 0
    if vud:
        listcand = [i for i in range(m)]
        firstcand = 0
        current = 0
    else:
        for jtest in range(m):
            lowest_score = ldown[jtest]
            if lowest_score >= valmax:
                notfound = False
                if lowest_score < max:
                    notNW.append(jtest)
                elif lowest_score > max:
                    notNW += candNW
                    candNW = [jtest]
                    max = lowest_score
                else:
                    candNW.append(jtest)
            else:
                notNW.append(jtest)
        if notfound:
            c = time.time()
            return "There is no co-necessary winner",b-a,c-b
        firstcand = len(notNW)
        current = firstcand
        listcand = notNW + candNW
    list_to_test = []
    for w in range(firstcand+1,m):
        if vud:
            v = Step3_borda_dico(listcand[current],listcand[w],dico_ranks,m,list_to_test)
        else:
            v = Step3_borda(listcand[current],listcand[w],U,D,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        if vud:
            v = Step3_borda_dico(listcand[current],listcand[w],dico_ranks,m)
        else:
            v = Step3_borda(listcand[current],listcand[w],U,D,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(listcand[current])
    for w in list_to_test:
        if vud:
            v = Step2_borda_dico(listcand[w],dico_ranks,m)
        else:
            v = Step2_borda(listcand[w],U,D,m)
        if v:
            ncw.append(listcand[w])
    c = time.time()
    if len(ncw) ==0:
        return "There is no co-necessary winner",b-a,c-b
    return "The necessary co-winners are "+str(ncw),b-a,c-b