import tools


def Order2Up_dico(dico,P,m,cont_models): #O(max(|P|,m)x|Up[u]) -> O(m²)
    Up = [[i] for i in range(m)]
    is_roots = [1 for i in range(m)]
    childs = [[] for i in range(m)]
    parents = [0 for i in range(m)]
    for k in P: #O(|P|)
        (n1,n2) = k
        is_roots[n2] = 0
        childs[n1].append(n2)
        parents[n2] += 1
    if cont_models:
        queue = []
        curr = 0
        revcurr = 0
        mapping = [-1 for i in range(m)]
        reversedmapping = [-1 for i in range(m)]
        mapsto = []
        for i in range(m): #O(m)
            if is_roots[i] == 1:
                queue.append(i)
                if len(childs[i]) > 0:
                    mapping[i] = curr
                    reversedmapping[curr] = i
                    curr += 1
                else:
                    mapping[i] = m-1-revcurr
                    reversedmapping[m-1-revcurr] = i
                    revcurr += 1
        queue2 = queue.copy()
        while queue != []: #m times
            u = queue.pop() #O(1)
            for e in childs[u]:  # |P| times
                if mapping[e] == -1:
                    mapping[e] = curr
                    reversedmapping[curr] = e
                    curr += 1
                mapsto.append((mapping[u],mapping[e])) #O(|Up[u]|)
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
        if str(mapsto) in dico.keys():
            canonUp = dico[str(mapsto)]
            Up = [[] for i in range(m)]
            for i in range(m):
                ym = reversedmapping[i]
                for j in canonUp[i]:
                    Up[ym].append(reversedmapping[j])
            return Up
        else:
            queue = queue2
            while queue != []: #m times
                u = queue.pop() #O(1)
                Up[u] = list(set(Up[u])) #O(|Up[u]|) 
                for e in childs[u]:  # |P| times
                    Up[e].extend(Up[u]) #O(|Up[u]|)
                    parents[e] -= 1
                    if parents[e] == 0:
                        queue.append(e)
            canonUp = [[] for i in range(m)]
            for i in range(m):
                ym = mapping[i]
                for j in Up[i]:
                    canonUp[ym].append(mapping[j])
            dico[str(mapsto)] = canonUp
            return Up
    else:
        queue = []
        for i in range(m): #O(m)
            if is_roots[i] == 1:
                queue.append(i)

        while queue != []: #m times
            u = queue.pop() #O(1)
            Up[u] = list(set(Up[u])) #O(|Up[u]|) 
            for e in childs[u]:  # |P| times
                Up[e].extend(Up[u]) #O(|Up[u]|)
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
        return Up


def Step1_dico(dico_ranks,dico_models,Profile,m): #O(nm²)
    test = len(Profile)//10
    ldown = [0 for i in range(m)]
    U = []
    D = []
    for i in range(test): #n
        P = Profile[i]
        s = str(sorted(P))
        if s in dico_ranks.keys():
            dico_ranks[s][0] += 1
            U.append(dico_ranks[s][1])
            D.append(dico_ranks[s][2])
        else:
            u = Order2Up_dico(dico_models,P,m,True)
            d = Up2Down(u,m,ldown)
            dico_ranks[s] = [1,u,d]
            U.append(u)
            D.append(d)
    cont_models = (len(dico_models) < test/10)
    if len(dico_ranks) < test:
        for i in range(test,len(Profile)): #n
            P = Profile[i]
            s = str(sorted(P))
            if s in dico_ranks.keys():
                dico_ranks[s][0] += 1
            else:
                u = Order2Up_dico(dico_models,P,m,cont_models)
                d = Up2Down(u,m,ldown)
                dico_ranks[s] = [1,u,d]
        return True,[],[],ldown
    else:
        for i in range(test,len(Profile)): #n
            P = Profile[i]
            u = Order2Up_dico(dico_models,P,m,cont_models)
            d = Up2Down(u,m,ldown)
            U.append(u)
            D.append(d)
        return False,U,D,ldown
    
def Step3_borda_dico(c,w,dico,m,l=[]): #O(nm)
    Sw = 0
    Sc = 0
    for i in range(m*m):
        for key in dico[i].keys(): #n
            [s,U,D] = dico[i][key]
            #print(s)
            if c in U[w]: #O(|U[i,w]|)
                block_size = intersect(D[c],U[w]) #O(1)
                Sc += block_size*s
            else:
                Sw += (m-len(U[w]))*s #O(1)
                Sc +=  (len(D[c])-1)*s #O(1)
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)

def Step2_borda_dico(c,dico,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_borda_dico(c,w,dico,m)):
                return False
        return True
    
    
def isThereNcW_borda_dico(Profile,m): #O(nm²)
    dico_ranks,dico_models = dict(),dict()
    vud,U,D,ldown = Step1_dico(dico_ranks,dico_models,Profile,m)
    jtest = 0
    notfound = True
    valmax = np.sum([i+1 for i in range(m)])*len(Profile)/m
    notNW = []
    candNW = []
    max = 0
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
        return "There is no co-necessary winner"
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
            v = Step3_borda(listcand[current],w,U,D,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        if vud:
            v = Step2_borda_dico(listcand[w],dico_ranks,m)
        else:
            v = Step2_borda(listcand[w],U,D,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str([listcand[w] for w in ncw])