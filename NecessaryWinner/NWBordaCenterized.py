import tools


def Up2Down_centerized(Up,m,ldown): #m x |Up[u]| -> O(m²)
    Down = [[] for i in range(m)]
    for i in range(m):
        for j in Up[i]:
            ldown[j] += 1
            Down[j].append(i)
    return Down
       
       
def Step1_centerized(dico,Profile,m): #O(nm²)
    j = 0
    test = len(Profile)//10
    ldown = [0 for i in range(m)]
    U = []
    D = []
    for i in range(test): #n
        P = Profile[i]
        s = str(P)
        if s in dico.keys():
            dico[s][0] += 1
            j += 1
            U.append(dico[s][1])
            D.append(dico[s][2])
        else:
            u = Order2Up(P,m)
            d = Up2Down_centerized(u,m,ldown)
            dico[s] = [1,u,d]
            U.append(u)
            D.append(d)
    if j >= test//10:
        for i in range(test,len(Profile)): #n
            P = Profile[i]
            s = str(P)
            if s in dico.keys():
                dico[s][0] += 1
            else:
                j += 1
                u = Order2Up(P,m)
                d = Up2Down_centerized(u,m,ldown)
                dico[s] = [1,u,d]
        return True,[],[],ldown
    else:
        for i in range(test,len(Profile)): #n
            P = Profile[i]
            s = str(P)
            u = Order2Up(P,m)
            d = Up2Down_centerized(u,m,ldown)
            U.append(u)
            D.append(d)
        return False,U,D,ldown
    
def Step3_borda_centerized(c,w,dico,m,l=[]): #O(nm)
    Sw = 0
    Sc = 0
    for key in dico.keys(): #n
        [s,U,D] = dico[key]
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

def Step2_borda_centerized(c,dico,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_borda_centerized(c,w,dico,m)):
                return False
        return True

    
    
def isThereNcW_borda_centerized(Profile,m): #O(nm²)
    dico = dict()
    vud,U,D,ldown = Step1_centerized(dico,Profile,m)
    jtest = 0
    notfound = True
    valmax = np.sum([i+1 for i in range(m)])*len(Profile)/m
    if vud:
        listcand = [i for i in range(m)]
        firstcand = 0
        current = 0
    else:
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
            v = Step3_borda_centerized(listcand[current],listcand[w],dico,m,list_to_test)
        else:
            v = Step3_borda(listcand[current],listcand[w],U,D,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        if vud:
            v = Step3_borda_centerized(listcand[current],listcand[w],dico,m)
        else:
            v = Step3_borda(listcand[current],w,U,D,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        if vud:
            v = Step2_borda_centerized(listcand[w],dico,m)
        else:
            v = Step2_borda(listcand[w],U,D,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str([listcand[w] for w in ncw])
            

