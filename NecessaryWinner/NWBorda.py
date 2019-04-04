# Most optimized version of Xia's algorithm in the general case


##
import tools


def Order2Up(P,m,lup): 
    Up = [[i] for i in range(m)]
    is_roots = [1 for i in range(m)]
    childs = [[] for i in range(m)]
    parents = [0 for i in range(m)]
    for k in P: 
        (n1,n2) = k
        is_roots[n2] = 0
        childs[n1].append(n2)
        parents[n2] +=1
    queue = []
    for i in range(m): 
        if is_roots[i] == 1:
            queue.append(i)
    while queue != []: 
        u = queue.pop() 
        lup[u] += len(Up[u])
        for e in childs[u]: 
            Up[e].extend(Up[u]) 
            parents[e] -= 1
            if parents[e] == 0:
                queue.append(e)
    return Up
    
def Step1(Profile,m): 

    U = []
    lup = [0 for i in range(m)]
    for P in Profile: 
        u = Order2Up(P,m,lup)
        U.append(u)
    minlup = min(lup)
    list_c = []
    for i in range(m):
        if lup[i] == minlup:
            list_c.append(i)
    lenc = len(list_c)

    D = [[] for i in range(lenc)]
    for P in Profile:
        Down = [[] for i in range(lenc)]
        childs = [[] for i in range(m)]
        for k in P: 
            (n1,n2) = k
            childs[n1].append(n2)
        for j in range(lenc):
            c = list_c[j]
            visited = [False for i in range(m)]
            Downc = [c]
            queue = childs[c].copy()
            while queue != []:
                nc = queue.pop()
                Downc.append(nc)
                for cc in childs[nc]:
                    if not(visited[cc]):
                        visited[cc] = True
                        queue.append(cc)
            D[j].append(Downc)
    return U,D,list_c

    
def Step3_borda(c,w,U,D,m,l=[]): 
    n = len(U)
    Sw = 0
    Sc = 0
    for i in range(n):
        if c in U[i][w]:
            block_size = intersect(D[i],U[i][w])
            Sc += block_size
        else:
            Sw += m-len(U[i][w])
            Sc += len(D[i])-1
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)

def Step2_borda(i,c,U,D,m): 
    for w in range(m): 
        if c != w:
            if not(Step3_borda(c,w,U,D,m)):
                return False
    return True
    
    
def isThereNcW_borda(Profile,m): 
    a = time.time()
    U,D,list_c = Step1(Profile,m)
    b = time.time()
    list_to_test = []
    notNW = []
    for i in range(m):
        if not(i in list_c):
            notNW.append(i)
    listcand = notNW + list_c
    firstcand = len(notNW)
    current = firstcand
    ncw = []
    for i in range(len(list_c)):
        v = Step2_borda(i,list_c[i],U,D[i],m)
        if v:
            ncw.append(w)
    c = time.time()
    if len(ncw) ==0:
        return "There is no co-necessary winner",b-a,c-b
    return "The necessary co-winners are "+str(ncw),b-a,c-b


