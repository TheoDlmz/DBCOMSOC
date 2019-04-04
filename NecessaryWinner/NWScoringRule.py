import tools
import step1

def precompute_score(s,m):
    M = np.zeros((m-1,m-1,m-1)) #min-1,max-1,block size
    for i in range(m):
        for j in range(i+1,m):
            M[i][j-1][j-i-1] = s[i] - s[j]
    for k in range(1,m):
        for i in range(m):
            for j in range(i+k+1,m):
                M[i][j-1][j-i-k-1] = min(s[i+k]-s[j],M[i][j-2][j-i-k-1])
    return M

def Step3_score(s,M,c,w,U,D,m):
    n = len(U)
    Sw = 0
    Sc = 0
    for i in range(n):
        if c in U[i][w]:
            min = s[0]-s[m-1]+1
            armin = -1
            block_size = intersect(D[i][c],U[i][w])
            Sc += M[len(U[i][c])-1,m-len(D[i][w])-1,block_size-1]
        else:
            Sw += s[len(U[i][w])-1]
            Sc += s[m - len(D[i][c])]
    return (Sw <= Sc)

def Step2_score(s,M,c,U,D,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_score(s,M,c,w,U,D,m)):
                return False
        return True

    
def isThereNcW_score(s,Profile,m):
    current = 0
    M = precompute_score(s,m)
    D,U,ldown = Step1(Profile,m)
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
    for w in range(firstcand,m): 
        v = Step3_score(s,M,listcand[current],listcand[w],U,D,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        v = Step3_score(s,M,listcand[current],listcand[w],U,D,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        v = Step2_score(s,M,listcand[w],U,D,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str([listcand[w] for w in ncw])
