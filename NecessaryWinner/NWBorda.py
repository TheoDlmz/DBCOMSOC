import tools
import step1

def Step3_borda(c,w,U,D,m,l=[]): #O(nm)
    n = len(U)
    Sw = 0
    Sc = 0
    for i in range(n): #n
        if c in U[i][w]: #O(|U[i,w]|)
            block_size = intersect(D[i][c],U[i][w]) #O(1)
            Sc += block_size
        else:
            Sw += m-len(U[i][w]) #O(1)
            Sc +=  len(D[i][c])-1 #O(1)
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)

def Step2_borda(c,U,D,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_borda(c,w,U,D,m)):
                return False
        return True

def NW_borda(c,Profile,m):
    D,U = Step1(Profile,m)
    return Step2_borda(c,U,D,m)
    
def isThereNW_borda(Profile,m): #O(nm²)
    current = 0
    D,U = Step1(Profile,m)
    for w in range(1,m): 
        v = Step3_borda(current,w,U,D,m)
        if not(v):
            current = w
    for w in range(current): 
        v = Step3_borda(current,w,U,D,m)
        if not(v):
            return "There is no necessary winner"
    return "The necessary winner is "+str(current)

    
def isThereNcW_borda(Profile,m): #O(nm²)
    current = 0
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
        v = Step3_borda(listcand[current],listcand[w],U,D,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        v = Step3_borda(listcand[current],listcand[w],U,D,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        v = Step2_borda(listcand[w],U,D,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str([listcand[w] for w in ncw])
    
    



