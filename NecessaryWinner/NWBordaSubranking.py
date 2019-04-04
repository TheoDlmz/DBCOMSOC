import tools
import step1


def Step1_sub(Profile,m):
    sr = []
    lup = [0 for i in range(m)]
    for i in range(len(Profile)):
        srv = [-1 for i in range(m)]
        if Profile[i] != []:
            P = [0 for i in range(m)]
            C = [[] for i in range(m)]
            for (a,b) in Profile[i]:
                C[a].append(b)
                P[b] += 1
            j = -1
            current_rank = 0
            i = 0
            while j == -1:
                if len(C[i])>P[i]:
                    j = i
                else:
                    i+=1
            while j != -1:
                srv[j] = current_rank
                lup[j] += current_rank
                current_rank +=1
                if len(C[j]) == 0:
                    j = -1
                else:
                    j = C[j][0]
            srv.append(current_rank)
            sr.append(srv)
        else:
            srv = [-1 for i in range(m)]
            srv.append(0)
            sr.append(srv)
    return sr,lup
    

def Step3_borda_sub(c,w,sr,m,l=[]): #O(nm)
    n = len(sr)
    Sw = 0
    Sc = 0
    for i in range(n): #n
        if (sr[i][c] >= 0) and (sr[i][c] < sr[i][w]): #O(|U[i,w]|)
            block_size = sr[i][w] - sr[i][c] #O(1)
            Sc += block_size
        else:
            if sr[i][c] != -1:
                Sc += sr[i][-1]-sr[i][c]-1
            Sw += m-1-max(sr[i][w],0)
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)
   

def Step2_borda_sub(c,sr,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_borda_sub(c,w,sr,m)):
                return False
        return True
        
        
def isThereNcW_borda_sub(Profile,m): #O(nm²)
    sr,lup = Step1_sub(Profile,m)
    test = min(lup)
    candPW = []
    notPW = []
    for j in range(m):
        if lup[j] == test:
            candPW.append(j)
        else:
            notPW.append(j)
    firstcand = len(notNW)
    current = firstcand
    listcand = notNW + candNW
    list_to_test = []
    for w in range(firstcand,m): 
        v = Step3_borda_sub(current,w,sr,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        v = Step3_borda_sub(current,w,sr,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        v = Step2_borda_sub(w,sr,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str(ncw)