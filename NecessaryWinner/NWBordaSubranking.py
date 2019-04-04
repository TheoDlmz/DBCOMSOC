# Use this implementation of Xia's algorithm if every voter's preference is in class UNILINEAR

##

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
    

def Step3_borda_sub(c,w,sr,m,l=[]): 
    n = len(sr)
    Sw = 0
    Sc = 0
    for i in range(n): 
        if (sr[i][c] >= 0) and (sr[i][c] < sr[i][w]): 
            block_size = sr[i][w] - sr[i][c] 
            Sc += block_size
        else:
            if sr[i][c] != -1:
                Sc += sr[i][-1]-sr[i][c]-1
            Sw += m-1-max(sr[i][w],0)
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)
   

def Step2_borda_sub(c,sr,m): 
    for w in range(m):
        if c != w:
            if not(Step3_borda_sub(c,w,sr,m)):
                return False
        return True
        
        
def isThereNcW_borda_sub(Profile,m): 
    a = time.time()
    sr,lup = Step1_sub(Profile,m)
    b = time.time()
    minlup = min(lup)
    candNW = []
    notNW = []
    for i in range(m):
        if lup[i] == minlup:
            candNW.append(i)
        else:
            notNW.append(i)
    firstcand = len(notNW)
    listcand = notNW + candNW
    current = firstcand
    list_to_test = []
    for w in range(firstcand+1,m): 
        v = Step3_borda_sub(listcand[current],listcand[w],sr,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        v = Step3_borda_sub(listcand[current],listcand[w],sr,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(listcand[current])
    for w in list_to_test:
        v = Step2_borda_sub(listcand[w],sr,m)
        if v:
            ncw.append(listcand[w])
    c = time.time()
    if len(ncw) ==0:
        return "There is no co-necessary winner",b-a,c-b
    return "The necessary co-winners are "+str(ncw),b-a,c-b