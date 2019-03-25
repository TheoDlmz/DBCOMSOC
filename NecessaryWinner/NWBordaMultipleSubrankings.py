import tools
import step1


        
def Step1_mulsub(Profile,m):
    sr = []
    cl = []
    for i in range(len(Profile)):
        srv = [(-1,-1) for i in range(m)]
        clv = []
        if Profile[i] != []:
            tops = []
            P = [0 for i in range(m)]
            C = [[] for i in range(m)]
            for (a,b) in Profile[i]:
                C[a].append(b)
                P[b] += 1
            i = 0
            for j in range(m):
                if P[j] == 0:
                    tops.append(j)
            for i in range(len(tops)):

                current_rank = 0
                j = tops[i]
                while j != -1:
                    srv[j] = (i,current_rank)
                    current_rank +=1
                    if len(C[j]) == 0:
                        j = -1
                    else:
                        j = C[j][0]
                clv.append(current_rank)
            cl.append(clv)
            sr.append(srv)
        else:
            srv = [(-1,-1) for i in range(m)]
            sr.append(srv)
            cl.append([])
    return sr,cl
    

def Step3_borda_mulsub(c,w,sr,cl,m,l=[]): #O(nm)
    n = len(sr)
    Sw = 0
    Sc = 0
    for i in range(n): #n
        (ssc,spc) = sr[i][c]
        (ssw,spw) = sr[i][w]
        if (spc >= 0) and (spc < spw) and (ssw == ssc): #O(|U[i,w]|)
            block_size = spw - spc #O(1)
            Sc += block_size
        else:
            if ssc != -1:
                Sc += cl[i][ssc]-spc-1
            Sw += m-1-max(spw,0)
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)
   

def Step2_borda_mulsub(c,sr,cl,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_borda_mulsub(c,w,sr,cl,m)):
                return False
        return True
        
        
def isThereNcW_borda_mulsub(Profile,m): #O(nm²)
    current = 0
    sr,cl = Step1_mulsub(Profile,m)
    list_to_test = []
    for w in range(1,m): 
        v = Step3_borda_mulsub(current,w,sr,cl,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        v = Step3_borda_mulsub(current,w,sr,cl,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        v = Step2_borda_mulsub(w,sr,cl,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str(ncw)
    
    
    