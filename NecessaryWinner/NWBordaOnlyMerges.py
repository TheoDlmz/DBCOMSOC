import tools
import step1

        
def Step1_merge(Profile,m):
    sr = []
    cl = []
    rk = []
    for i in range(len(Profile)):
        srv = [([],-1,-1) for i in range(m)]
        ranking_length = []
        rankings = [[] for i in range(m)]
        if Profile[i] != []:
            P = [[] for i in range(m)]
            C = [[] for i in range(m)]
            for (a,b) in Profile[i]:
                C[a].append(b)
                P[b].append(a)
            i = 0
            leaves = []
            P_count = [0 for i in range(m)]
            rankings = [[] for i in range(m)]
            tops = []
            rn = 0
            for j in range(m):
                if len(P[j]) == 0:
                    leaves.append(j)
                    rankings[j].append(rn)
                    rn += 1
                if len(C[j]) == 0:
                    tops.append(j)
                P_count[j] = len(P[j])
            ranking_length = [0 for i in range(rn)]
            up = [1 for i in range(m)]
            while leaves != []:
                a = leaves.pop()
                for b in C[a]:
                    P_count[b] -= 1
                    up[b] += up[a]
                    rankings[b].extend(rankings[a])
                    if P_count[b] == 0:
                        leaves.append(b)
            for i in range(len(tops)):
                current_rank = 0
                j_arr = [tops[i]]
                while j_arr != []:
                    j_arr_new = []
                    for j in j_arr:
                        srv[j] = (rankings[j],current_rank,up[j])
                        if len(P[j]) == 0:
                            ranking_length[rankings[j][0]] = current_rank+1
                        else:
                            j_arr_new.extend(P[j])
                    current_rank +=1
                    j_arr = j_arr_new.copy()
        cl.append(ranking_length)
        sr.append(srv)
    return sr,cl
    

def Step3_borda_merge(c,w,sr,cl,m,l=[]): #O(nm)
    n = len(sr)
    Sw = 0
    Sc = 0
    for i in range(n): #n
        (ssc,spc,uc) = sr[i][c]
        (ssw,spw,uw) = sr[i][w]
        if (spc >= 0) and (spc > spw) and (ssc[0] in ssw): #O(|U[i,w]|)
            block_size = spc - spw #O(1)
            Sc += block_size
        else:
            Sc += max(spc,0) #no->Dynamic programming to compute Down
            Sw += m-1-uw
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)
   

def Step2_borda_merge(c,sr,cl,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_borda_merge(c,w,sr,cl,m)):
                return False
        return True
        
        
def isThereNcW_borda_merge(Profile,m): #O(nm²)
    current = 0
    sr,cl = Step1_split(Profile,m)
    list_to_test = []
    for w in range(1,m): 
        v = Step3_borda_split(current,w,sr,cl,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        v = Step3_borda_split(current,w,sr,cl,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        v = Step2_borda_split(w,sr,cl,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str(ncw)