
import numpy as np
import random
from multiprocessing import Pool
import time

def intersect(a, b):
    return len(list(set(a) & set(b))) - 1 
    
## BORDA

def s1_borda_O(C,P,roots,m,lup): 
    Up = [[i] for i in range(m)]
    isMerge = (np.max([len(c) for c in C]) > 1)
    isSplit = (np.max([len(p) for p in P]) > 1)
    queue = roots.copy()
    parents = [len(p) for p in P]
    
    if isSplit and isMerge:
        while queue != []: 
            u = queue.pop() 
            Up[u] = list(set(Up[u])) 
            lup[u] += len(Up[u])-1
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    else:
        while queue != []: 
            u = queue.pop() 
            lup[u] += len(Up[u])-1
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    return Up 



    

def s1_borda_P(C,roots,m,lup): 
    srv = [-1 for i in range(m)]
    ranks = []
    current_rank = 0
    queue = roots.copy()
    ranks.append(0)
    while queue != []:
        j = 0
        for c in queue:
            srv[c] = current_rank
            lup[c] += ranks[-1]
            j += 1
        c0 = queue[0]
        queue = C[c0].copy()
        ranks.append(ranks[-1]+j)
        current_rank += 1
    return srv,ranks



def s1_borda_S(C,roots,m,lup):
    srv = [-1 for i in range(m)]
    j = -1
    current_rank = 0
    j = roots[0]
    while j != -1:
        srv[j] = current_rank
        lup[j] += current_rank
        current_rank +=1
        if len(C[j]) == 0:
            j = -1
        else:
            j = C[j][0]
    srv.append(current_rank)
    return srv
    
        
def s1_borda_M(C,roots,m,lup):
    srv = [(-1,-1) for i in range(m)]
    clv = []
    tops = []
    i = 0
    for i in range(len(roots)):
        current_rank = 0
        j = roots[i]
        while j != -1:
            srv[j] = (i,current_rank)
            lup[j] += current_rank
            current_rank +=1
            if len(C[j]) == 0:
                j = -1
            else:
                j = C[j][0]
        clv.append(current_rank)
    return srv,clv
    
  
def updown_borda(Population,m,verbose=False,optim_step1=True,optim_step2=True):
    lup = [0 for i in range(m)]
    sr_p = []
    rl_p = []
    sr_s = []
    sr_m = []
    cl_m = []
    U = []
    
    n = len(Population)
    
    other = []
    
    for p in range(n):
        pairs = Population[p]# .get_pairs()
        if pairs != []:
            lin = True
            P = [[] for i in range(m)]
            C = [[] for i in range(m)]
            for (a,b) in pairs:
                P[b].append(a)
                C[a].append(b)
            top = 0
            bot = 0
            roots = []
            leaves = []
            for i in range(m):
                if len(P[i]) > 1 or len(C[i]) > 1 or not(lin):
                    lin = False
                    if len(C[i]) == 0 and len(P[i]) > 0:
                        leaves.append(i)
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                else:
                    if len(C[i]) == 0  and len(P[i]) > 0:
                        leaves.append(i)
                        if len(P[i]) == 1:
                            bot +=1
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                        if len(C[i])== 1:
                            top +=1
            if not(optim_step1):
                U_i = s1_borda_O(C,P,roots,m,lup)
                U.append(U_i)
                other.append(pairs)
            elif top == bot and lin:
                if top == 1:
                    sr_s_i = s1_borda_S(C,roots,m,lup)
                    sr_s.append(sr_s_i)
                else:
                    sr_m_i,cl_m_i = s1_borda_M(C,roots,m,lup)
                    sr_m.append(sr_m_i)
                    cl_m.append(cl_m_i)
            else:
                level = [-1 for i in range(m)]
                current = 0
                q = roots.copy()
                sum = 0
                last =0
                while q != []:
                    temp = len(q)
                    sum += last*temp
                    last = temp
                    for c in q:
                        level[c] = current
                    current += 1
                    q = C[q[0]].copy()
                cont = False
                if sum == (len(pairs)):
                    cont = True
                    for i in range(m):
                        for x in C[i]:
                            if level[x] != level[i] + 1:
                                cont = False
                                break
                            if not(cont):
                                break
                if cont:
                    sr_p_i,rl_p_i = s1_borda_P(C,roots,m,lup)
                    sr_p.append(sr_p_i)
                    rl_p.append(rl_p_i)
                    
                else:
                    U_i = s1_borda_O(C,P,roots,m,lup)
                    U.append(U_i)
                    other.append(pairs)
    if optim_step2:
        minlup = min(lup)
        list_c = []
        for i in range(m):
            if lup[i] == minlup:
                list_c.append(i)
        lenc = len(list_c)
        D = [[] for i in range(lenc)]
        for P in other:
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
    else:
        list_c = []
        D = [[] for i in range(m)]
        for i in range(len(U)):
            U_i = U[i]
            for j_1 in range(m):
                D[j_1].append([])
            for j_1 in range(m):
                for j_2 in U_i[j_1]:
                    D[j_2][i].append(j_1)
                
    
    if verbose:
        print("S : "+str(len(sr_s))+", M : "+str(len(sr_m))+", P : "+str(len(sr_p))+", O : "+str(len(U)))
    return [[U,D],[sr_p,rl_p],[sr_s],[sr_m,cl_m]],list_c,lup




def s3_borda_O(c,w,U,D,m): 
    n = len(U)
    score_w = 0
    score_c = 0
    for i in range(n):
        if c in U[i][w]:
            block_size = intersect(D[i],U[i][w])
            score_c += block_size
        else:
            score_w += m-len(U[i][w])
            score_c += len(D[i])-1
    return score_w,score_c
    
def s3_borda_P(c,w,sr,ranks_l,m): 
    n = len(sr)
    score_w = 0
    score_c = 0
    for i in range(n): 
        if (sr[i][c] >= 0) and (sr[i][c] < sr[i][w]): 
            block_size = ranks_l[i][sr[i][w]] - ranks_l[i][sr[i][c]+1] + 1
            score_c += block_size
        else:
            if sr[i][c] != -1:
                score_c += ranks_l[i][-1]-ranks_l[i][sr[i][c]+1]
            score_w += m-1-ranks_l[i][max(0,sr[i][w])]
 
    return score_w,score_c
    
def s3_borda_S(c,w,sr,m): 
    n = len(sr)
    score_w = 0
    score_c = 0
    for i in range(n): 
        if (sr[i][c] >= 0) and (sr[i][c] < sr[i][w]): 
            block_size = sr[i][w] - sr[i][c] 
        
            score_c += block_size
        else:
            if sr[i][c] != -1:
                score_c += sr[i][-1]-sr[i][c]-1
            score_w += m-1-max(sr[i][w],0)

    return score_w,score_c
    

def s3_borda_M(c,w,sr,cl,m): 
    n = len(sr)
    score_w = 0
    score_c = 0
    for i in range(n): 
        (ssc,spc) = sr[i][c]
        (ssw,spw) = sr[i][w]
        if (spc >= 0) and (spc < spw) and (ssw == ssc): 
            block_size = spw - spc 
            score_c += block_size
        else:
            if ssc != -1:
                score_c += cl[i][ssc]-spc-1
            score_w += m-1-max(spw,0)

    return score_w,score_c
    
def s3_borda(c,w,UD,i,m,verbose=False):
    [U,D] = UD[0]
    [sr_p,rl_p] = UD[1]
    [sr_s] = UD[2]
    [sr_m,cl_m] = UD[3]
    score_w_o,score_c_o = s3_borda_O(c,w,U,D[i],m)
    score_w_p,score_c_p = s3_borda_P(c,w,sr_p,rl_p,m)
    score_w_s,score_c_s = s3_borda_S(c,w,sr_s,m)
    score_w_m,score_c_m = s3_borda_M(c,w,sr_m,cl_m,m)
    score_w = score_w_o + score_w_p + score_w_s + score_w_m
    score_c = score_c_o + score_c_p + score_c_s + score_c_m
    if verbose:
        print("Test "+str(c)+" ("+str(score_c)+") against "+str(w)+" ("+str(score_w)+")")
        
    return score_c >= score_w


def borda(Population,m,verbose=False,optim_step1=True,optim_step2=True):
    UD,list_c,lup = updown_borda(Population,m,verbose=verbose,optim_step1=optim_step1,optim_step2=optim_step2)
    NW = []
    if optim_step2:
        order = np.argsort(lup)
    else:
        list_c = [i for i in range(m)]
        order = [i for i in range(m)]
    for i in range(len(list_c)):
        isaNW = True
        for j in range(m):
            if list_c[i] != order[j]:
                if not(s3_borda(list_c[i],order[j],UD,i,m,verbose=verbose)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(list_c[i])
    return NW

## BORDA MultiThreading

def s1_borda_O_MT(inputs):
    t_begin = time.time()
    (pairs,m) = inputs
    P = [[] for i in range(m)]
    C = [[] for i in range(m)]
    for (a,b) in pairs:
        P[b].append(a)
        C[a].append(b)
    lup = [0 for i in range(m)]
    roots = []
    for i in range(m):
        if len(P[i]) == 0:
            roots.append(i)
    Up = [[i] for i in range(m)]
    isMerge = (np.max([len(c) for c in C]) > 1)
    isSplit = (np.max([len(p) for p in P]) > 1)
    queue = roots.copy()
    parents = [len(p) for p in P]
    
    if isSplit and isMerge:
        while queue != []: 
            u = queue.pop() 
            Up[u] = list(set(Up[u])) 
            lup[u] += len(Up[u])-1
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    else:
        while queue != []: 
            u = queue.pop() 
            lup[u] = len(Up[u])-1
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    t_end = time.time()
    return Up,lup,(t_end-t_begin)



    
  
def updown_borda_MT(Population,m,verbose=False,process=1):
    lup = [0 for i in range(m)]
    U = []
    sr_p = []
    rl_p = []
    sr_s = []
    sr_m = []
    cl_m = []
    n = len(Population)
    pairs_m = [(pair,m) for pair in Population]
    if process == 0:
        out = []
        t_out_before = time.time()
        t_in_before = time.time()
        for i in range(n):
            out.append(s1_borda_O_MT(pairs_m[i]))
        t_in_after = time.time()
        t_out_after = time.time()
    else:
        t_out_before = time.time()
        with Pool(process) as p:
            t_in_before = time.time()
            out = p.map(s1_borda_O_MT,pairs_m)
            t_in_after = time.time()
        t_out_after = time.time()
    lup = [0 for i in range(m)]
    sum_time = 0
    for i in range(n):
        lup_i = out[i][1]
        U.append(out[i][0])
        sum_time += out[i][2]
        for j in range(m):
            lup[j] += lup_i[j]
    if process != 0:
        print((t_out_after-t_out_before),sum_time,(t_out_after-t_out_before)-sum_time/process)
    minlup = min(lup)
    list_c = []
    for i in range(m):
        if lup[i] == minlup:
            list_c.append(i)
    lenc = len(list_c)
    D = [[] for i in range(lenc)]
    for P in Population:
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
    
    if verbose:
        print("S : "+str(len(sr_s))+", M : "+str(len(sr_m))+", P : "+str(len(sr_p))+", O : "+str(len(U)))
    return [[U,D],[sr_p,rl_p],[sr_s],[sr_m,cl_m]],list_c,lup





def borda_MT(Population,m,verbose=False,process=1):
    UD,list_c,lup = updown_borda_MT(Population,m,verbose=verbose,process=process)
    NW = []
    order = np.argsort(lup)
    for i in range(len(list_c)):
        isaNW = True
        for j in range(m):
            if list_c[i] != order[j]:
                if not(s3_borda(list_c[i],order[j],UD,i,m,verbose=verbose)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(list_c[i])
    return NW
    
## BORDA MultiThreading V1

def s1_borda_O_MT2(inputs):
    (pairs,m) = inputs
    U = []
    lup = [0 for i in range(m)]
    for poset in pairs:
        P = [[] for i in range(m)]
        C = [[] for i in range(m)]
        for (a,b) in poset:
            P[b].append(a)
            C[a].append(b)
        roots = []
        for i in range(m):
            if len(P[i]) == 0:
                roots.append(i)
        Up = [[i] for i in range(m)]
        isMerge = (np.max([len(c) for c in C]) > 1)
        isSplit = (np.max([len(p) for p in P]) > 1)
        queue = roots.copy()
        parents = [len(p) for p in P]
        if isSplit and isMerge:
            while queue != []: 
                u = queue.pop() 
                Up[u] = list(set(Up[u])) 
                lup[u] += len(Up[u])-1
                for e in C[u]: 
                    Up[e].extend(Up[u]) 
                    parents[e] -= 1
                    if parents[e] == 0:
                        queue.append(e)
        else:
            while queue != []: 
                u = queue.pop() 
                lup[u] += len(Up[u])-1
                for e in C[u]: 
                    Up[e].extend(Up[u]) 
                    parents[e] -= 1
                    if parents[e] == 0:
                        queue.append(e)
        U.append(Up)
    return U,lup



    
  
def updown_borda_MT2(Population,m,verbose=False,process=1):
    lup = [0 for i in range(m)]
    U = []
    sr_p = []
    rl_p = []
    sr_s = []
    sr_m = []
    cl_m = []
    n = len(Population)
    t1 = time.time()
    if process == 0:
        U,lup= s1_borda_O_MT2((Population,m))
    else:
        pairs = [(Population[(n*i)//process:(n*(i+1))//process],m) for i in range(process)]
        with Pool(process) as p:
            out = p.map(s1_borda_O_MT2,pairs)
        t2 = time.time()
        print(t2-t1)
        for i in range(process):
            lup_i = out[i][1]
            U.extend(out[i][0])
            for j in range(m):
                lup[j] += lup_i[j]
    minlup = min(lup)
    list_c = []
    for i in range(m):
        if lup[i] == minlup:
            list_c.append(i)
    lenc = len(list_c)
    D = [[] for i in range(lenc)]
    for P in Population:
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
    
    if verbose:
        print("S : "+str(len(sr_s))+", M : "+str(len(sr_m))+", P : "+str(len(sr_p))+", O : "+str(len(U)))
    return [[U,D],[sr_p,rl_p],[sr_s],[sr_m,cl_m]],list_c,lup





def borda_MT2(Population,m,verbose=False,process=1):
    UD,list_c,lup = updown_borda_MT2(Population,m,verbose=verbose,process=process)
    NW = []
    order = np.argsort(lup)
    for i in range(len(list_c)):
        isaNW = True
        for j in range(m):
            if list_c[i] != order[j]:
                if not(s3_borda(list_c[i],order[j],UD,i,m,verbose=verbose)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(list_c[i])
    return NW
    
 ## K-APPROVAL
 
 


def s1_kapp_O(C,P,roots,m,lup,k): 
    Up = [[i] for i in range(m)]
    isMerge = (np.max([len(c) for c in C]) > 1)
    isSplit = (np.max([len(p) for p in P]) > 1)
    queue = roots.copy()
    parents = [len(p) for p in P]
    
    if isSplit and isMerge:
        while queue != []: 
            u = queue.pop() 
            Up[u] = list(set(Up[u])) 
            if len(Up[u]) > k:
                lup[u] += 1
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    else:
        while queue != []: 
            u = queue.pop()
            if len(Up[u]) > k:
                lup[u] += 1
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    return Up 



    

def s1_kapp_P(C,roots,m,lup,k): 
    srv = [-1 for i in range(m)]
    ranks = []
    current_rank = 0
    queue = roots.copy()
    ranks.append(0)
    while queue != []:
        j = 0
        for c in queue:
            srv[c] = current_rank
            if current_rank >= k:
                lup[c] += 1
            j += 1
        c0 = queue[0]
        queue = C[c0].copy()
        ranks.append(ranks[-1]+j)
        current_rank += 1
    return srv,ranks



def s1_kapp_S(C,roots,m,lup,k):
    srv = [-1 for i in range(m)]
    j = -1
    current_rank = 0
    j = roots[0]
    while j != -1:
        srv[j] = current_rank
        if current_rank >= k:
            lup[j] += 1
        current_rank +=1
        if len(C[j]) == 0:
            j = -1
        else:
            j = C[j][0]
    srv.append(current_rank)
    return srv
    
        
def s1_kapp_M(C,roots,m,lup,k):
    srv = [(-1,-1) for i in range(m)]
    clv = []
    tops = []
    i = 0
    for i in range(len(roots)):
        current_rank = 0
        j = roots[i]
        while j != -1:
            srv[j] = (i,current_rank)
            if current_rank >= k:
                lup[j] += 1
            current_rank +=1
            if len(C[j]) == 0:
                j = -1
            else:
                j = C[j][0]
        clv.append(current_rank)
    return srv,clv
    
  
def updown_kapp(Population,m,k,verbose=False):
    lup = [0 for i in range(m)]
    sr_p = []
    rl_p = []
    sr_s = []
    sr_m = []
    cl_m = []
    U = []
    
    n = len(Population)
    
    other = []
    
    for p in range(n):
        pairs = Population[p] #.get_pairs()
        if pairs != []:
            lin = True
            P = [[] for i in range(m)]
            C = [[] for i in range(m)]
            for (a,b) in pairs:
                P[b].append(a)
                C[a].append(b)
            top = 0
            bot = 0
            roots = []
            leaves = []
            for i in range(m):
                if len(P[i]) > 1 or len(C[i]) > 1 or not(lin):
                    lin = False
                    if len(C[i]) == 0 and len(P[i]) > 0:
                        leaves.append(i)
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                else:
                    if len(C[i]) == 0  and len(P[i]) > 0:
                        leaves.append(i)
                        if len(P[i]) == 1:
                            bot +=1
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                        if len(C[i])== 1:
                            top +=1
            if top == bot and lin:
                if top == 1:
                    sr_s_i = s1_kapp_S(C,roots,m,lup,k)
                    sr_s.append(sr_s_i)
                else:
                    sr_m_i,cl_m_i = s1_kapp_M(C,roots,m,lup,k)
                    sr_m.append(sr_m_i)
                    cl_m.append(cl_m_i)
            else:
                level = [-1 for i in range(m)]
                current = 0
                q = roots.copy()
                sum = 0
                last =0
                while q != []:
                    temp = len(q)
                    sum += last*temp
                    last = temp
                    for c in q:
                        level[c] = current
                    current += 1
                    q = C[q[0]].copy()
                cont = False
                if sum == (len(pairs)):
                    cont = True
                    for i in range(m):
                        for x in C[i]:
                            if level[x] != level[i] + 1:
                                cont = False
                                break
                            if not(cont):
                                break
                if cont:
                    sr_p_i,rl_p_i = s1_kapp_P(C,roots,m,lup,k)
                    sr_p.append(sr_p_i)
                    rl_p.append(rl_p_i)
                    
                else:
                    U_i = s1_kapp_O(C,P,roots,m,lup,k)
                    U.append(U_i)
                    other.append(pairs)
    minlup = min(lup)
    list_c = []
    for i in range(m):
        if lup[i] == minlup:
            list_c.append(i)
    lenc = len(list_c)
    D = [[] for i in range(lenc)]
    for P in other:
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
    
    if verbose:
        print("S : "+str(len(sr_s))+", M : "+str(len(sr_m))+", P : "+str(len(sr_p))+", O : "+str(len(U)))
    return [[U,D],[sr_p,rl_p],[sr_s],[sr_m,cl_m]],list_c,lup




def s3_kapp_O(k,c,w,U,D,m): 
    n = len(U)
    score_w = 0
    score_c = 0
    for i in range(n):
        minpos_w = len(U[i][w])
        maxpos_c = m-len(D[i])+1
        if c in U[i][w]:
            if minpos_w <= k:
                score_c += 1
                score_w += 1
            elif maxpos_c <= k:
                score_c += 1
        else:
            if maxpos_c <= k:
                score_c += 1
            if minpos_w <= k:
                score_w += 1
    return score_w,score_c
    
def s3_kapp_P(k,c,w,sr,ranks_l,m): 
    n = len(sr)
    score_w = 0
    score_c = 0
    for i in range(n): 
        minpos_w = ranks_l[i][sr[i][w]] + 1
        maxpos_c = m-(ranks_l[i][-1]-ranks_l[i][sr[i][c]+1])
        
        if sr[i][c] >= 0 and (sr[i][c] < sr[i][w]):
            if minpos_w <= k:
                score_w += 1
                score_c += 1
            elif maxpos_c <= k:
                score_c += 1
        else:
            if maxpos_c <= k and sr[i][c] >=0:
                score_c += 1
            if minpos_w <= k or sr[i][w] < 0:
                score_w += 1
    return score_w,score_c
    
def s3_kapp_S(k,c,w,sr,m): 
    n = len(sr)
    score_w = 0
    score_c = 0
    M_i = np.zeros((m-1,m-1,m-1))
    for i in range(n): 
        minpos_w = max(sr[i][w],0) + 1
        maxpos_c = (m-sr[i][-1])+sr[i][c]+1
        if sr[i][c] >= 0 and sr[i][c] < sr[i][w]:
            if minpos_w <= k:
                score_w += 1
                score_c += 1
            elif maxpos_c <= k:
                score_c += 1
        else:
            if maxpos_c <= k and sr[i][c] >= 0:
                score_c += 1
            if minpos_w <= k or sr[i][w] < 0:
                score_w += 1
    return score_w,score_c
    

def s3_kapp_M(k,c,w,sr,cl,m): 
    n = len(sr)
    score_w = 0
    score_c = 0
    for i in range(n): 
        (ssc,spc) = sr[i][c]
        (ssw,spw) = sr[i][w]
        minpos_w = spw + 1
        maxpos_c = m-(cl[i][ssc] - spc)
      
        if spc > 0 and (spc < spw and ssw == ssc):
            if minpos_w <= k:
                score_w += 1
                score_c += 1
            elif maxpos_c <= k:
                score_c += 1
        else:
            if spc != -1 and maxpos_c <= k:
                score_c += 1
            if spw == -1 or minpos_w <= k:
                score_w += 1
    return score_w,score_c
    
def s3_kapp(k,c,w,UD,i,m,verbose=False):
    [U,D] = UD[0]
    [sr_p,rl_p] = UD[1]
    [sr_s] = UD[2]
    [sr_m,cl_m] = UD[3]
    score_w_o,score_c_o = s3_kapp_O(k,c,w,U,D[i],m)
    score_w_p,score_c_p = s3_kapp_P(k,c,w,sr_p,rl_p,m)
    score_w_s,score_c_s = s3_kapp_S(k,c,w,sr_s,m)
    score_w_m,score_c_m = s3_kapp_M(k,c,w,sr_m,cl_m,m)
    score_w = score_w_o + score_w_p + score_w_s + score_w_m
    score_c = score_c_o + score_c_p + score_c_s + score_c_m
    if verbose:
        print("Test "+str(c)+" ("+str(score_c)+") against "+str(w)+" ("+str(score_w)+")")
    return score_c >= score_w


def kapp(Population,m,k,verbose=False):
    UD,list_c,lup = updown_kapp(Population,m,k,verbose=verbose)
    NW = []
    order = np.argsort(lup)
    for i in range(len(list_c)):
        isaNW = True
        for j in range(m):
            if list_c[i] != order[j]:
                if not(s3_kapp(k,list_c[i],order[j],UD,i,m,verbose=verbose)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(list_c[i])
    return NW
    
 
def plurality(Population,m,verbose=False):
    return kapp(Population,m,1,verbose=verbose)
    
def veto(Population,m,verbose=False):
    return kapp(Population,m,m-1,verbose=verbose)
    
    
## ANY POSITIONAL SCORING RULE

def precompute_score(rule,m):
    M = np.zeros((m-1,m-1,m-1)) 
    for i in range(m):
        for j in range(i+1,m):
            M[i][j-1][j-i-1] = rule[i] - rule[j]
    for k in range(1,m):
        for i in range(m):
            for j in range(i+k+1,m):
                M[i][j-1][j-i-k-1] = min(rule[i+k]-rule[j],M[i][j-2][j-i-k-1])
    return M
    

def s1_psr_O(C,P,roots,m,lup,rule): 
    Up = [[i] for i in range(m)]
    isMerge = (np.max([len(c) for c in C]) > 1)
    isSplit = (np.max([len(p) for p in P]) > 1)
    queue = roots.copy()
    parents = [len(p) for p in P]
    
    if isSplit and isMerge:
        while queue != []: 
            u = queue.pop() 
            Up[u] = list(set(Up[u])) 
            lup[u] += rule[0]-rule[len(Up[u])-1]
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    else:
        while queue != []: 
            u = queue.pop()
            lup[u] += rule[0]-rule[len(Up[u])-1]
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    return Up 



    

def s1_psr_P(C,roots,m,lup,rule): 
    srv = [-1 for i in range(m)]
    ranks = []
    current_rank = 0
    queue = roots.copy()
    ranks.append(0)
    while queue != []:
        j = 0
        for c in queue:
            srv[c] = current_rank
            lup[c] += (rule[0]-rule[current_rank])
            j += 1
        c0 = queue[0]
        queue = C[c0].copy()
        ranks.append(ranks[-1]+j)
        current_rank += 1
    return srv,ranks



def s1_psr_S(C,roots,m,lup,rule):
    srv = [-1 for i in range(m)]
    j = -1
    current_rank = 0
    j = roots[0]
    while j != -1:
        srv[j] = current_rank
        lup[j] += rule[0] - rule[current_rank]
        current_rank +=1
        if len(C[j]) == 0:
            j = -1
        else:
            j = C[j][0]
    srv.append(current_rank)
    return srv
    
        
def s1_psr_M(C,roots,m,lup,rule):
    srv = [(-1,-1) for i in range(m)]
    clv = []
    tops = []
    i = 0
    for i in range(len(roots)):
        current_rank = 0
        j = roots[i]
        while j != -1:
            srv[j] = (i,current_rank)
            lup[j] += rule[0] - rule[current_rank]
            current_rank +=1
            if len(C[j]) == 0:
                j = -1
            else:
                j = C[j][0]
        clv.append(current_rank)
    return srv,clv
    
  
def updown_psr(Population,m,rule,verbose=False):
    lup = [0 for i in range(m)]
    sr_p = []
    rl_p = []
    sr_s = []
    sr_m = []
    cl_m = []
    U = []
    D = []
    n = len(Population)
    
   
    
    for p in range(n):
        pairs = Population[p] #.get_pairs()
        if pairs != []:
            lin = True
            P = [[] for i in range(m)]
            C = [[] for i in range(m)]
            for (a,b) in pairs:
                P[b].append(a)
                C[a].append(b)
            top = 0
            bot = 0
            roots = []
            leaves = []
            for i in range(m):
                if len(P[i]) > 1 or len(C[i]) > 1 or not(lin):
                    lin = False
                    if len(C[i]) == 0 and len(P[i]) > 0:
                        leaves.append(i)
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                else:
                    if len(C[i]) == 0  and len(P[i]) > 0:
                        leaves.append(i)
                        if len(P[i]) == 1:
                            bot +=1
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                        if len(C[i])== 1:
                            top +=1
            if top == bot and lin:
                if top == 1:
                    sr_s_i = s1_psr_S(C,roots,m,lup,rule)
                    sr_s.append(sr_s_i)
                else:
                    sr_m_i,cl_m_i = s1_psr_M(C,roots,m,lup,rule)
                    sr_m.append(sr_m_i)
                    cl_m.append(cl_m_i)
            else:
                level = [-1 for i in range(m)]
                current = 0
                q = roots.copy()
                sum = 0
                last =0
                while q != []:
                    temp = len(q)
                    sum += last*temp
                    last = temp
                    for c in q:
                        level[c] = current
                    current += 1
                    q = C[q[0]].copy()
                cont = False
                if sum == (len(pairs)):
                    cont = True
                    for i in range(m):
                        for x in C[i]:
                            if level[x] != level[i] + 1:
                                cont = False
                                break
                            if not(cont):
                                break
                if cont:
                    sr_p_i,rl_p_i = s1_psr_P(C,roots,m,lup,rule)
                    sr_p.append(sr_p_i)
                    rl_p.append(rl_p_i)
                    
                else:
                    U_i = s1_psr_O(C,P,roots,m,lup,rule)
                    U.append(U_i)
                    D_i = [[] for i in range(m)]
                    for elem_down in range(m):
                            for elem_up in U_i[elem_down]:
                                D_i[elem_up].append(elem_down)
                    D.append(D_i)
    minlup = min(lup)
    list_c = []
    for i in range(m):
        if lup[i] == minlup:
            list_c.append(i)
    lenc = len(list_c)
    
    if verbose:
        print("S : "+str(len(sr_s))+", M : "+str(len(sr_m))+", P : "+str(len(sr_p))+", O : "+str(len(U)))
    return [[U,D],[sr_p,rl_p],[sr_s],[sr_m,cl_m]],list_c,lup




def s3_psr_O(rule,M,c,w,U,D,m,optim_prepross=True): 
    n = len(U)
    score_w = 0
    score_c = 0
    for i in range(n):
        if c in U[i][w]:
            block_size = intersect(D[i][c],U[i][w])
            if block_size == 0:
                print("wtf")
            if optim_prepross:
                score_c += M[max(len(U[i][c]),len(U[i][w])-block_size)-1,min(m-len(D[i][w]),m-len(D[i][c])+block_size)-1,block_size-1]
            else:
                minim = rule[len(U[i][c])-1] - rule[len(U[i][c])-1+block_size]
                for i in range(len(U[i][c]),m-len(D[i][c])-block_size+1):
                    if rule[i] - rule[i+block_size] < minim:
                        minim = rule[i] - rule[i+block_size]
                score_c += minim
        else:
            score_w += rule[len(U[i][w])-1]
            score_c += rule[m - len(D[i][c])]

    return score_w,score_c
    
def s3_psr_P(rule,M,c,w,sr,ranks_l,m,optim_prepross=True): 
    n = len(sr)
    score_w = 0
    score_c = 0
    for i in range(n): 
        if (sr[i][c] >= 0) and (sr[i][c] < sr[i][w]):
            
            block_size = ranks_l[i][sr[i][w]] - ranks_l[i][sr[i][c]+1] + 1 
            score_c += M[ranks_l[i][sr[i][c]],m-(ranks_l[i][-1]-ranks_l[i][sr[i][w]+1] + 1)-1,block_size-1]
        else:
            score_w += rule[ranks_l[i][max(0,sr[i][w])]]
            if sr[i][c] == -1:
                score_c += rule[-1]
            else:
                score_c += rule[m - (ranks_l[i][-1]-ranks_l[i][sr[i][c]+1] + 1)]

    return score_w,score_c
    
def s3_psr_S(rule,M,c,w,sr,m,optim_prepross=True): 
    n = len(sr)
    score_w = 0
    score_c = 0
    M_i = np.zeros((m-1,m-1,m-1))
    for i in range(n): 
        if sr[i][c] >= 0 and sr[i][c] < sr[i][w]:
            block_size = sr[i][w] - sr[i][c]
            score_c += M[sr[i][c],m-1-(sr[i][-1]-sr[i][w]),block_size-1]
        else:
            score_w += rule[max(sr[i][w],0)]
            if sr[i][c] == -1:
                score_c += rule[-1]
            else:
                score_c += rule[(m - sr[i][-1])+sr[i][c]]
    return score_w,score_c
    

def s3_psr_M(rule,M,c,w,sr,cl,m,optim_prepross=True): 
    n = len(sr)
    score_w = 0
    score_c = 0
    for i in range(n): 
        (ssc,spc) = sr[i][c]
        (ssw,spw) = sr[i][w]
        if spc >=0 and (spc < spw) and ssw == ssc:
            block_size = spw-spc
            score_c += M[spc-1,m-1-(cl[i][ssw]-spw),block_size-1]
        else:
            score_w += rule[max(spw,0)]
            if spc == -1:
                score_c += rule[-1]
            else:
                score_c += rule[m - (cl[i][ssc]-spc)]
    return score_w,score_c
    
def s3_psr(rule,M,c,w,UD,m,verbose=False,optim_prepross=True):
    [U,D] = UD[0]
    [sr_p,rl_p] = UD[1]
    [sr_s] = UD[2]
    [sr_m,cl_m] = UD[3]
    score_w_o,score_c_o = s3_psr_O(rule,M,c,w,U,D,m,optim_prepross=optim_prepross)
    score_w_p,score_c_p = s3_psr_P(rule,M,c,w,sr_p,rl_p,m,optim_prepross=optim_prepross)
    score_w_s,score_c_s = s3_psr_S(rule,M,c,w,sr_s,m,optim_prepross=optim_prepross)
    score_w_m,score_c_m = s3_psr_M(rule,M,c,w,sr_m,cl_m,m,optim_prepross=optim_prepross)
    score_w = score_w_o + score_w_p + score_w_s + score_w_m
    score_c = score_c_o + score_c_p + score_c_s + score_c_m
    if verbose:
        print("Test "+str(c)+" ("+str(score_c)+") against "+str(w)+" ("+str(score_w)+")")
    return score_c >= score_w


def positional_scoring_rule(Population,m,rule,verbose=False,optim_prepross=True):
    UD,list_c,lup = updown_psr(Population,m,rule,verbose=verbose)
    if optim_prepross:
        M = precompute_score(rule,m)
    else:
        M = []
    NW = []
    order = np.argsort(lup)
    for i in range(len(list_c)):
        isaNW = True
        for j in range(m):
            if list_c[i] != order[j]:
                if not(s3_psr(rule,M,list_c[i],order[j],UD,m,verbose=verbose,optim_prepross=optim_prepross)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(list_c[i])
    return NW
    
    
### Bucklin

def s1_bucklin_O(C,P,roots,m,lup,rule): 
    Up = [[i] for i in range(m)]
    isMerge = (np.max([len(c) for c in C]) > 1)
    isSplit = (np.max([len(p) for p in P]) > 1)
    queue = roots.copy()
    parents = [len(p) for p in P]
    
    if isSplit and isMerge:
        while queue != []: 
            u = queue.pop() 
            Up[u] = list(set(Up[u])) 
            lup[u] += rule[0]-rule[len(Up[u])-1]
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    else:
        while queue != []: 
            u = queue.pop()
            lup[u] += rule[0]-rule[len(Up[u])-1]
            for e in C[u]: 
                Up[e].extend(Up[u]) 
                parents[e] -= 1
                if parents[e] == 0:
                    queue.append(e)
    return Up 



    

def s1_bucklin_P(C,roots,m,lup,rule): 
    srv = [-1 for i in range(m)]
    ranks = []
    current_rank = 0
    queue = roots.copy()
    ranks.append(0)
    while queue != []:
        j = 0
        for c in queue:
            srv[c] = current_rank
            lup[c] += (rule[0]-rule[current_rank])
            j += 1
        c0 = queue[0]
        queue = C[c0].copy()
        ranks.append(ranks[-1]+j)
        current_rank += 1
    return srv,ranks



def s1_bucklin_S(C,roots,m,lup,rule):
    srv = [-1 for i in range(m)]
    j = -1
    current_rank = 0
    j = roots[0]
    while j != -1:
        srv[j] = current_rank
        lup[j] += rule[0] - rule[current_rank]
        current_rank +=1
        if len(C[j]) == 0:
            j = -1
        else:
            j = C[j][0]
    srv.append(current_rank)
    return srv
    
        
def s1_bucklin_M(C,roots,m,lup,rule):
    srv = [(-1,-1) for i in range(m)]
    clv = []
    tops = []
    i = 0
    for i in range(len(roots)):
        current_rank = 0
        j = roots[i]
        while j != -1:
            srv[j] = (i,current_rank)
            lup[j] += rule[0] - rule[current_rank]
            current_rank +=1
            if len(C[j]) == 0:
                j = -1
            else:
                j = C[j][0]
        clv.append(current_rank)
    return srv,clv
    
  
def updown_bucklin(Population,m,rule,verbose=False):
    lup = [0 for i in range(m)]
    sr_p = []
    rl_p = []
    sr_s = []
    sr_m = []
    cl_m = []
    U = []
    D = []
    n = len(Population)
    
   
    
    for p in range(n):
        pairs = Population[p] #.get_pairs()
        if pairs != []:
            lin = True
            P = [[] for i in range(m)]
            C = [[] for i in range(m)]
            for (a,b) in pairs:
                P[b].append(a)
                C[a].append(b)
            top = 0
            bot = 0
            roots = []
            leaves = []
            for i in range(m):
                if len(P[i]) > 1 or len(C[i]) > 1 or not(lin):
                    lin = False
                    if len(C[i]) == 0 and len(P[i]) > 0:
                        leaves.append(i)
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                else:
                    if len(C[i]) == 0  and len(P[i]) > 0:
                        leaves.append(i)
                        if len(P[i]) == 1:
                            bot +=1
                    elif len(P[i]) == 0  and len(C[i]) > 0:
                        roots.append(i)
                        if len(C[i])== 1:
                            top +=1
            if top == bot and lin:
                if top == 1:
                    sr_s_i = s1_bucklin_S(C,roots,m,lup,rule)
                    sr_s.append(sr_s_i)
                else:
                    sr_m_i,cl_m_i = s1_bucklin_M(C,roots,m,lup,rule)
                    sr_m.append(sr_m_i)
                    cl_m.append(cl_m_i)
            else:
                level = [-1 for i in range(m)]
                current = 0
                q = roots.copy()
                sum = 0
                last =0
                while q != []:
                    temp = len(q)
                    sum += last*temp
                    last = temp
                    for c in q:
                        level[c] = current
                    current += 1
                    q = C[q[0]].copy()
                cont = False
                if sum == (len(pairs)):
                    cont = True
                    for i in range(m):
                        for x in C[i]:
                            if level[x] != level[i] + 1:
                                cont = False
                                break
                            if not(cont):
                                break
                if cont:
                    sr_p_i,rl_p_i = s1_bucklin_P(C,roots,m,lup,rule)
                    sr_p.append(sr_p_i)
                    rl_p.append(rl_p_i)
                    
                else:
                    U_i = s1_bucklin_O(C,P,roots,m,lup,rule)
                    U.append(U_i)
                    D_i = [[] for i in range(m)]
                    for elem_down in range(m):
                            for elem_up in U_i[elem_down]:
                                D_i[elem_up].append(elem_down)
                    D.append(D_i)
    if verbose:
        print("S : "+str(len(sr_s))+", M : "+str(len(sr_m))+", P : "+str(len(sr_p))+", O : "+str(len(U)))
    return [[U,D],[sr_p,rl_p],[sr_s],[sr_m,cl_m]],lup





def s3_bucklin_O(rule,M,c,w,U,D,m): 
    n = len(U)
    score_w = [0 for i in range(m)]
    score_c = [0 for i in range(m)]
    for i in range(n):
        if c in U[i][w]:
            block_size = intersect(D[i][c],U[i][w])
            if block_size == 0:
                print("wtf")
           
            score_c += M[max(len(U[i][c]),len(U[i][w])-block_size)-1,min(m-len(D[i][w]),m-len(D[i][c])+block_size)-1,block_size-1]
        else:
            score_w += rule[len(U[i][w])-1]
            score_c += rule[m - len(D[i][c])]

    return score_w,score_c
    
def s3_bucklin_P(rule,M,c,w,sr,ranks_l,m): 
    n = len(sr)
    score_w = [0 for i in range(m)]
    score_c = [0 for i in range(m)]
    for i in range(n): 
        if (sr[i][c] >= 0) and (sr[i][c] < sr[i][w]):
            
            block_size = ranks_l[i][sr[i][w]] - ranks_l[i][sr[i][c]+1] + 1 
            score_c += M[ranks_l[i][sr[i][c]],m-(ranks_l[i][-1]-ranks_l[i][sr[i][w]+1] + 1)-1,block_size-1]
        else:
            score_w += rule[ranks_l[i][max(0,sr[i][w])]]
            if sr[i][c] == -1:
                score_c += rule[-1]
            else:
                score_c += rule[m - (ranks_l[i][-1]-ranks_l[i][sr[i][c]+1] + 1)]

    return score_w,score_c
    
def s3_bucklin_S(rule,M,c,w,sr,m): 
    n = len(sr)
    score_w = [0 for i in range(m)]
    score_c = [0 for i in range(m)]
    M_i = np.zeros((m-1,m-1,m-1))
    for i in range(n): 
        if sr[i][c] >= 0 and sr[i][c] < sr[i][w]:
            block_size = sr[i][w] - sr[i][c]
            score_c += M[sr[i][c],m-1-(sr[i][-1]-sr[i][w]),block_size-1]
        else:
            score_w += rule[max(sr[i][w],0)]
            if sr[i][c] == -1:
                score_c += rule[-1]
            else:
                score_c += rule[(m - sr[i][-1])+sr[i][c]]
    return score_w,score_c
    

def s3_bucklin_M(rule,M,c,w,sr,cl,m): 
    n = len(sr)
    score_w = [0 for i in range(m)]
    score_c = [0 for i in range(m)]
    for i in range(n): 
        (ssc,spc) = sr[i][c]
        (ssw,spw) = sr[i][w]
        if spc >=0 and (spc < spw) and ssw == ssc:
            block_size = spw-spc
            score_c += M[spc-1,m-1-(cl[i][ssw]-spw),block_size-1]
        else:
            score_w += rule[max(spw,0)]
            if spc == -1:
                score_c += rule[-1]
            else:
                score_c += rule[m - (cl[i][ssc]-spc)]
    return score_w,score_c
    
def s3_bucklin(rule,M,c,w,UD,m,verbose=False):
    [U,D] = UD[0]
    [sr_p,rl_p] = UD[1]
    [sr_s] = UD[2]
    [sr_m,cl_m] = UD[3]
    score_w_o,score_c_o = s3_bucklin_O(c,w,U,D,m)
    score_w_p,score_c_p = s3_bucklin_P(c,w,sr_p,rl_p,m)
    score_w_s,score_c_s = s3_bucklin_S(c,w,sr_s,m)
    score_w_m,score_c_m = s3_bucklin_M(c,w,sr_m,cl_m,m)
    score_w = score_w_o + score_w_p + score_w_s + score_w_m
    score_c = score_c_o + score_c_p + score_c_s + score_c_m
    if verbose:
        print("Test "+str(c)+" ("+str(score_c)+") against "+str(w)+" ("+str(score_w)+")")
    return score_c >= score_w



def bucklin(Population,m,verbose=False):
    UD,lup = updown_bucklin(Population,m,rule,verbose=verbose)
    NW = []
    order = np.argsort(lup)
    eliminated = [0 for i in range(m)]
    for i in range(m):
        isaNW = True
        if eliminated[i] == 0:
            for j in range(m):
                if i != j:
                    if not(s3_bucklin(order[i],order[j],UD,m,verbose=verbose)):
                        isaNW = False
                        eliminated[i] = 1
                        break
                    else:
                        eliminated[j] = 1
            if isaNW:
                NW.append(order[i])
    return NW
    
