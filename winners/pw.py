
import numpy as np
import maxflow as mf


## Plurality

def get_roots(population,m):
    n = len(population)
    roots_list = []
    for i in range(n):
        roots_i = [1]*m
        v = population[i]
        for x in v:
            (a,b) =x
            roots_i[b] = 0
        roots_list.append(roots_i)
    return roots_list

def aggregate_plurality(roots_list,m):
    dico_roots = dict()
    tab_roots = []
    count_roots = []
    i = 0
    for roots_i in roots_list:
        if str(roots_i) in dico_roots.keys():
            count_roots[dico_roots[str(roots_i)]] += 1
        else:
            count_roots.append(1)
            tab_roots.append(roots_i)
            dico_roots[str(roots_i)] = i
            i+=1
    return tab_roots,count_roots
        
    
def build_graph_plurality(graph,score_c,count_roots,roots_list,m,c):
    P1 = len(count_roots)
    size = P1 + m - 1
    nodes = graph.add_nodes(size)
    for i in range(P1):
        graph.add_tedge(i,count_roots[i],0)
        for j in range(m-1):
            if roots_list[i][j] >0:
                graph.add_edge(i,P1+j,count_roots[i],0)
    for i in range(m-1):
        graph.add_tedge(P1+i,0,score_c)
    return size

def try_approx_plurality(score_c,count_roots,roots_list,m):
    n = len(count_roots)
    init = [score_c for i in range(m-1)]
    for i in range(n):
        score_tab = list(np.argsort(init))
        score_tab.reverse()
        left = count_roots[i]
        for k in range(m-1):
            j = score_tab[k]
            if roots_list[i][j] == 1:
                suppr = min(init[j],left)
                left -= suppr
                init[j] -= suppr
        if left > 0:
            return False
    return True
    
        
        

def possible_winner_plurality(roots_list,count_roots,m,c,verbose=False):
    roots_list_without_c = []
    score_c = 0
    count_roots_without_c = []
    maxflow_wanted = 0
    for i in range(len(roots_list)):
        if roots_list[i][c] == 1:
            score_c += count_roots[i]
        else:
            l = roots_list[i].copy()
            l.pop(c)
            maxflow_wanted += count_roots[i]
            count_roots_without_c.append(count_roots[i])
            roots_list_without_c.append(l)
    if score_c > maxflow_wanted:
        if verbose:
            print(str(c)+" : Default winner ("+str(score_c)+")")
        return True
    if score_c < (score_c+maxflow_wanted)/m:
        if verbose:
            print(str(c)+" : Default loser ("+str(score_c)+")")
        return False
    if try_approx_plurality(score_c,count_roots_without_c,roots_list_without_c,m):
        if verbose:
            print(str(c)+" : Winner with approx")
        return True
    graph = mf.GraphInt()
    size = build_graph_plurality(graph,score_c,count_roots_without_c,roots_list_without_c,m,c)
    maxflow = graph.maxflow()
    if maxflow >= maxflow_wanted:
        if verbose:
            print(str(c)+" : Winner with graph")
        return True
    else:
        if verbose:
            print(str(c)+" : Loser ("+str(maxflow)+"/"+str(maxflow_wanted)+")")
        return False


def plurality(population,m,verbose=False):
    roots_list_net = get_roots(population,m)
    roots_list,count_roots = aggregate_plurality(roots_list_net,m)
    winners = []
    for c in range(m):
        if possible_winner_plurality(roots_list,count_roots,m,c,verbose=verbose):
            winners.append(c)
    return winners
    

## Veto



def get_leaves(population,m):
    n = len(population)
    leaves_list = []
    for i in range(n):
        leaves_i = [1]*m
        v = population[i]
        for x in v:
            (a,b) =x
            leaves_i[a] = 0
        leaves_list.append(leaves_i)
    return leaves_list

def aggregate_veto(leaves_list,m):
    dico_leaves = dict()
    tab_leaves = []
    count_leaves = []
    i = 0
    for leaves_i in leaves_list:
        if str(leaves_i) in dico_leaves.keys():
            count_leaves[dico_leaves[str(leaves_i)]] += 1
        else:
            count_leaves.append(1)
            tab_leaves.append(leaves_i)
            dico_leaves[str(leaves_i)] = i
            i+=1
    return tab_leaves,count_leaves
        
    
def build_graph_veto(graph,zero_c,count_leaves,leaves_list,m,c):
    P1 = len(count_leaves)
    size = P1 + m - 1
    nodes = graph.add_nodes(size)
    for i in range(P1):
        graph.add_tedge(i,count_leaves[i],0)
        for j in range(m-1):
            if leaves_list[i][j] >0:
                graph.add_edge(i,P1+j,count_leaves[i],0)
    for i in range(m-1):
        graph.add_tedge(P1+i,0,zero_c)
    return size

def try_approx_veto(zero_c,count_leaves,leaves_list,m):
    n = len(count_leaves)
    init = [zero_c for i in range(m-1)]
    for i in range(n):
        score_tab = list(np.argsort(init))
        score_tab.reverse()
        if np.max(init) <= 0:
            return True
        left = count_leaves[i]
        for k in range(m-1):
            j = score_tab[k]
            if leaves_list[i][j] == 1:
                suppr = min(init[j],left)
                left -= suppr
                init[j] -= suppr
    if np.max(init) <=0:
        return True
    else:
        return False
    
        
        

def possible_winner_veto(leaves_list,count_leaves,m,c,n,verbose=False):
    leaves_list_without_c = []
    zero_c = 0
    count_leaves_without_c = []
    maxflow_wanted = 0
    for i in range(len(leaves_list)):
        if leaves_list[i][c] == 1 and np.sum(leaves_list[i]) == 1:
            zero_c += count_leaves[i]
        else:
            l = leaves_list[i].copy()
            l.pop(c)
            count_leaves_without_c.append(count_leaves[i])
            leaves_list_without_c.append(l)
    maxflow_wanted = zero_c*(m-1)
    if zero_c < (n-zero_c)/(m-1):
        if verbose:
            print(str(c)+" : Default winner ("+str(zero_c)+")")
        return True
    if zero_c > n/2:
        if verbose:
            print(str(c)+" : Default loser ("+str(zero_c)+")")
        return False
    if try_approx_veto(zero_c,count_leaves_without_c,leaves_list_without_c,m):
        if verbose:
            print(str(c)+" : Winner with approx")
        return True
    graph = mf.GraphInt()
    size = build_graph_veto(graph,zero_c,count_leaves_without_c,leaves_list_without_c,m,c)
    maxflow = graph.maxflow()
    if maxflow >= maxflow_wanted:
        if verbose:
            print(str(c)+" : Winner with graph")
        return True
    else:
        if verbose:
            print(str(c)+" : Loser ("+str(maxflow)+"/"+str(maxflow_wanted)+")")
        return False


def veto(population,m,verbose=False):
    leaves_list_net = get_leaves(population,m)
    leaves_list,count_leaves = aggregate_veto(leaves_list_net,m)
    winners = []
    for c in range(m):
        if possible_winner_veto(leaves_list,count_leaves,m,c,len(population),verbose=verbose):
            winners.append(c)
    return winners
    
    

## Approx borda

from . import nw

##

def max_rank_approx(U,D,m,c,rule,danger=[],verbose=False):
    n = len(U)
    index_order = [i for i in range(n)]
    np.random.shuffle(index_order)
    score = np.zeros(m)
    for i in range(n):
        new_index = index_order[i]
        given = [0 for i in range(m)]
        
        Up_c = []
        Down_c = []
        free_list = []
        free_score = []
        for j in range(m):
            if j!=c:
                if j in U[new_index][c]:
                    Up_c.append(j)
                elif j in D[new_index][c]:
                    Down_c.append(j)
                else:
                    free_list.append(j)
                    free_score.append(score[j])
        argscore_free = np.argsort(free_score)
        argscore_free = argscore_free[::-1]
        j_incr = len(free_list)
        for j in range(len(danger)):
            w = danger[j]

            if w not in Down_c and w in free_list:
                place_needed = 0
                for child_w in D[new_index][w]:
                    if child_w not in Down_c and child_w != c:
                        place_needed += 1
                if j_incr -place_needed >=0 :
                    for child_w in D[new_index][w]:
                        if child_w not in Down_c and child_w != c:
                            Down_c.append(child_w)
                    j_incr -= place_needed
        rank_c = len(U[new_index][c])-1
        free_cand = 0
        
        while rank_c < (m-len(D[new_index][c])-(len(free_list)-j_incr)) and rule[rank_c] == rule[rank_c+1]:
            rank_c += 1
            free_cand += 1
        score_c = rule[rank_c]
        given[c] = score_c
        j_incr = j_incr-free_cand
        for j in range(len(free_list)):
            
            w = free_list[argscore_free[j]]
            if j_incr == 0 and w not in Down_c:
                Up_c.append(w)
            elif w not in Down_c:
                place_needed = 0
                for child_w in D[new_index][w]:
                    if child_w not in Down_c:
                        place_needed += 1
                if j_incr -place_needed >=0 :
                    for child_w in D[new_index][w]:
                        if child_w not in Down_c:
                            Down_c.append(child_w)
                    j_incr -= place_needed
                else:
                    Up_c.append(free_list[argscore_free[j]])
        danger_child = set()
        for dangerous_candidate in danger:
            danger_child.update(D[new_index][dangerous_candidate])
            
        parents = [[] for i in range(m)]
        child = [[] for i in range(m)]
        parents_count = [0 for i in range(m)]
        child_count = [0 for i in range(m)]
        for j in range(m):
            if j in Up_c:
                for elem in D[new_index][j]:
                    if elem in Up_c and elem != j:
                        parents[elem].append(j)
                        child_count[j] += 1
            elif j in Down_c:
                for elem in U[new_index][j]:
                    if elem in Down_c and elem != j:
                        child[elem].append(j) 
                        parents_count[j] += 1
        orphan_down = []
        for j in Down_c:
            if parents_count[j] == 0:
                orphan_down.append(j)
        orphan_up = []
        for j in Up_c:
            if child_count[j] == 0:
                orphan_up.append(j)
        queue_down = orphan_down
        rank_down = rank_c+1
        while queue_down != []:
            if len(danger) == 0:
                minqueue = score[queue_down[0]]
                candmin = 0
                for j in range(1,len(queue_down)):
                    w = queue_down[j]
                    if score[w] < minqueue:
                        minqueue = score[w]
                        candmin = j
                
            else:
                minqueue = score[queue_down[0]]
                candmin = 0
                danger_not_in = (queue_down[0] in danger)
                for j in range(1,len(queue_down)):
                    w = queue_down[j]
                    if (w not in danger) and (danger_not_in):
                        minqueue = score[w]
                        candmin = j
                        danger_not_in = False
                    elif score[w] < minqueue and ((w not in danger) or danger_not_in):
                        minqueue = score[w]
                        candmin = j
            w_min = queue_down[candmin]
            queue_down.pop(candmin)
            given[w_min] = rule[rank_down]
            rank_down += 1
            for child_w in child[w_min]:
                parents_count[child_w] -= 1
                if parents_count[child_w] == 0:
                    queue_down.append(child_w)
                    
        queue_up = orphan_up
        rank_up = rank_c-1
        while queue_up != []:
            if len(danger) == 0:
                maxqueue = score[queue_up[0]]
                candmax = 0
                for j in range(1,len(queue_up)):
                    w = queue_up[j]
                    if score[w] > maxqueue:
                        maxqueue = score[w]
                        candmax = j
            else:
                maxqueue = score[queue_up[0]]
                candmax = 0
                danger_in = (queue_up[0] in danger)
                danger_down_in =(queue_up[0] in danger_child)
                for j in range(1,len(queue_up)):
                    w = queue_up[j]
                    if (not(danger_in) and (w in danger)):
                        maxqueue = score[w]
                        candmax = j
                        danger_in = True
                    elif (not(danger_down_in) and (w in danger_child)):
                        maxqueue = score[w]
                        candmax = j
                        danger_down_in = True
                    elif (score[w] > maxqueue and (w in danger or (not(danger_in) and (w in danger_child)) or (not(danger_in or danger_down_in)))):
                        maxqueue = score[w]
                        candmax = j
            w_max = queue_up[candmax]
            queue_up.pop(candmax)
            given[w_max] = rule[rank_up]
            rank_up -= 1
            for parents_w in parents[w_max]:
                child_count[parents_w] -= 1
                if child_count[parents_w] == 0:
                    queue_up.append(parents_w)        
        score += given
        # if danger != []:
        #     print(given[danger[0]],given[c],len(D[new_index][danger[0]]),len(U[new_index][danger[0]]),len(U[new_index][c]),danger[0] in U[new_index][c])
    maxscore = np.max(score)
    if score[c] == maxscore:
        return True,0
    if verbose:
        print("The maximum is not "+str(c)+" ("+str(score[c])+") but "+str(np.argmax(score))+" ("+str(np.max(score))+")")
        if len(danger) > 0:
            print("Were minimized : "+str(danger))
    return False  ,np.argmax(score)        

# def max_rank_approx(UD,m,c,rule,verbose=False):
#     n = len(population)
#     score = np.zeros(m)
#     indexs = [0,0,0,0]
#     [U,D] = UD[0]
#     [sr_p,rl_p] = UD[1]
#     [sr_s] = UD[2]
#     [sr_m,cl_m] = UD[3]
#     list_index =[0]*len(UD[0][0])+[1]*len(UD[1][0]) + [2]*len(UD[2][0]) + [3]*len(UD[3][0])
#     np.random.shuffle(list_index)
#     for i in range(n):
#         new_kind = list_index[i]
#         new_index = indexs[new_kind]
#         indexs[new_kind] += 1
#         given = [0 for i in range(m)]
#         if new_kind == 0:
#             rank_c = len(U[new_index][c])-1
#             score_c = rule[rank_c]
#             given[c] = score_c
#             Up_c = []
#             Down_c = []
#             for j in range(m):
#                 if j != c and j in U[new_index][c]:
#                     Up_c.append(j)
#                 elif j != c:
#                     Down_c.append(j)
#             parents = [[] for i in range(m)]
#             child = [[] for i in range(m)]
#             parents_count = [0 for i in range(m)]
#             child_count = [0 for i in range(m)]
#             for j in range(m):
#                 if j in Up_c:
#                     for elem in D[new_index][j]:
#                         if elem in Up_c and elem != j:
#                             parents[elem].append(j)
#                             child_count[j] += 1
#                 elif j in Down_c:
#                     for elem in U[new_index][j]:
#                         if elem in Down_c and elem != j:
#                             child[elem].append(j) 
#                             parents_count[j] += 1
#             orphan_down = []
#             for j in Down_c:
#                 if parents_count[j] == 0:
#                     orphan_down.append(j)
#             orphan_up = []
#             for j in Up_c:
#                 if child_count[j] == 0:
#                     orphan_up.append(j)
#                     
#             queue_down = orphan_down
#             rank_down = rank_c+1
#             while queue_down != []:
#                 minqueue = score[queue_down[0]]
#                 candmin = 0
#                 for j in range(1,len(queue_down)):
#                     w = queue_down[j]
#                     if score[w] < minqueue:
#                         minqueue = score[w]
#                         candmin = j
#                 w_min = queue_down[candmin]
#                 queue_down.pop(candmin)
#                 given[w_min] = rule[rank_down]
#                 rank_down += 1
#                 for child_w in child[w_min]:
#                     parents_count[child_w] -= 1
#                     if parents_count[child_w] == 0:
#                         queue_down.append(child_w)
#                         
#             queue_up = orphan_up
#             rank_up = rank_c-1
#             while queue_up != []:
#                 maxqueue = score[queue_up[0]]
#                 candmax = 0
#                 for j in range(1,len(queue_up)):
#                     w = queue_up[j]
#                     if score[w] < maxqueue:
#                         maxqueue = score[w]
#                         candmax = j
#                 w_max = queue_up[candmax]
#                 queue_up.pop(candmax)
#                 given[w_max] = rule[rank_up]
#                 rank_up -= 1
#                 for parents_w in parents[w_max]:
#                     child_count[parents_w] -= 1
#                     if child_count[parents_w] == 0:
#                         queue_up.append(parents_w)        
#           
#           
#         elif new_kind == 1:
#             sub_rank_c = sr_p[new_index][c]
#             rank_c = rl_p[new_index][sub_rank_c]
#             score_c = rule[rank_c]
#             given[c] = score_c
#             
#             tab_elem = [[] for i in range(len(rl_p[new_index]))]
#             for j in range(m):
#                 if j!=c:
#                     tab_elem[sr_p[new_index][j]].append(j)
#                     
#             rank_up = [min_rank for min_rank in rl_p[new_index][:sub_rank_c]]
#             argscore = np.argsort(score)
#             for index_c in argscore:
#                 sr_new = sr_p[new_index][index_c]
#                 if sr_new != -1 and sr_new < sub_rank_c:
#                     given[index_c] = rule[rank_up[sr_new]]
#                     rank_up[sr_new] += 1
#             
#             
#                         
#             queue_down = orphan_down
#             rank_down = rank_c+1
#             while queue_down != []:
#                 minqueue = score[queue_down[0]]
#                 candmin = 0
#                 for j in range(1,len(queue_down)):
#                     w = queue_down[j]
#                     if score[w] < minqueue:
#                         minqueue = score[w]
#                         candmin = j
#                 w_min = queue_down[candmin]
#                 queue_down.pop(candmin)
#                 given[w_min] = rule[rank_down]
#                 rank_down += 1
#                 for child_w in child[w_min]:
#                     parents_count[child_w] -= 1
#                     if parents_count[child_w] == 0:
#                         queue_down.append(child_w)
#                         
#             queue_up = orphan_up
#             rank_up = rank_c-1
#             while queue_up != []:
#                 maxqueue = score[queue_up[0]]
#                 candmax = 0
#                 for j in range(1,len(queue_up)):
#                     w = queue_up[j]
#                     if score[w] < maxqueue:
#                         maxqueue = score[w]
#                         candmax = j
#                 w_max = queue_up[candmax]
#                 queue_up.pop(candmax)
#                 given[w_max] = rule[rank_up]
#                 rank_up -= 1
#                 for parents_w in parents[w_max]:
#                     child_count[parents_w] -= 1
#                     if child_count[parents_w] == 0:
#                         queue_up.append(parents_w)        
#                 
#             
#             
#         elif new_kind == 2:
#             rank_c = sr_s[new_index][c]
#             score_c = rule[max(rank_c,0)]
#             not_ranked = []
#             not_ranked_score = []
#             ranked = [0]*(sr_s[new_index][-1]-rank_c-1)            
#             given[c] = score_c
#             for j in range(m):
#                 if j != c and sr_s[new_index][j] < rank_c and sr_s[new_index][j] >= 0:
#                     given[j] = rule[sr_s[new_index][j]]
#                 elif j != c and sr_s[new_index][j] == -1:
#                     not_ranked.append(j)
#                     not_ranked_score.append(score[j])
#                 elif j != c:
#                     ranked[sr_s[new_index][j]-rank_c-1] = j
#             remaining = len(not_ranked) + len(ranked)
#             rank_down = max(rank_c,0)+1
#             for i in range(remaining):
#                 if len(not_ranked) != 0:
#                     max_nr = np.min(not_ranked_score)
#                     i_max_nr = np.argmin(not_ranked_score)
#                     if len(ranked) == 0 or max_nr > score[ranked[0]]:
#                         elem = not_ranked.pop(i_max_nr)
#                         not_ranked_score.pop(i_max_nr)
#                     else:
#                         elem = ranked.pop(0)
#                 else:
#                     elem = ranked.pop(0)
#                 given[elem] = rule[rank_down]
#                 rank_down += 1
#                 
#                 
#                 
#         elif new_kind == 3:
#             score_c = rule[sr_m]
#             given[c] = score_c
#         score += given
# 
#     maxscore = np.max(score)
#     if score[c] == maxscore:
#         return True
#     if verbose:
#         print("The maximum is not "+str(c)+" ("+str(score[c])+") but "+str(np.argmax(score))+" ("+str(np.max(score))+")")
#     return False
# 


def approx_positional_scoring_rule(population,m,rule,shuffle=1,type=0,verbose=False):
    M = nw.precompute_score(rule,m)
    lup = [0 for i in range(m)]
    U = []
    D = []
    total_score = np.sum(rule)*len(population)
    maximum_score = np.max(rule)*len(population)

    for i in range(len(population)):
        P = [[] for i in range(m)]
        C = [[] for i in range(m)]
        for (a,b) in population[i]:
            P[b].append(a)
            C[a].append(b)
        roots = []
        for j in range(m):
            if len(P[j]) == 0:
                roots.append(j)
        U_i = nw.s1_psr_O(C,P,roots,m,lup,rule)
        U.append(U_i)
        D_i = [[] for i in range(m)]
        for elem_down in range(m):
            for elem_up in U_i[elem_down]:
                D_i[elem_up].append(elem_down)
        D.append(D_i)
    #[[U,D],[sr_p,rl_p],[sr_s],[sr_m,cl_m]],_,lup = nw.updown_psr(population,m,rule,verbose=verbose)
    
    arglup = np.argsort(lup)
    possible_winners = []
    for i in range(m):
        w = arglup[i]
        max_w = maximum_score-lup[w]
        if max_w < (total_score-max_w)/(m-1): 
            if verbose:
                print(str(w)+" : Default Loser")
        else:
            is_a_PW = True
            for c in possible_winners:
                score_w,score_c = nw.s3_psr_O(rule,M,c,w,U,D,m)
                if verbose:
                    print("Test "+str(c)+" ("+str(score_c)+") against "+str(w)+" ("+str(score_w)+")")
                if score_w < score_c:
                    is_a_PW = False
                    break
            if is_a_PW:
                possible_winners.append(w)
        
    if len(possible_winners) <= 2:
        return possible_winners, []
    else:
        sure_PW = possible_winners[:2]
        possible_PW = []
        for candidate in possible_winners[2:]:
            max_w = maximum_score-lup[candidate]
            if max_w >= (total_score)/2: 
                sure_PW.append(candidate)
                print(str(candidate)+" : Default Winner")
            else:
                verified = False
                for i in range(shuffle):
                    danger = []
                    while (verified == False):
                        found_instance,winner = max_rank_approx(U,D,m,candidate,rule,danger=danger,verbose=verbose)
                        if found_instance:
                            verified = True
                            break
                        else:
                            if winner in danger:
                                break 
                            else:
                                danger.append(winner)
                    if verified:
                        break
                if verified:
                    sure_PW.append(candidate)
                else:
                    possible_PW.append(candidate)
    return sure_PW,possible_PW
            


def approx_borda(population,m,shuffle=1,verbose=False):
    return approx_positional_scoring_rule(population,m,[m-1-i for i in range(m)],shuffle=shuffle,verbose=verbose)
    
def approx_kapproval(population,m,k,shuffle=1,verbose=False):
    return approx_positional_scoring_rule(population,m,[1]*k+[0]*(m-k),shuffle=shuffle,verbose=verbose)
    
    
## Winner set



def list_of_first_set(population,m):
    n = len(population)
    matrix_rank = []
    for i in range(n):
        roots = [1]*m
        v = population[i]
        for x in v:
            (a,b) =x
            if (roots[b]==1):
                roots[b] = 0
        matrix_rank.append(roots)
    return matrix_rank

def aggregate_set(matrix_rank,m):
    dico = dict()
    roots_list = []
    roots_count = []
    i = 0
    for roots in matrix_rank:
        if str(roots) in dico.keys():
            roots_count[dico[str(roots)]] += 1
        else:
            roots_count.append(1)
            roots_list.append(roots)
            dico[str(roots)] = i
            i+=1
    return roots_list,roots_count
        
def score_set(dico,roots_list,roots_count,m,set_cand,n_voters):
    cstr = str(sorted(set_cand))
    if cstr in dico.keys():
        score = dico[cstr]
        return score
    lenc = len(set_cand)
    if lenc == 1:
        dico[cstr] = n_voters
        return n_voters
    else:
        for i in range(lenc):
            cand_i = set_cand[i]
            roots_list_i = []
            roots_count_i = []
            n_voters_i = 0
            for j in range(len(roots_list)):
                ok = True
                k = 0
                while k < lenc and ok:
                    if set_cand[k] != cand_i and roots_list[j][set_cand[k]] == 1:
                        ok = False
                    k += 1
                if not(ok):
                    n_voters_i += roots_count[j]
                    roots_list_i.append(roots_list[j])
                    roots_count_i.append(roots_count[j])
            set_i = set_cand.copy()
            set_i.pop(i)
            comp_n_voters_i = n_voters - n_voters_i
            score_set_i = score_set(dico,roots_list_i,roots_count_i,m,set_i,n_voters_i)
            if comp_n_voters_i >= score_set_i*(lenc-1):
                dico[cstr] = score_set_i
                return score_set_i
            elif comp_n_voters_i + (n_voters_i-score_set_i) >= score_set_i:
                g = mf.GraphInt()
                g.add_nodes(len(roots_count)+lenc)
                for j in range(len(roots_count)):
                    g.add_tedge(j,roots_count[j],0)
                    for k in range(lenc):
                        if roots_list[j][set_cand[k]] == 1:
                            g.add_edge(j,len(roots_count)+k,roots_count[j],0)
                for k in range(lenc):
                    g.add_tedge(len(roots_count)+k,0,score_set_i)
                maxflow = g.maxflow()
                if maxflow == score_set_i*lenc:
                    dico[cstr] = score_set_i
                    return score_set_i
        g = mf.GraphInt()
        g.add_nodes(len(roots_count)+lenc)
        for j in range(len(roots_count)):
            g.add_tedge(j,roots_count[j],0)
            for k in range(lenc):
                if roots_list[j][set_cand[k]] == 1:
                    g.add_edge(j,len(roots_count)+k,roots_count[j],0)
        for k in range(lenc):
            g.add_tedge(len(roots_count)+k,0,n_voters//lenc)
        maxflow = g.maxflow()
        dico[cstr] = n_voters//lenc
        return n_voters//lenc

def build_matrix_set(g,score,roots_list,roots_count,m):
    g.add_nodes(len(roots_count)+m)
    for i in range(len(roots_count)):
        g.add_tedge(i,roots_count[i],0)
        for j in range(m):
            if roots_list[i][j] == 1:
                g.add_edge(i,len(roots_count)+j,roots_count[i],0)
    for j in range(m):
        g.add_tedge(len(roots_count)+j,0,score)
            
    

def possibility_set(dico,roots_list,roots_count,m,set_cand,n_total):
    n_set = 0
    for i in range(len(roots_list)):
        ok = False
        for j in set_cand:
            if roots_list[i][j] == 1:
                ok = True
                break
        if ok:
            n_set += roots_count[i]
    score = score_set(dico,roots_list,roots_count,m,set_cand,n_set)
    #verif
    # gv = mf.GraphInt()
    # gv.add_nodes(len(roots_count)+len(set_cand))
    # for i in range(len(roots_count)):
    #     gv.add_tedge(i,roots_count[i],0)
    #     for k in range(len(set_cand)):
    #         if roots_list[i][set_cand[k]] == 1:
    #             gv.add_edge(i,len(roots_list)+k,roots_count[i],0)
    # for k in range(len(set_cand)):
    #     gv.add_tedge(len(roots_count)+k,0,score)
    # maxflow = gv.maxflow()
    # print("test",maxflow,score,score*len(set_cand))
    
    # if score*len(set_cand) >= (n_total - score): #This is false because maybe one of them cannot have less voters than score+1
    #     return True
    
    if score*len(set_cand) < (n_total*len(set_cand))/m:
        return False
    g = mf.GraphInt()
    maxwanted = build_matrix_set(g,score,roots_list,roots_count,m)
    maxflow = g.maxflow()
    if maxflow == n_total:
        return True
    else:
        return False
    
def plurality_set(population,m,set_cand):
    dico =dict()
    n_total = len(population)
    roots = list_of_first_set(population,m)
    roots_list,roots_count = aggregate_set(roots,m)
    return possibility_set(dico,roots_list,roots_count,m,set_cand,n_total)
    
    
