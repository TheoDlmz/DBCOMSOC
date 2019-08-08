
import numpy as np
import maxflow as mf
import random 
import time
from gurobipy import *

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
        
    
def build_graph_plurality(graph,score_c,count_roots,roots_list,m,c,blocked=[]):
    P1 = len(count_roots)
    size = P1 + m - 1
    nodes = graph.add_nodes(size)
    for i in range(P1):
        graph.add_tedge(i,count_roots[i],0)
        for j in range(m-1):
            if roots_list[i][j] >0:
                graph.add_edge(i,P1+j,count_roots[i],0)
    for i in range(m-1):
        if (i < c and i in blocked) or (i >= c and (i+1) in blocked):
            graph.add_tedge(P1+i,0,score_c-1)
        else:
            graph.add_tedge(P1+i,0,score_c)
    return size

def try_approx_plurality(score_c,c,count_roots,roots_list,m,blocked=[]):
    n = len(count_roots)
    init = [score_c for i in range(m-1)]
    for cand in blocked:
        if cand <c :
            init[cand] -= 1
        elif cand > c:
            init[cand-1] -= 1
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
    
        
        

def possible_winner_plurality(roots_list,count_roots,m,c,verbose=False,blocked=[]):
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
    if try_approx_plurality(score_c,c,count_roots_without_c,roots_list_without_c,m,blocked=blocked):
        if verbose:
            print(str(c)+" : Winner with approx")
        return True
    graph = mf.GraphInt()
    size = build_graph_plurality(graph,score_c,count_roots_without_c,roots_list_without_c,m,c,blocked=blocked)
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
        
    
def build_graph_veto(graph,zero_c,count_leaves,leaves_list,m,c,blocked=[]):
    P1 = len(count_leaves)
    size = P1 + m - 1
    nodes = graph.add_nodes(size)
    for i in range(P1):
        graph.add_tedge(i,count_leaves[i],0)
        for j in range(m-1):
            if leaves_list[i][j] >0:
                graph.add_edge(i,P1+j,count_leaves[i],0)
    for i in range(m-1):
        if (i < c and i in blocked) or (i >= c and (i+1) in blocked):
            graph.add_tedge(P1+i,0,zero_c+1)
        else:
            graph.add_tedge(P1+i,0,zero_c)
    return size

def try_approx_veto(zero_c,c,count_leaves,leaves_list,m,blocked=[]):
    n = len(count_leaves)
    init = [zero_c for i in range(m-1)]
    for cand in blocked:
        if cand < c:
            init[cand] += 1
        else:
            init[cand-1] += 1
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
    
        
        

def possible_winner_veto(leaves_list,count_leaves,m,c,n,verbose=False,blocked=[]):
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
    if zero_c > n/2:
        if verbose:
            print(str(c)+" : Default loser ("+str(zero_c)+")")
        return False
    if try_approx_veto(zero_c,c,count_leaves_without_c,leaves_list_without_c,m,blocked=blocked):
        if verbose:
            print(str(c)+" : Winner with approx")
        return True
    graph = mf.GraphInt()
    size = build_graph_veto(graph,zero_c,count_leaves_without_c,leaves_list_without_c,m,c,blocked=blocked)
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
    
##pw partitioned

def poset2partitioned(poset,m):
    childs = [[] for i in range(m)]
    is_roots = [1]*m
    for (a,b) in poset:
        childs[a].append(b)
        is_roots[b] = 0
    roots = []
    for i in range(m):
        if is_roots[i] == 1:
            roots.append(i)
    
    current_rank = 0
    queue = roots.copy()
    partitioned = []
    while queue != []:
        partitioned.append(queue)
        c0 = queue[0]
        queue = childs[c0].copy()
    return partitioned
    

def remaining_cand(partitioned,k,scores,m):
    seen = len(partitioned[0])
    j = 0
    while seen <= k:
        for cand in partitioned[j]:
            scores[cand] += 1
        j +=1
        seen += len(partitioned[j])
    n_remaining = len(partitioned[j])-(seen-k)
    return n_remaining,partitioned[j]
    
def kapp_partitioned_cand(c,graphs_info,min_scores,m):
    n = len(graphs_info)
    graph = mf.GraphInt()
    graph.add_nodes(n+m)
    score_c = min_scores[c]
    total_needed = 0
    for i in range(n):
        n_remaining,set_remaining = graphs_info[i]
        if c in set_remaining and n_remaining>0:
            score_c += 1
            n_remaining -= 1
        total_needed += n_remaining
        graph.add_tedge(i,n_remaining,0)
        for cand in set_remaining:
            if cand != c:
                graph.add_edge(i,n+cand,1,0)
    for cand in range(m):
        if cand != c:
            graph.add_tedge(n+cand,0,score_c-min_scores[cand])
    maxflow = graph.maxflow()
    return maxflow == total_needed
    


def kapp_partitioned(P,m,k,list=[]):
    min_scores = [0]*m
    graphs_info = []
    for poset in P:
        partitioned = poset2partitioned(poset,m)
        graphs_info.append(remaining_cand(partitioned,k,min_scores,m))
    pw = []
    if list == []:
        list = [i for i in range(m)]
    for cand in list:
        if kapp_partitioned_cand(cand,graphs_info,min_scores,m):
            pw.append(cand)
    return pw

## Approx borda

from . import nw

##

def max_rank_approx(U,D,m,c,rule,danger=[],danger_weights=[],verbose=False,blocked=[]):
    if danger_weights ==[]:
        danger_weights = [1]*len(danger)
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
        danger_child = [[] for i in range(m)]
        for dangerous_candidate in danger:
            for dangerous_child in D[new_index][dangerous_candidate]:
                danger_child[dangerous_child].append(dangerous_candidate)
            
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
                max_danger = len(danger_child[queue_up[0]])
                maxqueue_danger = 0
                for j in range(max_danger):
                    maxqueue_danger += score[danger_child[queue_up[0]][j]]
                    
                for j in range(1,len(queue_up)):
                    w = queue_up[j]
                    if (not(danger_in) and (w in danger)):
                        maxqueue = score[w]
                        candmax = j
                        danger_in = True
                    elif (len(danger_child[w]) > max_danger):
                        maxqueue = score[w]
                        candmax = j
                        max_danger = len(danger_child[w])
                        maxqueue_danger = 0
                        for j2 in range(len(danger_child[w])):
                            #maxqueue_danger = max(m-danger.index(danger_child[w][j2]),maxqueue_danger)
                            maxqueue_danger += score[danger_child[w][j2]]
                    elif (len(danger_child[w]) > 0) and (len(danger_child) == max_danger):
                        maxqueue_danger_w = 0
                        for j2 in range(max_danger):
                            maxqueue_danger_w += score[danger_child[w][j2]]
                            #maxqueue_danger = max(m-danger.index(danger_child[w][j2]),maxqueue_danger)
                        if maxqueue_danger_w > maxqueue_danger:
                            maxqueue = score[w]
                            candmax = j
                            maxqueue_danger = maxqueue_danger_w
                    elif (score[w] > maxqueue and (w in danger or (not(danger_in) and  max_danger == 0))):
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
        # if len(danger) == 2:
        #     print(new_index,given[danger[0]],given[danger[1]],given[c],len(D[new_index][danger[0]]),len(D[new_index][danger[1]]),len(set(D[new_index][danger[0]]+D[new_index][danger[1]])),len(U[new_index][danger[0]]),len(U[new_index][c]),danger[0] in U[new_index][c],danger[1] in U[new_index][c])
    maxscore = np.max(score)
    if score[c] == maxscore:
        for cand_blocked in blocked:
            if score[cand_blocked] == maxscore:
                return False,cand_blocked
        count_max = 0
        for i in range(m):
            if score[i] == maxscore:
                count_max += 1
        return True,count_max
    if verbose:
        print("The maximum is not "+str(c)+" ("+str(score[c])+") but "+str(np.argmax(score))+" ("+str(np.max(score))+")")
        if len(danger) > 0:
            print("Were minimized : "+str(danger))
    return False  ,np.argmax(score) 

    
    
    
    
    
def random_choice_approx(U,D,m,c,rule,danger=[],verbose=False,blocked=[]):
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
        danger_child = [[] for i in range(m)]
        for dangerous_candidate in danger:
            for dangerous_child in D[new_index][dangerous_candidate]:
                danger_child[dangerous_child].append(dangerous_candidate)
            
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
        minscore = np.min(score)
        maxscore = np.max(score)
        proba = [1/(score[j]-minscore+1) for j in range(m)]
        while queue_down != []:
            proba_i = [proba[j] for j in queue_down]
            for cand_w in range(len(queue_down)):
                if queue_down[cand_w] in danger:
                    proba_i[cand_w] *= 0.00000001
                proba_i[cand_w] /= max(len(danger_child[queue_down[cand_w]]),1)
            candmin = random.choices([i for i in range(len(queue_down))],proba_i)[0]
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
        proba = [1/(maxscore+1-score[j]) for j in range(m)]
        while queue_up != []:
            proba_i = [proba[j] for j in queue_up]
            for cand_w in range(len(queue_up)):
                if queue_up[cand_w] in danger:
                    proba_i[cand_w] += 1
                proba_i[cand_w] += len(danger_child[queue_up[cand_w]])/m
           # print(len(queue_up),len(proba_i))
            candmax = random.choices([i for i in range(len(queue_up))],proba_i)[0]
            w_max = queue_up[candmax]
            queue_up.pop(candmax)
            given[w_max] = rule[rank_up]
            rank_up -= 1
            for parents_w in parents[w_max]:
                child_count[parents_w] -= 1
                if child_count[parents_w] == 0:
                    queue_up.append(parents_w)        
        score += given
        # if len(danger) == 2:
        #     print(new_index,given[danger[0]],given[danger[1]],given[c],len(D[new_index][danger[0]]),len(D[new_index][danger[1]]),len(set(D[new_index][danger[0]]+D[new_index][danger[1]])),len(U[new_index][danger[0]]),len(U[new_index][c]),danger[0] in U[new_index][c],danger[1] in U[new_index][c])
    maxscore = np.max(score)
    if score[c] == maxscore:
        for cand_blocked in blocked:
            if score[cand_blocked] == maxscore:
                return False,cand_blocked
        count_max = 0
        for i in range(m):
            if score[i] == maxscore:
                count_max += 1
        return True,count_max
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
def s3_kapp_O_level_2(k,c_list,w,U,D,m): 
    [c1,c2] = c_list
    n = len(U)
    score_w = 0
    score_combl = 0
    for i in range(n):
        minpos_w = len(U[i][w])
        maxpos_c1 = m-len(D[i][c1])+1
        maxpos_c2 = m-len(D[i][c2])+1
        maxpos_c12 = m-len(set(D[i][c1]+D[i][c2]))+  1
        c1_in = c1 in U[i][w]
        c2_in = c2 in U[i][w]
        if c1_in or c2_in:
            if minpos_w > k and maxpos_c1 > k and maxpos_c2 > k and maxpos_c12 <= k:
                score_combl += 1
        else:
            if maxpos_c1 > k and maxpos_c2 > k and maxpos_c12 <= k:
                score_combl += 1
       # print(i,minpos_w,maxpos_c1,maxpos_c2,maxpos_c12,k)
    return score_combl
    
    
    
def approx_positional_scoring_rule(population,m,rule,shuffle=1,type=0,heuristic=0,verbose=False,max_tries=10,list_q=[],blocked=[],maxdiff=False,max_competition=1000):
    if type == 2:
        kapp = int(np.sum(rule))
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
        matrix_score = np.ones(m)*np.inf
        w = arglup[i]
        max_w = maximum_score-lup[w]
        if max_w < (total_score-max_w)/(m-1): 
            if verbose:
                print(str(w)+" : Default Loser")
        else:
            is_a_PW = True
            count_compet = 0
            for c in possible_winners:
                if count_compet == max_competition:
                    break
                else:
                    count_compet += 1
                score_w,score_c = nw.s3_psr_O(rule,M,c,w,U,D,m)
                matrix_score[c] = score_w-score_c
                if verbose:
                    print("Test "+str(c)+" ("+str(score_c)+") against "+str(w)+" ("+str(score_w)+")")
                if score_w < score_c:
                    is_a_PW = False
                    break
            if is_a_PW:
                cont = True
                if len(possible_winners) > 1 and type==2:
                    arg_opponents = np.argsort(matrix_score)
                    c_list = list(arg_opponents[:2])
                    c2 = c_list[1]
                    j = 2
                    while matrix_score[arg_opponents[j]] == matrix_score[c2]:
                        c_list.append(arg_opponents[j])
                        j += 1
                    for i_c_1 in range(len(c_list)):
                        for i_c_2 in range(i_c_1+1,len(c_list)):
                            c1 = c_list[i_c_1]
                            c2 = c_list[i_c_2]
                            score_to_combl = matrix_score[c1]+matrix_score[c2]
                            score_combl = s3_kapp_O_level_2(kapp,[c1,c2],w,U,D,m)
                            if verbose:
                                print(str(w)+" vs "+" : "+str(c1)+","+str(c2)+" --> ("+str(score_combl)+"/"+str(score_to_combl)+")")
                            if score_combl > score_to_combl:
                                cont = False
                if cont:
                    possible_winners.append(w)

    if len(possible_winners) <= 2 and list_q == [] and not(maxdiff):
        return possible_winners, []
    else:
        if list_q == []:
            if not(maxdiff):
                sure_PW = possible_winners[:2]
                cand_to_test = possible_winners[2:]
            else:
                sure_PW = []
                cand_to_test = possible_winners[::]
        else:
            sure_PW = []
            cand_to_test = []
            for cand_list in list_q:
                if cand_list in possible_winners:
                    cand_to_test.append(cand_list)
        possible_PW = []
        alone_winners = []
        for candidate in cand_to_test:
            if verbose:
                print("Testing "+str(candidate))
            max_w = maximum_score-lup[candidate]
            if (max_w >= (total_score)/2 and list_q == []) or (max_w > (total_score)/2): 
                sure_PW.append(candidate)
                print(str(candidate)+" : Default Winner")
            else:
                verified = False
                for i in range(shuffle):
                    danger = []
                    count_tries = 0
                    while (verified == False):
                        if heuristic == 0:
                            found_instance,winner = max_rank_approx(U,D,m,candidate,rule,danger=danger,verbose=verbose,blocked=blocked)
                        else:
                            found_instance,winner = random_choice_approx(U,D,m,candidate,rule,danger=danger,verbose=verbose,blocked=blocked)
                        if found_instance:
                            verified = True
                            if maxdiff and winner == 1:
                                alone_winners.append(candidate)
                            break
                        else:
                            if winner in danger:
                                break 
                            else:
                                danger.append(winner)
                                count_tries += 1
                                if count_tries == max_tries:
                                    break
                    if verified:
                        break
                if verified:
                    if list_q == []:
                        sure_PW.append(candidate)
                    else:
                        return True,[]
                else:
                    possible_PW.append(candidate)
    if list_q == []:
        if maxdiff:
            return sure_PW+possible_PW,alone_winners
        else:
            return sure_PW,possible_PW
    else:
        return False,possible_PW
            

    
def pw_pruning(population,m,rule,type=0,verbose=False,max_competition=10):
    if type == 2:
        kapp = int(np.sum(rule))
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
    arglup = np.argsort(lup)
    possible_winners = []
    for i in range(m):
        matrix_score = np.ones(m)*np.inf
        w = arglup[i]
        max_w = maximum_score-lup[w]
        if max_w < (total_score-max_w)/(m-1): 
            if verbose:
                print(str(w)+" : Default Loser")
        else:
            is_a_PW = True
            count_compet = 0
            for c in possible_winners:
                if count_compet == max_competition:
                    break
                else:
                    count_compet += 1
                score_w,score_c = nw.s3_psr_O(rule,M,c,w,U,D,m)
                matrix_score[c] = score_w-score_c
                if verbose:
                    print("Test "+str(c)+" ("+str(score_c)+") against "+str(w)+" ("+str(score_w)+")")
                if score_w < score_c:
                    is_a_PW = False
                    break
            if is_a_PW:
                cont = True
                if len(possible_winners) > 1 and type==2:
                    arg_opponents = np.argsort(matrix_score)
                    c_list = list(arg_opponents[:2])
                    c2 = c_list[1]
                    j = 2
                    while matrix_score[arg_opponents[j]] == matrix_score[c2]:
                        c_list.append(arg_opponents[j])
                        j += 1
                    for i_c_1 in range(len(c_list)):
                        for i_c_2 in range(i_c_1+1,len(c_list)):
                            c1 = c_list[i_c_1]
                            c2 = c_list[i_c_2]
                            score_to_combl = matrix_score[c1]+matrix_score[c2]
                            score_combl = s3_kapp_O_level_2(kapp,[c1,c2],w,U,D,m)
                            if verbose:
                                print(str(w)+" vs "+" : "+str(c1)+","+str(c2)+" --> ("+str(score_combl)+"/"+str(score_to_combl)+")")
                            if score_combl > score_to_combl:
                                cont = False
                if cont:
                    possible_winners.append(w)
    return possible_winners[:2],possible_winners[2:]
    
            
            

def approx_borda(population,m,shuffle=1,heuristic=0,verbose=False,max_tries=10,list_q=[],blocked=[],maxdiff=False,max_compet=1000):
    return approx_positional_scoring_rule(population,m,[m-1-i for i in range(m)],type=1,heuristic=heuristic,shuffle=shuffle,verbose=verbose,max_tries=max_tries,list_q=list_q,blocked=blocked,maxdiff=maxdiff,max_competition=max_compet)
    
def approx_kapproval(population,m,k,shuffle=1,heuristic=0,verbose=False,max_tries=10,list_q=[],blocked=[],maxdiff=False,max_compet=1000):
    return approx_positional_scoring_rule(population,m,[1]*k+[0]*(m-k),type=2,heuristic=heuristic,shuffle=shuffle,verbose=verbose,max_tries=max_tries,list_q=list_q,blocked=blocked,maxdiff=maxdiff,max_competition=max_compet)
    
    
## Winner set Plurality



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
            if comp_n_voters_i >= score_set_i:#*(lenc-1):
                dico[cstr] = score_set_i
                return score_set_i
            elif comp_n_voters_i + (n_voters_i-score_set_i*(lenc-1)) >= score_set_i:
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
    
    if score < n_total/m:
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
    
## Winner set Veto



def list_of_last_set(population,m):
    n = len(population)
    matrix_rank = []
    for i in range(n):
        leaves = [1]*m
        v = population[i]
        for x in v:
            (a,b) =x
            if (leaves[a]==1):
                leaves[a] = 0
        matrix_rank.append(leaves)
    return matrix_rank

def aggregate_set_veto(matrix_rank,m):
    dico = dict()
    leaves_list = []
    leaves_count = []
    i = 0
    for leaves in matrix_rank:
        if str(leaves) in dico.keys():
            leaves_count[dico[str(leaves)]] += 1
        else:
            leaves_count.append(1)
            leaves_list.append(leaves)
            dico[str(leaves)] = i
            i+=1
    return leaves_list,leaves_count
        
def zero_set_veto(dico,leaves_list,leaves_count,m,set_cand,n_voters):
    cstr = str(sorted(set_cand))
    if cstr in dico.keys():
        zero = dico[cstr]
        return zero
    lenc = len(set_cand)
    if lenc == 1:
        dico[cstr] = n_voters
       # print("1",set_cand,n_voters)
        return n_voters
    else:
        for i in range(lenc):
            cand_i = set_cand[i]
            leaves_list_i = []
            leaves_count_i = []
            n_voters_i = 0
            for j in range(len(leaves_list)):
                if leaves_list[j][cand_i] == 0:
                    n_voters_i += leaves_count[j]
                    leaves_list_i.append(leaves_list[j])
                    leaves_count_i.append(leaves_count[j])
            set_i = set_cand.copy()
            set_i.pop(i)
            comp_n_voters_i = n_voters - n_voters_i
            zero_set_i = zero_set_veto(dico,leaves_list_i,leaves_count_i,m,set_i,n_voters_i)
            if comp_n_voters_i <= zero_set_i:
                dico[cstr] = zero_set_i
               # print("<",set_cand,zero_set_i)
                return zero_set_i
        zeros = np.ceil(n_voters/lenc)
        dico[cstr] = zeros
       # print("=",set_cand,zeros)
        return zeros

def build_matrix_set_veto(g,zeros,leaves_list,leaves_count,m):
    g.add_nodes(len(leaves_count)+m)
    for i in range(len(leaves_count)):
        g.add_tedge(i,leaves_count[i],0)
        for j in range(m):
            if leaves_list[i][j] == 1:
                g.add_edge(i,len(leaves_count)+j,leaves_count[i],0)
    for j in range(m):
        g.add_tedge(len(leaves_count)+j,0,zeros)
            
    

def possibility_set_veto(dico,leaves_list,leaves_count,m,set_cand,n_total):
    n_set = 0
    compl = [x for x in range(m) if x not in set_cand]
    leaves_list_zero = []
    leaves_count_zero = []
    for i in range(len(leaves_list)):
        ok = False
        for j in compl:
            if leaves_list[i][j] == 1:
                ok = True
                break
        if not(ok):
            n_set += leaves_count[i]
            leaves_list_zero.append(leaves_list[i])
            leaves_count_zero.append(leaves_count[i])
    zeros = zero_set_veto(dico,leaves_list_zero,leaves_count_zero,m,set_cand,n_set)
    if zeros > n_total/m:
        return False
    g = mf.GraphInt()
    maxwanted = build_matrix_set_veto(g,zeros,leaves_list,leaves_count,m)
    maxflow = g.maxflow()
    if maxflow == zeros*m:
        return True
    else:
        return False
    
def veto_set(population,m,set_cand):
    dico =dict()
    n_total = len(population)
    leaves = list_of_last_set(population,m)
    leaves_list,leaves_count = aggregate_set_veto(leaves,m)
    return possibility_set_veto(dico,leaves_list,leaves_count,m,set_cand,n_total)
     
##


def build_model_kapp(poset,m,k,Ws,blocked=[]):
    n = len(poset)
    model = Model("possible_winner")    
    x = model.addVars(n, m, vtype = GRB.BINARY, name = "x" )
    
    W = Ws[0]
    for cand in range(m):
        if cand in Ws:
            if cand != W:
                model.addConstr( sum( x[l, cand] for l in range(n)) - sum(x[l,W] for l in range(n)) == 0)
        else:
            if cand in blocked:
                model.addConstr( sum( x[l, cand] for l in range(n)) <=  sum(x[l,W] for l in range(n))-1)
            else:
                model.addConstr( sum( x[l, cand] for l in range(n)) - sum(x[l,W] for l in range(n)) <= 0)
    model.addConstrs((sum(x[l,j] for j in range(m)) == k) for l in range(n))
    for l in range(n):
        for (a,b) in poset[l]:
            model.addConstr(x[l,a] - x[l,b] >= 0)
    return model
    
    
def build_model_borda(poset,m,Ws,blocked=[]):
    n = len(poset)
    model = Model("possible_winner")    
    x = model.addVars(n, m,m, vtype = GRB.BINARY, name = "x" )
    for l in range(n):
        for (a,b) in poset[l]:
            model.addConstr(x[l,a,b] == 1)
       
    model.addConstrs( (x[l, i, j] + x[l, j, i] == 1) 
                        for l in range(n) 
                        for i in range(m) 
                        for j in range(m) 
                        if i!=j )
    model.addConstrs( (x[l, i, j] + x[l, j, k] + x[l, k, i] <= 2) 
                        for l in range(n)
                        for i in range(m)
                        for j in range(m)
                        for k in range(m)
                        if i != j and i != k and j != k)
    W = Ws[0]
    for cand in range(m):
            if cand in Ws:
                if cand != W:
                    model.addConstr( sum( x[l, cand, j] for l in range(n)
                                    for j in range(m) if j != cand) ==
                                sum(x[l,W,k] for l in range(n)
                                for k in range(m) if k!= W))
            else:
                if cand in blocked:
                    model.addConstr( sum( x[l, cand, j] for l in range(n)
                                    for j in range(m) if j != cand) <
                                sum(x[l,W,k] for l in range(n)
                                for k in range(m) if k!= W))
                else:
                    model.addConstr( sum( x[l, cand, j] for l in range(n)
                                    for j in range(m) if j != cand) <= 
                                sum(x[l,W,k] for l in range(n)
                                for k in range(m) if k!= W))
    return model
    
def build_model_psr(poset,m,rule,Ws,blocked=[]):
    n = len(poset)
    model = Model("possible_winner")    
    x = model.addVars(n, m,m, vtype = GRB.BINARY, name = "x" )
    for l in range(n):
        for (a,b) in poset[l]:
            model.addConstr(sum(p*x[l,b,p] for p in range(m)) >= sum(p*x[l,a,p] for p in range(m)))
       
    model.addConstrs( sum(x[l,i,p] for p in range(m)) == 1 for i in range(m) for l in range(n))
    model.addConstrs( sum(x[l,i,p] for i in range(m)) == 1 for p in range(m) for l in range(n))
    W = Ws[0]
    for cand in range(m):
            if cand in Ws:
                if cand != W:
                    model.addConstr( sum( sum(rule[p]*x[l, cand, p] for p in range(m)) for l in range(n))
                                     == sum( sum(rule[p]*x[l, W, p] for p in range(m)) for l in range(n)))
            else:
                if cand in blocked:
                    model.addConstr( sum( sum(rule[p]*x[l, cand, p] for p in range(m)) for l in range(n))
                                     < sum( sum(rule[p]*x[l, W, p] for p in range(m)) for l in range(n)))
                else:
                    model.addConstr( sum( sum(rule[p]*x[l, cand, p] for p in range(m)) for l in range(n))
                                     <= sum( sum(rule[p]*x[l, W, p] for p in range(m)) for l in range(n)))
    return model

def kapp(P,m,k,verbose=False,shuffle=1,max_tries=3,heuristic=0):
    pws,pwp = approx_kapproval(P,m,k,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose)
    for cand in pwp:
        model = build_model_kapp(P,m,k,[cand])
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            pws.append(cand)
    return pws      
        
        


# def borda(P,m,verbose=False,shuffle=1,max_tries=3,heuristic=0):
#     pws,pwp = approx_borda(P,m,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose)
#     for cand in pwp:
#         model = build_model_borda(P,m,[cand])
#         model.optimize()
#         if model.status == GRB.Status.OPTIMAL:
#             pws.append(cand)
#     return pws
#     
#     
    

def borda(P,m,verbose=False,shuffle=1,max_tries=3,heuristic=0):
    pws,pwp = approx_borda(P,m,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose)
    borda_rule = [m-1-i for i in range(m)]
    for cand in pwp:
        model = build_model_psr(P,m,borda_rule,[cand])
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            pws.append(cand)
    return pws
    