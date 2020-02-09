import mysql.connector
from pycomsoc.winners import pw
import numpy as np
from gurobipy import *

## This code compile code to run query with possible and necessary answers 

def set_to_test(results,m):
    if len(results) == 1:
        dict_set = dict()
        sets = [[] for i in range(m-1)]
        cand_unic = [0 for i in range(m)]
        for r_i in results[0]:
            set_bool = [0 for i in range(m)]
            for cand in r_i:
                cand_unic[cand] = 1
                set_bool[cand] = 1
            tup_set = tuple(set_bool)
            if tup_set not in dict_set.keys():
                sets[np.sum(set_bool)-1].append(set_bool)
                dict_set[tup_set] = True
        return cand_unic,sets
    else:
        cand_unic,sets_rec = set_to_test(results[1:],m)
        result = results[0]
        sets = [[] for i in range(m-1)]
        dict_set = dict()
        for r_i in results[0]:
            for i in range(m-1):
                for s in sets_rec[i]:
                    s_copy = s.copy()
                    for cand in r_i:
                        s_copy[cand] = 1
                        cand_unic[cand] = 1
                    tup_set = tuple(s_copy)
                    if tup_set not in dict_set.keys():
                        sets[np.sum(s_copy)-1].append(s_copy)
                        dict_set[tup_set] = True
        return cand_unic,sets
        
def delete_doublons(sets):
    m = len(sets)+1
    for i in range(m-1):
        for s_i in sets[i]:
            for j in range(i+1,m-1):
                to_pop = []
                for ind_j,s_j in enumerate(sets[j]):
                    sub = True
                    for k in range(m):
                        if s_j[k] - s_i[k] < 0:
                            sub = False
                            break
                    if sub:
                        to_pop.append(ind_j)
                to_pop.reverse()
                for ind_j in to_pop:
                    sets[j].pop(ind_j)
    
                            
        
def plurality_pw(population,m,list,verbose=False):
    roots_list_net = pw.get_roots(population,m)
    roots_list,count_roots = pw.aggregate_plurality(roots_list_net,m)
    list = [l[0] for l in list]
    for c in list:
        if pw.possible_winner_plurality(roots_list,count_roots,m,c,verbose=verbose):
                return True
    return False
    
def plurality_pw_sets(population,m,cand_unic,sets,verbose=False):
    roots_list_net = pw.get_roots(population,m)
    roots_list,count_roots = pw.aggregate_plurality(roots_list_net,m)
    cand_wins = [0 for i in range(m)]
    for i in range(m):
        if cand_unic[i]:
            if pw.possible_winner_plurality(roots_list,count_roots,m,i,verbose=verbose):
                cand_wins[i] = 1
    for set in sets[0]:
        if cand_wins[np.argmax(set)] == 1:
            return True
    dico =dict()
    n_total = len(population)
    roots = pw.list_of_first_set(population,m)
    roots_list,roots_count = pw.aggregate_set(roots,m)
    for i in range(1,m-1):
        sets_len_i = sets[i]
        for set in sets_len_i:
            if np.min([cand_wins[j] - set[j] for j in range(m)]) >= 0:
                if pw.possibility_set(dico,roots_list,roots_count,m,[x for x in range(m) if set[x] == 1],n_total):
                    return True
    return False
    
    
def veto_pw_sets(population,m,cand_unic,sets,verbose=False):
    leaves_list_net = pw.get_leaves(population,m)
    leaves_list,count_leaves = pw.aggregate_veto(leaves_list_net,m)
    n_total = len(population)
    cand_wins = [0 for i in range(m)]
    for i in range(m):
        if cand_unic[i]:
            if pw.possible_winner_veto(leaves_list,count_leaves,m,i,n_total,verbose=verbose):
                cand_wins[i] = 1
    for set in sets[0]:
        if cand_wins[np.argmax(set)] == 1:
            return True
    dico =dict()
    leaves = pw.list_of_last_set(population,m)
    leaves_list,leaves_count = pw.aggregate_set(leaves,m)
    for i in range(1,m-1):
        sets_len_i = sets[i]
        for set in sets_len_i:
            if np.min([cand_wins[j] - set[j] for j in range(m)]) >= 0:
                if pw.possibility_set_veto(dico,leaves_list,leaves_count,m,[x for x in range(m) if set[x] == 1],n_total):
                    return True
    return False
    
def plurality_nw(population,m,list,verbose=False):
    roots_list_net = pw.get_roots(population,m)
    roots_list,count_roots = pw.aggregate_plurality(roots_list_net,m)
    list = [l[0] for l in list]
    compl = [x for x in range(m) if x not in list]
    for c in compl:
        if pw.possible_winner_plurality(roots_list,count_roots,m,c,verbose=verbose,blocked=list):
            return False
    return True
    
def plurality_nw_sets(population,m,sets,verbose=False):
    roots_list_net = pw.get_roots(population,m)
    roots_list,count_roots = pw.aggregate_plurality(roots_list_net,m)
    cand_wins = [0 for i in range(m)]
    to_test = [1 for i in range(m)]
    for set in sets[0]:
        to_test[np.argmax(set)] = 0
    wins = []
    for i in range(m):
        if to_test[i]:
            if pw.possible_winner_plurality(roots_list,count_roots,m,i,verbose=verbose):
                wins.append(i)
    wins_not_alone = []
    for cand in wins:
        if pw.possible_winner_plurality(roots_list,count_roots,m,i,verbose=verbose,blocked=[j for j in range(m) if j != i]):
            return False
        else:
            zeros = [0]*m
            zeros[cand] = 1
            wins_not_alone.append(zeros)
    dico =dict()
    n_total = len(population)
    roots = pw.list_of_first_set(population,m)
    roots_list,roots_count = pw.aggregate_set(roots,m)
    for i in range(1,m-1):
        sets_len_i = sets[i]
        seen_this_time = dict()
        new_winners = []
        for set_of_winners in wins_not_alone:
            for i in range(m):
                if set_of_winners[i] == 0:
                    set_copy = set_of_winners.copy()
                    set_copy[i] = 1
                    tuple_set = tuple(set_copy)
                    if (tuple_set in seen_this_time.keys()) or (set_copy in sets_len_i):
                        ()
                    else:
                        ok = True
                        for j in range(m):
                            if set_copy[j] == 1:
                                set_copy2 = set_copy.copy()
                                set_copy2[j] = 0
                                if set_copy2 not in wins_not_alone:
                                    ok = False
                        if ok:
                            if pw.possibility_set(dico,roots_list,roots_count,m,[x for x in range(m) if set_copy[x] == 1],n_total):
                                new_winners.append(set_copy)
                                if pw.possibility_set(dico,roots_list,roots_count,m,[x for x in range(m) if set_copy[x] == 1],n_total,blocked=[x for x in range(m) if set_copy[x] == 0]):
                                    return False
        wins_not_alone = new_winners.copy()
        if wins_not_alone == []:
            return True
    return True
    
def veto_pw(population,m,list,verbose=False):
    leaves_list_net = pw.get_leaves(population,m)
    leaves_list,count_leaves = pw.aggregate_veto(leaves_list_net,m)
    list = [l[0] for l in list]
    for c in list:
        if pw.possible_winner_veto(leaves_list,count_leaves,m,c,len(population),verbose=verbose):
            return True
    return False
    
def veto_nw(population,m,list,verbose=False):
    leaves_list_net = pw.get_leaves(population,m)
    leaves_list,count_leaves = pw.aggregate_veto(leaves_list_net,m)
    list = [l[0] for l in list]
    compl = [x for x in range(m) if x not in list]
    for c in compl:
        if pw.possible_winner_veto(leaves_list,count_leaves,m,c,len(population),verbose=verbose,blocked=list):
            return False
    return True

def veto_nw_sets(population,m,sets,verbose=False):
    leaves_list_net = pw.get_leaves(population,m)
    leaves_list,count_leaves = pw.aggregate_veto(leaves_list_net,m)
    cand_wins = [0 for i in range(m)]
    n_total = len(population)
    to_test = [1 for i in range(m)]
    for set in sets[0]:
        to_test[np.argmax(set)] = 0
    wins = []
    for i in range(m):
        if to_test[i]:
            if pw.possible_winner_veto(leaves_list,count_leaves,m,i,n_total,verbose=verbose):
                wins.append(i)
    wins_not_alone = []
    for cand in wins:
        if pw.possible_winner_veto(leaves_list,count_leaves,m,i,n_total,verbose=verbose,blocked=[j for j in range(m) if j != i]):
            return False
        else:
            zeros = [0]*m
            zeros[cand] = 1
            wins_not_alone.append(zeros)
    dico =dict()
    leaves = pw.list_of_last_set(population,m)
    leaves_list,leaves_count = pw.aggregate_set_veto(leaves,m)
    for i in range(1,m-1):
        sets_len_i = sets[i]
        seen_this_time = dict()
        new_winners = []
        for set_of_winners in wins_not_alone:
            for i in range(m):
                if set_of_winners[i] == 0:
                    set_copy = set_of_winners.copy()
                    set_copy[i] = 1
                    tuple_set = tuple(set_copy)
                    if (tuple_set in seen_this_time.keys()) or (set_copy in sets_len_i):
                        ()
                    else:
                        ok = True
                        for j in range(m):
                            if set_copy[j] == 1:
                                set_copy2 = set_copy.copy()
                                set_copy2[j] = 0
                                if set_copy2 not in wins_not_alone:
                                    ok = False
                        if ok:
                            if pw.possibility_set_veto(dico,leaves_list,leaves_count,m,[x for x in range(m) if set_copy[x] == 1],n_total):
                                new_winners.append(set_copy)
                                if pw.possibility_set_veto(dico,leaves_list,leaves_count,m,[x for x in range(m) if set_copy[x] == 1],n_total,blocked=[x for x in range(m) if set_copy[x] == 0]):
                                    return False
        wins_not_alone = new_winners.copy()
        if wins_not_alone == []:
            return True
    return True
    

def kapp_pw(population,m,k,list,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    list = [l[0] for l in list]
    found,pwp = pw.approx_kapproval(population,m,k,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose,list_q=list)
    if found:
        return True
    for cand in pwp:
        model = pw.build_model_kapp(population,m,k,[cand])
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            return True
    return False  
    
def kapp_pw_sets(population,m,k,sets,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    pws,pwp = pw.approx_kapproval(population,m,k,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose)
    cand_wins = [0 for i in range(m)]
    for i in range(m):
        if (i in pws) or (i in pwp):
            cand_wins[i] = 1
    
    for set in sets[0]:
        if cand_wins[np.argmax(set)] == 1:
            return True
            
    for i in range(1,m-1):
        sets_len_i = sets[i]
        for set in sets_len_i:
            if np.min([cand_wins[j] - set[j] for j in range(m)]) >= 0:
                model = pw.build_model_kapp(population,m,k,[x for x in range(m) if set[x] == 1])
                model.optimize()
                if model.status == GRB.Status.OPTIMAL:
                    return True
    return False
    
    
def kapp_nw(population,m,k,list,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    list = [l[0] for l in list]
    compl = [x for x in range(m) if x not in list]
    found,pwp = pw.approx_kapproval(population,m,k,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose,list_q=compl,blocked=list)
    if found:
        return False
    for cand in pwp:
        model = pw.build_model_kapp(population,m,k,[cand],blocked=list)
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            return False
    return True  
    
def kapp_nw_sets(population,m,k,sets,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    winners,winners_alone = pw.approx_kapproval(population,m,k,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose,maxdiff=True)
    to_test = [0 for i in range(m)]
    for i in range(m):
        if i in winners:
            to_test[i] = 1
    for set in sets[0]:
        to_test[np.argmax(set)] = 0
    wins_not_alone = []
    for i in range(m):
        if to_test[i] == 1:
            if i in winners_alone:
                return False
            else:
                model = pw.build_model_kapp(population,m,k,[i],blocked=[x for x in range(m) if x != i])
                model.optimize()
                if model.status == GRB.Status.OPTIMAL:
                    return False
                else:
                    zeros = [0]*m
                    zeros[cand] = 1
                    wins_not_alone.append(zeros)
    for i in range(1,m-1):
        sets_len_i = sets[i]
        seen_this_time = dict()
        new_winners = []
        for set_of_winners in wins_not_alone:
            for i in range(m):
                if set_of_winners[i] == 0:
                    set_copy = set_of_winners.copy()
                    set_copy[i] = 1
                    tuple_set = tuple(set_copy)
                    if (tuple_set in seen_this_time.keys()) or (set_copy in sets_len_i):
                        ()
                    else:
                        ok = True
                        for j in range(m):
                            if set_copy[j] == 1:
                                if j not in winners:
                                    ok = False
                                    break
                                set_copy2 = set_copy.copy()
                                set_copy2[j] = 0
                                if set_copy2 not in wins_not_alone:
                                    ok = False
                                    break
                        if ok :
                            model = pw.build_model_kapp(population,m,k,[x for x in range(m) if set[x] == 1])
                            model.optimize()
                            if model.status == GRB.Status.OPTIMAL:
                                model = pw.build_model_kapp(population,m,k,[x for x in range(m) if set[x] == 1],blocked=[x for x in range(m) if set[x] == 0])
                                model.optimize()
                                if model.status == GRB.Status.OPTIMAL:
                                    return False
                                else:
                                    new_winners.append(set_copy)
        wins_not_alone = new_winners.copy()
        if wins_not_alone == []:
            return True
    return True  
    
    
def borda_pw(population,m,list,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    list = [l[0] for l in list]
    found,pwp = pw.approx_borda(population,m,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose,list_q=list)
    if found:
        return True
    for cand in pwp:
        model = pw.build_model_borda(population,m,[cand])
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            return True
    return False  
    
def borda_pw_sets(population,m,sets,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    pws,pwp = pw.approx_borda(population,m,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose)
    cand_wins = [0 for i in range(m)]
    for i in range(m):
        if (i in pws) or (i in pwp):
            cand_wins[i] = 1
    
    for set in sets[0]:
        if cand_wins[np.argmax(set)] == 1:
            return True
            
    for i in range(1,m-1):
        sets_len_i = sets[i]
        for set in sets_len_i:
            if np.min([cand_wins[j] - set[j] for j in range(m)]) >= 0:
                model = pw.build_model_borda(population,m,[x for x in range(m) if set[x] == 1])
                model.optimize()
                if model.status == GRB.Status.OPTIMAL:
                    return True
    return False
    
    
def borda_nw(population,m,list,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    list = [l[0] for l in list]
    compl = [x for x in range(m) if x not in list]
    found,pwp = pw.approx_borda(population,m,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose,list_q=compl,blocked=list)

    if found:
        return False
    for cand in pwp:
        model = pw.build_model_borda(population,m,[cand],blocked=list)
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            return False
    return True 

def borda_nw_sets(population,m,sets,verbose=False,shuffle=2,heuristic=0,max_tries=3):
    winners,winners_alone = pw.approx_borda(population,m,shuffle=shuffle,max_tries=max_tries,heuristic=heuristic,verbose=verbose,maxdiff=True)
    to_test = [0 for i in range(m)]
    for i in range(m):
        if i in winners:
            to_test[i] = 1
    for set in sets[0]:
        to_test[np.argmax(set)] = 0
    wins_not_alone = []
    for i in range(m):
        if to_test[i] == 1:
            if i in winners_alone:
                return False
            else:
                model = pw.build_model_borda(population,m,[i],blocked=[x for x in range(m) if x != i])
                model.optimize()
                if model.status == GRB.Status.OPTIMAL:
                    return False
                else:
                    zeros = [0]*m
                    zeros[cand] = 1
                    wins_not_alone.append(zeros)
    for i in range(1,m-1):
        sets_len_i = sets[i]
        seen_this_time = dict()
        new_winners = []
        for set_of_winners in wins_not_alone:
            for i in range(m):
                if set_of_winners[i] == 0:
                    set_copy = set_of_winners.copy()
                    set_copy[i] = 1
                    tuple_set = tuple(set_copy)
                    if (tuple_set in seen_this_time.keys()) or (set_copy in sets_len_i):
                        ()
                    else:
                        ok = True
                        for j in range(m):
                            if set_copy[j] == 1:
                                if j not in winners:
                                    ok = False
                                    break
                                set_copy2 = set_copy.copy()
                                set_copy2[j] = 0
                                if set_copy2 not in wins_not_alone:
                                    ok = False
                                    break
                        if ok :
                            model = pw.build_model_borda(population,m,[x for x in range(m) if set[x] == 1])
                            model.optimize()
                            if model.status == GRB.Status.OPTIMAL:
                                model = pw.build_model_borda(population,m,[x for x in range(m) if set[x] == 1],blocked=[x for x in range(m) if set[x] == 0])
                                model.optimize()
                                if model.status == GRB.Status.OPTIMAL:
                                    return False
                                else:
                                    new_winners.append(set_copy)
        wins_not_alone = new_winners.copy()
        if wins_not_alone == []:
            return True
    return True  
    
    
class Query():
    def __init__(self,q,db,m,ballots="ballots"):
        (name,atoms) = q
        self.name = name
        dico_var = dict()
        indice_table = 0
        nb_atoms = len(atoms)
        M = np.zeros((nb_atoms,nb_atoms))
        elemts = dict()
        for i in range(nb_atoms):
            at = atoms[i]
            (table,tuple) = at
            if table == "WINNER":
                if len(tuple) != 1:
                    raise ValueError("Only one variable for winner atoms")
                else:
                    (k,row,val) = tuple[0]
                    if k == "NUMBER":
                        raise ValueError("Only variables on winner atoms")
                    else:
                        if val in elemts.keys():
                            j = elemts[val]
                            M[i][j] = 1
                            M[j][i] = 1
                        else:
                            elemts[val] = i
            else:
                for elem in tuple:
                    (k,row,val) = elem
                    if val in elemts.keys():
                        j = elemts[val]
                        M[i][j] = 1
                        M[j][i] = 1
                    else:
                        elemts[val] = i
        connex_comp = [-1 for i in range(nb_atoms)]
        seen = 0
        cc = -1
        queue = []
        while seen < nb_atoms:
            if queue == []:
                j = 0
                while connex_comp[j] != -1:
                    j+=1
                queue.append(j)
                cc += 1
            else:
                j = queue.pop()
                seen += 1
                connex_comp[j] = cc
                for i in range(nb_atoms):
                    if M[j][i] != 0 and connex_comp[i] == -1:
                        queue.append(i)
        cc += 1
        winners_var = [[] for i in range(cc)]
        query_from = ["" for i in range(cc)]
        query_where = ["" for i in range(cc)]
        self.cq = ["" for i in range(cc)]
        for ind_at,atom in enumerate(atoms):
            (table,tuple) = atom
            if table == "WINNER":
                (k,row,val) = tuple[0]
                winners_var[connex_comp[ind_at]].append(val)
            else:
                if query_from[connex_comp[ind_at]] == "":
                    query_from[connex_comp[ind_at]] += table+" as t"+str(indice_table)
                else:
                    query_from[connex_comp[ind_at]] += ", "+table+" as t"+str(indice_table)
                for elem in tuple:
                    (k,row,val) = elem
                    if k == "NUMBER":
                        query_where[connex_comp[ind_at]] += "t"+str(indice_table)+"."+row+" = "+str(val)+" AND "
                    elif k == "BOOL":
                        query_where[connex_comp[ind_at]] += "t"+str(indice_table)+"."+row+" = "+val+" AND "
                    elif k == "STRING":
                        query_where[connex_comp[ind_at]] += "t"+str(indice_table)+"."+row+' = "'+val+'" AND '
                    else:
                        if val in dico_var.keys():
                            ident = dico_var[val]
                            query_where[connex_comp[ind_at]] += "t"+str(indice_table)+"."+row+" = "+ident+" AND "
                        else:
                            dico_var[val] = "t"+str(indice_table)+"."+row
                indice_table += 1
        query_where = [q[:len(q)-4] for q in query_where]
        query_select = ["" for i in range(cc)]
        self.cq_results = [[] for i in range(cc)]
        for i in range(cc):
            for cand in winners_var[i]:
                if cand in dico_var.keys():
                    ident = dico_var[cand]
                    if query_select[i] == "":
                        query_select[i] += ident
                    else:
                        query_select[i] += ","+ident
                else:
                    raise ValueError("Winner variable not defined")
            if query_where[i] == "":
                self.cq[i] =  "SELECT DISTINCT "+query_select[i]+" FROM "+query_from[i]+";"
            else:
                self.cq[i] =  "SELECT DISTINCT "+query_select[i]+" FROM "+query_from[i]+" WHERE "+query_where[i]+";"
            cursor = db.cursor()
            print(self.cq[i])
            cursor.execute(self.cq[i])
            results = cursor.fetchall()
            self.cq_results[i] = [[xi for xi in x] for x in results]
        P = []
        dico_user = dict()
        get_ballots = "SELECT * FROM "+ballots+";"
        cursor.execute(get_ballots)
        results_ballots = cursor.fetchall()
        incr = 0
        for (id_u,id_y,id_n) in results_ballots:
            if id_u in dico_user.keys():
                P[dico_user[id_u]].append((id_y,id_n))
            else:
                dico_user[id_u] = incr
                incr += 1
                P.append([(id_y,id_n)])
        self.ballots = P
        self.m = m
        
    def possibility(self,rule="plurality",k=0,verbose=False):
        if len(self.cq_results) == 1 and len(self.cq_results[0]) > 0 and len(self.cq_results[0][0]) == 1:
            
            
            if rule == "plurality":
                return plurality_pw(self.ballots,self.m,self.cq_results[0],verbose=verbose)
            elif rule == "veto":
                return veto_pw(self.ballots,self.m,self.cq_results[0],verbose=verbose)
            elif rule == "k_approval":
                if k == 0:
                    raise ValueError("k must be different than 0")
                return kapp_pw(self.ballots,self.m,k,self.cq_results[0],verbose=verbose)
            elif rule == "borda":
                return borda_pw(self.ballots,self.m,self.cq_results[0],verbose=verbose)
            else:
                raise ValueError("Rule unknown. Try 'plurality', 'veto', 'k_approval' or 'borda'.")
        elif self.cq_results[0] == []:
            return False
        else:
            l = set_to_test(self.cq_results,self.m)
            cand_unic,sets = l
            delete_doublons(sets)
            if rule == "plurality":
                return plurality_pw_sets(self.ballots,self.m,cand_unic,sets,verbose=verbose)
            elif rule == "veto":
                return veto_pw_sets(self.ballots,self.m,cand_unic,sets,verbose=verbose)
            elif rule == "k_approval":
                if k == 0:
                    raise ValueError("k must be different than 0")
                return kapp_pw_sets(self.ballots,self.m,k,sets,verbose=verbose)
            elif rule == "borda":
                return borda_pw_sets(self.ballots,self.m,sets,verbose=verbose)
            else:
                raise ValueError("Rule unknown. Try 'plurality', 'veto', 'k_approval' or 'borda'.")
                    
    def necessity(self,rule="plurality",k=0,verbose=False):
        if rule == "plurality":
            nec = True
            for i in range(len(self.cq_results)):
                if len(self.cq_results[i]) == 0:
                    return False
                elif len(self.cq_results[i][0]) == 1:
                    nec = nec and plurality_nw(self.ballots,self.m,self.cq_results[i],verbose=verbose)
                else:
                    l = set_to_test([self.cq_results[i]],self.m)
                    cand_unic,sets = l
                    delete_doublons(sets)
                    nec = nec and plurality_nw_sets(self.ballots,self.m,sets,verbose=verbose)
            return nec
        elif rule == "veto":
            nec = True
            for i in range(len(self.cq_results)):
                if len(self.cq_results[i]) == 0:
                    return False
                elif len(self.cq_results[i][0]) == 1:
                    nec = nec and veto_nw(self.ballots,self.m,self.cq_results[i],verbose=verbose)
                else:
                    l = set_to_test([self.cq_results[i]],self.m)
                    cand_unic,sets = l
                    delete_doublons(sets)
                    nec = nec and veto_nw_sets(self.ballots,self.m,sets,verbose=verbose)
            return nec
        elif rule == "k_approval":
            if k == 0:
                raise ValueError("k must be different than 0")
            nec = True
            for i in range(len(self.cq_results)):
                if len(self.cq_results[i]) == 0:
                    return False
                elif len(self.cq_results[i][0]) == 1:
                    nec = nec and kapp_nw(self.ballots,self.m,k,self.cq_results[i],verbose=verbose)
                else:
                    l = set_to_test([self.cq_results[i]],self.m)
                    cand_unic,sets = l
                    delete_doublons(sets)
                    nec = nec and kapp_nw_sets(self.ballots,self.m,k,sets,verbose=verbose)
                return nec
        elif rule == "borda":
            nec = True
            for i in range(len(self.cq_results)):
                if len(self.cq_results[i]) == 0:
                    return False
                elif len(self.cq_results[i][0]) == 1:
                    nec = nec and borda_nw(self.ballots,self.m,self.cq_results[i],verbose=verbose)
                else:
                    l = set_to_test([self.cq_results[i]],self.m)
                    cand_unic,sets = l
                    delete_doublons(sets)
                    nec = nec and borda_nw_sets(self.ballots,self.m,sets,verbose=verbose)
                return nec
        else:
            raise ValueError("Rule unknown. Try 'plurality', 'veto', 'k_approval' or 'borda'.")


#4. > x 
#5. clean data