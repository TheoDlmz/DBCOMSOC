from typing import List, Tuple
import numpy as np
import random
from random import choices

def density(poset,m):
    M = np.zeros((m,m))
    M_path= np.zeros((m))
    for (a,b) in poset:
        M[a][b] = 1
    for j in range(m-1):
        M_path = M + np.dot(M_path,M)
    v_with_tc = []
    d = 0
    for i in range(m):
        for j in range(m):
            if M_path[i][j] != 0:
                d += 1
    return 2*d/(m*(m-1))


def transitive_closure(poset,m):
    M = np.zeros((m,m))
    M_path= np.zeros((m))
    for (a,b) in poset:
        M[a][b] = 1
    for j in range(m-1):
        M_path = M + np.dot(M_path,M)
    v_with_tc = []
    for i in range(m):
        for j in range(m):
            if M_path[i][j] != 0:
                v_with_tc.append((i,j))
    return v_with_tc
    
    
def init_proba_rank(Pi,m):
    M = np.zeros((m,m)) 
    for i in range(m):
        M[i][0] = Pi[0][i] 
    mul = 1
    mul_bis = 1
    for j in range(m):
        M[0][j] = mul*Pi[j][0]
        M[m-1][j] = mul_bis*Pi[j][m-1-j]
        mul_bis *= (1-Pi[j][m-1-j])
        mul *= (1-Pi[j][0])
    return M
    
def delete_first(M):
    m = len(M)
    M_new = np.zeros((m-1,m-1))
    for i in range(m-1):
        for j in range(m-1):
            M_new[i][j] = M[i+1][j]
    return M_new
    
def solve_proba_rank(Pi,m):
    M = init_proba_rank(Pi,m)
    if m == 1:
        return M
    else:
        Pi_m_minus_1 = delete_first(Pi)
        M_minus_1 = solve_proba_rank(Pi_m_minus_1,m-1)
        for j in range(1,m):
            for i in range(1,m-1):
                M[i][j] = np.sum(Pi[0][:i])*M_minus_1[i-1,j-1] + np.sum(Pi[0][(i+1):])*M_minus_1[i,j-1]
        return M #M[i][j] = proba that i go to rank j
        
        
def solve_pairs(Pi,m):
    if m == 2:
        return [[0,Pi[0][0]],[Pi[0][1],0]]
    else:
        M =np.zeros((m,m))
        Pi_m_minus_1 = delete_first(Pi)
        M_minus_1 = solve_pairs(Pi_m_minus_1,m-1)
       
        for j in range(1,m):
            M[0][j] = Pi[0][0] + np.sum(Pi[0][(1):j])*M_minus_1[0][j-1]
            if j < m-1:
                M[0][j] += np.sum(Pi[0][(j+1):])*M_minus_1[0][j]
            for i in range(1,j):
                if i < j:
                    M[i][j] = Pi[0][i] + np.sum(Pi[0][:i])*M_minus_1[i-1][j-1] + np.sum(Pi[0][(i+1):j])*M_minus_1[i][j-1]
                    if j < m-1:
                        M[i][j] += np.sum(Pi[0][(j+1):])*M_minus_1[i][j]
                elif i > j:
                    M[i][j] = Pi[0][i] + np.sum(Pi[0][:j])*M_minus_1[i-1][j-1] + np.sum(Pi[0][(j+1):i])*M_minus_1[i-1][j] 
                    if i < m-1:
                        M[i][j] += np.sum(Pi[0][(i+1):])*M_minus_1[i][j]
        for i in range(m):
            for j in range(i+1,m):
                M[j][i] = 1- M[i][j]
        return M #proba to have pairs (i,j)
        
        
def solve_poset(Pi,m):
    if m==2:
        return [[[0,Pi[0][0]],[Pi[0][1],0]],[[0,0],[0,0]]]
    else:
        M = np.zeros((m,m,m))
        Pi_m_minus_1 = delete_first(Pi)
        M_minus_1 = solve_poset(Pi_m_minus_1,m-1)
        #k = 0
        for i in range(m):
            for j in range(m):
                if i !=j:
                    M[0][i][j] = Pi[0][i]
        for k in range(1,m):
            for i in range(m):
                for j in range(m):
                    if i > j:
                        M[k][i][j] = np.sum(Pi[0][j+1:i])*M_minus_1[k-1][i-1][j] 
                        if j > 0:
                            M[k][i][j] += np.sum(Pi[0][:j])*M_minus_1[k-1][i-1][j-1]
                        if i < m -1 :
                            M[k][i][j] += np.sum(Pi[0][i+1:])*M_minus_1[k-1][i][j]
                    elif i < j:
                        M[k][i][j] =  np.sum(Pi[0][i+1:j])*M_minus_1[k-1][i][j-1] 
                        if i > 0:
                            M[k][i][j] += np.sum(Pi[0][:i])*M_minus_1[k-1][i-1][j-1]
                        if j < m -1 :
                            M[k][i][j] += np.sum(Pi[0][j+1:])*M_minus_1[k-1][i][j]
        return M #proba to have pairs (i,j)

class Mallows(object):

    def __init__(self, center: List, phi: float):
        self.center = list(center)
        self.phi = phi

        self.m: int = len(self.center)
        self.item_to_rank = {item: rank for rank, item in enumerate(self.center)}

        self.pij_matrix: Tuple[Tuple[float]] = self.calculate_pij_matrix()
        self.normalization_constant = self.calculate_normalization_constant()

    def __str__(self):
        return f'Mallows(center={self.center}, phi={self.phi})'

    def get_prob_i_j(self, i, j) -> float:
        return self.pij_matrix[i][j]
    
    def set_center(self,center):
        self.center = list(center)

    def get_rank_of_item(self, item):
        return self.item_to_rank[item]

    def sample_a_ranking(self) -> List:
        ranking = []
        insertion_range = []

        for step, item in enumerate(self.center):
            insertion_range.append(step)
            sample_index = choices(insertion_range, weights=self.pij_matrix[step])[0]

            ranking.insert(sample_index, item)

        return ranking

    def sample_a_permutation(self) -> List:
        return self.sample_a_ranking()

    def calculate_normalization_constant(self) -> float:
        try:
            norm = (1 - self.phi) ** (-self.m)
            phi_i = self.phi
            for i in range(1, self.m + 1):
                norm *= (1 - phi_i)
                phi_i *= self.phi
        except ZeroDivisionError:
            norm = factorial(self.m)
        return norm

    def calculate_kendall_tau_distance(self, permutation) -> int:
        dist = 0
        for i, e_i in enumerate(permutation[:-1]):
            for e_j in permutation[i:]:
                if self.get_rank_of_item(e_i) > self.get_rank_of_item(e_j):
                    dist +=1 

        return dist

    def calculate_prob_by_distance(self, distance):
        return (self.phi ** distance) / self.normalization_constant

    def calculate_prob_of_permutation(self, permutation):
        dist = self.calculate_kendall_tau_distance(permutation)
        return self.calculate_prob_by_distance(dist)

    def calculate_pij_matrix(self):

        pij = []
        for i in range(self.m):
            pi = [self.phi ** (i - j) for j in range(i + 1)]
            summation = sum(pi)
            pi = [p / summation for p in pi]
            pij.append(tuple(pi))

        return tuple(pij)
        
        
class Poset(object):
    def __init__(self,pairs:List):
        self.pairs = list(pairs)
        self.elements = self.get_elements()
        self.m = int(np.max(self.elements))+1
        #if self.is_there_cycle():
         #   raise ValueError("This poset contains cycle")

    def get_elements(self) -> List:
        elements = []
        for (a,b) in self.pairs:
            if a not in elements:
                elements.append(a)
            if b not in elements:
                elements.append(b)
        elements.sort()
        return elements
    
    def is_there_TC(self,verbose=False) -> bool:
        M = np.zeros((self.m,self.m))
        M_path= np.zeros((self.m,self.m))
        for (e1,e2) in self.pairs:
                M[self.elements.index(e1)][self.elements.index(e2)] = 1
        for j in range(self.m-1):
            M_path = M + np.dot(M_path,M)
        M_path = np.dot(M_path,M)
        v_sans_tc_i = []
        isok = True
        for (e1,e2) in self.pairs:
            if M_path[self.elements.index(e1)][self.elements.index(e2)] != 0:
            
                return True
        return False
        
    def remove_TC(self):
        M = np.zeros((self.m,self.m))
        M_path= np.zeros((self.m,self.m))
        pairs_without_TC = []
        for (e1,e2) in self.pairs:
                M[self.elements.index(e1)][self.elements.index(e2)] = 1
        for j in range(self.m-1):
            M_path = M + np.dot(M_path,M)
        M_path = np.dot(M_path,M)
        v_sans_tc_i = []
        isok = True
        for (e1,e2) in self.pairs:
            if M_path[self.elements.index(e1)][self.elements.index(e2)] == 0:
                pairs_without_TC.append((e1,e2))
        self.pairs = pairs_without_TC
        
        
    def is_there_cycle(self) -> bool:
            M = np.zeros((self.m,self.m))
            M_path = np.zeros((self.m,self.m))
            for (e1,e2) in self.pairs:
                M[self.elements.index(e1)][self.elements.index(e2)] = 1
            for j in range(self.m):
                M_path = M + np.dot(M_path,M)
            count = 0
            for j in range(self.m):
                if M_path[j][j] != 0:
                    count += 1
            return count
            
    def missing(self,all_elements) -> List:
        missing_elem = []
        elements = self.get_elements()
        for e in all_elements:
            if e not in elements:
                missing_elem.append(e)
        return missing_elem
    
    def direct_down_i(self,i) -> List:
        if self.is_there_TC():
            self.remove_TC()
        down_i = []
        for (e1,e2) in self.pairs:
            if e1 == i:
                down_i.append(e2)
        return down_i
        
    def direct_up_i(self,i) -> List:
        if self.is_there_TC():
            self.remove_TC()
        up_i = []
        for (e1,e2) in self.pairs:
            if e2 == i:
                up_i.append(e1)
        return up_i
        
    def roots(self) -> List:
        roots_list = []
        for i in range(len(self.elements)):
            if len(self.direct_up_i(self.elements[i])) == 0:
                roots_list.append(self.elements[i])
        return roots_list
        
    def leaves(self) -> List:
        leaves_list = []
        for i in range(len(self.elements)):
            if len(self.direct_down_i(self.elements[i])) == 0:
                leaves_list.append(self.elements[i])
        return leaves_list
        
    def up(self) -> List:
        up = [{self.elements[i]} for i in range(len(self.elements))]
        for i in range(len(self.elements)):
            up[i].update(self.direct_up_i(self.elements[i]))
        parents = [len(self.direct_up_i(self.elements[i])) for i in range(len(self.elements))]
        roots_list = self.roots()
        queue = roots_list.copy()
        while queue != []:
            elem = queue.pop()
            down_elem = self.direct_down_i(elem)
            for child in down_elem:
                up[self.elements.index(child)].update(up[self.elements.index(elem)])
                parents[self.elements.index(child)] -= 1
                if parents[self.elements.index(child)] == 0:
                    queue.append(child)
        return up
        
    def up_i(self,i) -> List:
        U = self.up()
        return U[i]
        
    def down(self) -> List:
        down = [{self.elements[i]} for i in range(len(self.elements))]
        for i in range(len(self.elements)):
            down[i].update(self.direct_down_i(self.elements[i]))
        children = [len(self.direct_down_i(self.elements[i])) for i in range(len(self.elements))]
        leaves_list = self.leaves()
        queue = leaves_list.copy()
        while queue != []:
            elem = queue.pop()
            up_elem = self.direct_up_i(elem)
            for child in up_elem:
                down[self.elements.index(child)].update(down[self.elements.index(elem)])
                children[self.elements.index(child)] -= 1
                if children[self.elements.index(child)] == 0:
                    queue.append(child)
        return down
        
    def down_i(self,i) -> List:
        D = self.down()
        return D[i]
    
    def get_pairs(self) -> List:
        return self.pairs
        
    def get_height(self) -> int:
        queue = self.roots()
        parents = [len(self.direct_up_i(i)) for i in range(self.m)]
        height_i = [1]*self.m
        while queue != []:
            next_elem = queue.pop()
            h_i = height_i[next_elem]
            for child in self.direct_down_i(next_elem):
                height_i[child] = max(height_i[child],h_i+1)
                parents[child] -= 1
                if parents[child] ==0:
                    queue.append(child)
        return int(np.max(height_i))
    
    def get_density(self) -> int:
        return density(self.pairs,self.m)



class RSM(object):
    def __init__(self,center:List,pi:List,p:List):
        self.center = list(center)
        self.pi = list(pi)
        self.p = list(p)
        self.m = len(center)
        if self.m == 0:
            raise ValueError("You cannot have an empty reference ranking")
        if len(p) != self.m:
            raise ValueError("The probability vector p must have length m")
        if len(pi) != self.m or len(pi[0]) != self.m:
            raise ValueError("The probability matrix pi must have shape (m,m)")
        self.item_to_rank = {item: rank for rank, item in enumerate(self.center)}
        if np.array([p_i > 1 or p_i < 0 for p_i in p]).any():
            raise ValueError("Every element in p must be between 0 and 1")
        for i in range(0,self.m):
            for j in range(0,self.m):
                if pi[i][j] < 0:
                    raise ValueError("Weights cannot be negative")
                elif j >= self.m-i and pi[i][j] != 0:
                    raise ValueError("For all j >= m-i, pi[i,j] must be equal to 0")
        
    def __str__(self):
        return "Poset(center="+str(self.center)+", pi="+str(self.pi)+", p="+str(self.p)+")"
    
    def set_p(self,p):
        if len(p) != self.m:
            raise ValueError("The probability vector p must have length m")
        if np.array([p_i > 1 or p_i < 0 for p_i in p]).any():
            raise ValueError("Every element in p must be between 0 and 1")
        self.p = list(p)
    def set_uniform_p(self,p):
        if p> 1 or p <0:
            raise ValueError("Every element in p must be between 0 and 1")
        self.p = [p]*self.m
        
    def set_pi(self,pi):
        if len(pi) != self.m or len(pi[0]) != self.m:
            raise ValueError("The probability matrix pi must have shape (m,m)")
        self.pi = list(pi)
    
    def set_center(self,center):
        if len(center) != self.m:
            raise ValueError("The reference ranking must have length m")
        self.center = list(center)
        self.item_to_rank = {item: rank for rank, item in enumerate(self.center)}
    
    def get_p(self) -> List:
        return self.p
    
    def get_pi(self) -> List:
        pi_alt = self.pi.copy()
        for i in range(self.m):
            Z_i = np.sum(pi_alt[i])
            for j in range(self.m):
                pi_alt[i][j] = pi_alt[i][j]/Z_i
        return pi_alt
    
    def get_center(self) -> List:
        return self.center
    
    def get_candidates(self) -> int:
        return self.m
    
    def get_prob_i_j(self,i,j) -> float:
        return self.pi[i][j]/np.sum(self.pi[i])
    
    def get_p_i(self,i) -> float:
        return self.p[i]
    
    def calculate_kendall_tau_distance(self, permutation) -> int:
        dist = 0
        for i, e_i in enumerate(permutation[:-1]):
            for e_j in permutation[i:]:
                if self.get_rank_of_item(e_i) > self.get_rank_of_item(e_j):
                    dist +=1 

        return dist

    def sample_a_ranking(self) -> List:
        ranking = []
        remaining_candidates = self.center.copy()

        I = [i for i in range(self.m)]
        for step in range(self.m):
            sample_index = choices(I[:self.m-step], weights=self.pi[step][:self.m-step])[0]
            next_candidate = remaining_candidates.pop(sample_index)
            ranking.append(next_candidate)

        return ranking
        
    def sample_pairs(self) -> List:
        children = [[] for i in range(self.m)]
        parents = [set() for i in range(self.m)]
        I = [i for i in range(self.m)]
        remaining_candidates = self.center.copy()

        for step in range(self.m-1):
            sample_index = choices(I[:self.m-step], weights=self.pi[step][:self.m-step])[0]
            next_candidate = remaining_candidates.pop(sample_index)
            for k in range(self.m-step-1):
                c = remaining_candidates[k]
                rand = np.random.rand()
                if rand <= self.p[step]:
                    children[next_candidate].append(c)
                    parents[c].update({next_candidate})
                    parents[c].update(parents[next_candidate])
            for parent in parents[next_candidate]:
                l = len(children[parent])
                for j in range(1,l+1):
                    j2 = l-j
                    if children[parent][j2] in children[next_candidate]:
                        children[parent].pop(j2)
        partial_order = []
        for i in range(self.m):
            for c in children[i]:
                partial_order.append((i,c))
        return partial_order
        
    def sample_pairs_with_TC(self) -> List:
        children = [[] for i in range(self.m)]
        I = [i for i in range(self.m)]
        remaining_candidates = self.center.copy()

        for step in range(self.m-1):
            sample_index = choices(I[:self.m-step], weights=self.pi[step][:self.m-step])[0]
            next_candidate = remaining_candidates.pop(sample_index)
            for k in range(self.m-step-1):
                c = remaining_candidates[k]
                rand = np.random.rand()
                if rand <= self.p[step]:
                    children[next_candidate].append(c)
        partial_order = []
        for i in range(self.m):
            for c in children[i]:
                partial_order.append((i,c))
        return partial_order
        
    def sample_a_poset(self) -> Poset:
        pairs = []
        while pairs == []:
            pairs = self.sample_pairs()
        return Poset(pairs)
        
    def sample_a_poset_with_TC(self) -> Poset:
        return Poset(self.sample_pairs_with_TC())
        
    def generate_a_population(self,n,tc=False) -> List:
        pop = []
        for i in range(n):
            pairs = []
            try_count = 0
            while pairs == []:
                if tc:
                    pairs = self.sample_pairs_with_TC()
                else:
                    pairs = self.sample_pairs()
                try_count += 1
                if try_count == 10:
                    raise ValueError("Too many empty preferences")
            pop.append(pairs)
        return pop
        
    def set_topk(self,k):
        if k < 0 or k > self.m:
            raise ValueError("k must be between 0 and m")
        self.p = [1]*k + [0]*(self.m-k)
    
    def pi_linear(self,k):
        if k < 0 or k > self.m:
            raise ValueError("k must be between 0 and m")
        for i in range(k):
            new_row = [1]*(self.m-i) + [0]*i
            self.pi[i] = new_row.copy()
        
    def set_linear(self):
        self.p = [1]*self.m
    
    def shuffle_center(self):
        np.random.shuffle(self.center)
    
    def drop_k(self,k):
        self.p[k:] = self.p[:(self.m-k)]
        self.p[:k] = [0]*k
        self.pi_linear(k)
        
    def set_random_p(self):
        self.p = [np.random.rand() for i in range(self.m)]
        
    def proba_rank(self):
        M = np.zeros((self.m,self.m))
        self.normalize_pi()
        M_proba = solve_proba_rank(self.pi,self.m)
        for i in range(self.m):
            for j in range(self.m):
                M[self.center[i]][self.center[j]] = M_proba[i][j]
        return M
        
    def proba_preference(self):
        M = np.zeros((self.m,self.m))
        self.normalize_pi()
        M_proba = solve_pairs(self.pi,self.m)

        for i in range(self.m):
            for j in range(self.m):
                M[self.center[i]][self.center[j]] = M_proba[i][j]
        
        return M
        
    def proba_pairs(self):
        self.normalize_pi()
        proba_brut = solve_poset(self.pi,self.m)
        p = self.p
        M = np.zeros((self.m,self.m))
        for i in range(self.m):
            for j in range(self.m):
                sum = 0
                for k in range(self.m):
                    sum += proba_brut[k][i][j]*p[k]
                M[self.center[i]][self.center[j]] = sum
        return M
    
    def normalize_pi(self):
        for i in range(self.m):
            sum_i = np.sum(self.pi[i])
            
            self.pi[i] = self.pi[i]/sum_i

class Mallows_RSM(RSM):
    def __init__(self,center:List,phi:float,p:List=[]):
        self.center = list(center)
        self.phi = phi
        self.m = len(center)
        self.pi = self.q_mallows()
        
        if self.m == 0:
            raise ValueError("You cannot have an empty reference ranking")
        if p == []:
            self.p = [1]*self.m
        elif len(p) != self.m:
            raise ValueError("The probability vector p must have length m")
        else:
            self.p = list(p)
            
        if phi>1 or phi < 0:
            raise ValueError("Phi must be between 0 and 1")
       
    def q_mallows(self):
        row = [0 for i in range(self.m)]
        pi = []
        phi_i = 1
        for i in range(self.m):
            row[i] = phi_i
            phi_i *= self.phi
            row_i = row.copy()
            pi = [row_i] + pi
        return pi

    def calculate_normalization_constant(self) -> float:
        try:
            norm = (1 - self.phi) ** (-self.m)
            phi_i = self.phi
            for i in range(1, self.m + 1):
                norm *= (1 - phi_i)
                phi_i *= self.phi
        except ZeroDivisionError:
            norm = factorial(self.m)
        return norm

    def calculate_prob_by_distance(self, distance):
        return (self.phi ** distance) / self.normalization_constant


    def calculate_prob_of_permutation(self, permutation):
        dist = self.calculate_kendall_tau_distance(permutation)
        return self.calculate_prob_by_distance(dist)
    
    def get_phi(self):
        return self.phi
    
    def set_phi(self,phi):
        if phi>1 or phi < 0:
            raise ValueError("Phi must be between 0 and 1")
        self.phi = phi
        self.pi = self.q_mallows()


class drop_cand(object):
    def __init__(self,center:List,phi:float=1):
        self.center = list(center)
        self.m = len(center)
        self.phi = phi
        self.mallow = Mallows(self.center,self.phi)
        
    def sample_pairs(self,kmin:int=0,kmax:int=-1):
        if kmin > self.m - 1 or kmin <0:
            raise ValueError("Please choose a value of k min between 0 and m-1")
        if kmax == -1:
            kmax = self.m - 1
        elif kmax > self.m - 1 or kmax <0:
            raise ValueError("Please choose a value of k max between 0 and m-1")
        elif kmin > kmax:
            raise ValueError("kmax must be greater than kmin")
        
        r = self.mallow.sample_a_ranking()
        k = np.random.randint(kmin,kmax)
        keeped_list=list(np.random.choice(list(range(0, self.m)), self.m-k, replace=False))
        pairs = []
        for i in range(len(keeped_list)-1):
            pairs.append((r[keeped_list[i]],r[keeped_list[i+1]]))
        return pairs
    
    def sample_poset(self,kmin:int=0,kmax:int=-1):
        return Poset(self.sample_pairs(kmin,kmax))
        
    def get_phi(self):
        return self.phi
        
    def get_center(self):
        return self.center