##IMPORTS 

from collections import defaultdict 
import numpy as np

##FORD FULKERSON ALGO

#This class represents a directed graph using adjacency matrix representation 
class Graph: 
    def __init__(self,graph): 
        self.graph = graph # residual graph 
        self. ROW = len(graph) 
        #self.COL = len(gr[0]) 
        

    def BFS(self,s, t, parent): 

        # Mark all the vertices as not visited 
        visited =[False]*(self.ROW) 
        
        # Create a queue for BFS 
        queue=[] 
        
        # Mark the source node as visited and enqueue it 
        queue.append(s) 
        visited[s] = True
        
        # Standard BFS Loop 
        while queue: 

            #Dequeue a vertex from queue and print it 
            u = queue.pop(0) 
        
            # Get all adjacent vertices of the dequeued vertex u 
            # If a adjacent has not been visited, then mark it 
            # visited and enqueue it 
            for ind, val in enumerate(self.graph[u]): 
                if visited[ind] == False and val > 0 : 
                    queue.append(ind) 
                    visited[ind] = True
                    parent[ind] = u 

        # If we reached sink in BFS starting from source, then return 
        # true, else false 
        return True if visited[t] else False
        
    
    # Returns tne maximum flow from s to t in the given graph 
    def FordFulkerson(self, source, sink): 

        # This array is filled by BFS and to store path 
        parent = [-1]*(self.ROW) 

        max_flow = 0 # There is no flow initially 

        # Augment the flow while there is path from source to sink 
        while self.BFS(source, sink, parent) : 

        # Find minimum residual capacity of the edges along the 
            # path filled by BFS. Or we can say find the maximum flow 
            # through the path found. 
            path_flow = float("Inf") 
            s = sink 
            while(s != source): 
                path_flow = min (path_flow, self.graph[parent[s]][s]) 
                s = parent[s] 
            
            # Add path flow to overall flow 
            max_flow += path_flow 

            # update residual capacities of the edges and reverse edges 
            # along the path 
            v = sink 
            while(v != source): 
                u = parent[v] 
                self.graph[u][v] -= path_flow 
                self.graph[v][u] += path_flow 
                v = parent[v] 

        return max_flow 

#This code is contributed by Neelam Yadav (found the implementation on internet)

###Algorithm for possible winner under plurality


def votes2posiblewinnerB(votes,m):
    n = len(votes)
    matrixRank2 = []
    for i in range(n):
        seen = [1]*m
        v = votes[i]
        for x in v:
            (a,b) =x
            if (seen[b]==1):
                seen[b] = 0
        matrixRank2.append(seen)
    return matrixRank2

def pwStep2(matrixRank2,m):
    #pw2m = [2**i for i in range(m)]
    dico = dict()
    tab = []
    numb = []
    i = 0
    for s in matrixRank2:
        #v = 0
        #for i in range(m):
        #    v += pw2m*s[i]
        if str(s) in dico.keys():
            numb[dico[str(s)]] += 1
        else:
            numb.append(1)
            tab.append(s)
            dico[str(s)] = i
            i+=1
    return tab,numb
        
    
def build_matrixB(score,N,M,m,c,query):
    P1 = len(N)
    if query == []:
        query = [1 for i in range(m)]
    size = P1 + m + 1
    matrix = np.zeros((size,size))
    for i in range(P1):
        matrix[0,i+1] = N[i]
        matrix[i+1,1+P1:size-1] = M[i]
    for i in range(m-1):
        matrix[size-i-1,size-1] = score-query[i]
    return matrix,size

def pretest(score,N,M,m,maxflow):
    oth_score = np.array([0 for i in range(m-1)])
    for i in range(len(M)):
        oth_score += M[i]
    argsorted = np.argsort(oth_score)
    for i in range(len(argsorted)):
        e = argsorted[i]
        if oth_score[e] < 0:
            return False
        suppr = min(score-1,oth_score[e])
        maxflow -= suppr
        if maxflow <= 0:
            print("yes",i,e,oth_score)
            return True
        else:
            suppr_ok = 0
            j = 0
            while suppr_ok < suppr:
                if M[j][e] > 0:
                    oth_score = [oth_score[k] - min(M[j][k],suppr-suppr_ok) for k in range(m-1)]
                    suppr_ok +=M[j][e]
                j += 1
    print(oth_score)
    return False

                    

def possibleWinnerB(t,n,m,c,q=[]):
    
    M= []
    score = 0
    N = []
    maxwanted = 0
    for i in range(len(n)):
        if t[i][c] == 1:
            score += n[i]
        else:
            l = t[i].copy()
            l.pop(c)
            l = np.array(l)*n[i]
            maxwanted += n[i]
            N.append(n[i])
            M.append(l)
    if score>maxwanted:
        return [c]
    #if score >= maxwanted/10:
     #   if pretest(score,N,M,m,maxwanted):
      #      return [c]
    M,size = build_matrixB(score,N,M,m,c,[])
    g = Graph(M)
    source = 0
    sink = size-1
    maxflow = g.FordFulkerson(source, sink)
    if maxflow >= maxwanted:
        return [c]
    else:
        return []


def isTherePossibleWinnerB(votes,m):
    possible = votes2posiblewinnerB(votes,m)
    t,n = pwStep2(possible,m)
    set = []
    for c in range(m):
        set += possibleWinnerB(t,n,m,c)
    print("The set of possible winner is ",set)
    return set




##Test



def testPW(n,m,dataset):
    print("START")
    s1,s2=0,0
    for i in range(10):
        votes = read_dataset(dataset,n,m,i)
        a = time.time()
        nw = isTherePossibleWinnerB(votes,m)
        b = time.time()
        s2 += b-a
        print(i,"(",(b-a),")")
    print(m," x ",n," : ",s2/(10))
    print("END")