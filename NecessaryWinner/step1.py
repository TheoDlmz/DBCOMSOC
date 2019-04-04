import tools

def Order2Up(P,m): #O(max(|P|,m)x|Up[u]) -> O(m²)
    Up = [[i] for i in range(m)]
    is_roots = [1 for i in range(m)]
    childs = [[] for i in range(m)]
    parents = [0 for i in range(m)]
    for k in P: #O(|P|)
        (n1,n2) = k
        is_roots[n2] = 0
        childs[n1].append(n2)
        parents[n2] +=1
    queue = []
    for i in range(m): #O(m)
        if is_roots[i] == 1:
            queue.append(i)
    while queue != []: #m times
        u = queue.pop() #O(1)
        Up[u] = list(set(Up[u])) #O(|Up[u]|) 
        for e in childs[u]:  # |P| times
            Up[e].extend(Up[u]) #O(|Up[u]|)
            parents[e] -= 1
            if parents[e] == 0:
                queue.append(e)
    return Up
    
def Up2Down(Up,m,ldown): #m x |Up[u]| -> O(m²)
    Down = [[] for i in range(m)]
    
    for i in range(m):
        for j in Up[i]:
            ldown[j] += 1
            Down[j].append(i)
    return Down
       
    
def Step1(Profile,m): #O(nm²)
    D = []
    U = []
    ldown = [0 for i in range(m)]
    for P in Profile: #n
        u = Order2Up(P,m)
        U.append(u)
        d = Up2Down(u,m,ldown)
        D.append(d)
    return D,U,ldown
