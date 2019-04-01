import tools
import step1

def precompute_score(s,m):
    M = np.zeros((m-1,m-1,m-1)) #min-1,max-1,block size
    for i in range(m):
        for j in range(i+1,m):
            M[i][j-1][j-i-1] = s[i] - s[j]
    for k in range(1,m):
        for i in range(m):
            for j in range(i+k+1,m):
                M[i][j-1][j-i-k-1] = min(s[i+k]-s[j],M[i][j-2][j-i-k-1])
    return M

def Step3_score(s,M,c,w,U,D,m):
    n = len(U)
    Sw = 0
    Sc = 0
    for i in range(n):
        if c in U[i][w]:
            min = s[0]-s[m-1]+1
            armin = -1
            block_size = intersect(D[i][c],U[i][w])
            Sc += M[len(U[i][c])-1,m-len(D[i][w])-1,block_size-1]
        else:
            Sw += s[len(U[i][w])-1]
            Sc += s[m - len(D[i][c])]
    return (Sw <= Sc)


    
def isThereNW_score(s,Profile,m):
    current = 0
    M = precompute_score(s,m)
    D,U = Step1(Profile,m)
    for w in range(1,m):
        v = Step3_score(s,M,current,w,U,D,m)
        if not(v):
            current = w
    for w in range(current): 
        v = Step3_score(s,M,current,w,U,D,m)
        if not(v):
            return "There is no necessary winner"
    return "The necessary winner is "+str(current)