import tools
import step1


def Step3_kApproval(k,c,w,U,D,m): 
    n = len(U)
    Sw = 0
    Sc = 0
    for i in range(n):
        if c in U[i][w]:
            maxpos = (m-len(D[i][c]))
            minpos = len(U[i][c])
            block_size = intersect(D[i][c],U[i][w])
            if (maxpos < k) and ((minpos + block_size) > k):
                Sc += 1
        elif (len(U[i][w]) <= k) and ((m-len(D[i][c])) >= k):
            Sw +=1
    print(Sw,Sc)

    
def isThereNW_kApproval(k,Profile,m):
    current = 0
    D,U,_ = Step1(Profile,m)
    for w in range(1,m):
        v = Step3_kApproval(k,current,w,U,D,m)
        if not(v):
            current = w
    for w in range(0,current):
        v = Step3_kApproval(k,current,w,U,D,m)
        if not(v):
            return "There is no necessary winner"
    return "The necessary winner is "+str(current)