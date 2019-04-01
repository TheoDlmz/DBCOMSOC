import tools


def Step1_centerized(dico,Profile,m): #O(nm²)
    for P in Profile: #n
        if str(P) in dico.keys():
            dico[str(P)][0] += 1
        else:
            u = Order2Up(P,m)
            d = Up2Down(u,m)
            dico[str(P)] = [1,u,d]
    
def Step3_borda_centerized(c,w,dico,m,l=[]): #O(nm)
    Sw = 0
    Sc = 0
    for key in dico.keys(): #n
        [s,U,D] = dico[key]
        #print(s)
        if c in U[w]: #O(|U[i,w]|)
            block_size = intersect(D[c],U[w]) #O(1)
            Sc += block_size*s
        else:
            Sw += (m-len(U[w]))*s #O(1)
            Sc +=  (len(D[c])-1)*s #O(1)
    if Sw == Sc:
        l.append(w)
    return (Sw <= Sc)

def Step2_borda_centerized(c,U,D,m): #O(nm²)
    for w in range(m): #m
        if c != w:
            if not(Step3_borda_centerized(c,w,dico,m)):
                return False
        return True

    
    
def isThereNcW_bordaC(Profile,m): #O(nm²)
    current = 0
    dico = dict()
    Step1_centerized(dico,Profile,m)
    list_to_test = []
    for w in range(1,m): 
        v = Step3_borda_centerized(current,w,dico,m,list_to_test)
        if not(v):
            current = w
    i = 0
    for w in range(current):
        i+=1
        v = Step3_bordaC(current,w,dico,m)
        if not(v):
            break
    ncw = []
    if i == current:
        ncw.append(current)
    for w in list_to_test:
        v = Step2_borda_centerized(w,dico,m)
        if v:
            ncw.append(w)
    if len(ncw) ==0:
        return "There is no co-necessary winner"
    return "The necessary co-winners are "+str(ncw)