import mallows

def get_partial_order(m,k,complete_list): # Select k elem on an m list
    removed_list=list(np.random.choice(list(range(0, m)), m-k, replace=False))
    partial_list = [x for x in complete_list if x not in removed_list]
    return partial_list


def calculateMatrix(m,phi):

        pij = []
        for i in range(m):
            pi = [phi ** (i - j-1) for j in range(i)]
            pij.append(pi)

        return pij

def create_dataset_split(n,m,phi,psi,k,sigma=True):
    Id = list(range(0, m))
    if sigma:
        center = np.random.permutation(Id).tolist()
    else:
        center = list(range(0, m))
    mallows = Mallows(center, phi)
    M = calculateMatrix(m,psi)
    votes = []
    for _ in range(n):
        pairs = []
        removed_list=list(np.random.choice(Id[1:], m-k-1, replace=False))
        r = mallows.sample_a_permutation()
        for i in removed_list:
            j = choices(Id[:i], weights=M[i])[0]
            pairs.append((r[j],r[i]))
        votes.append(pairs)
    return votes


def create_dataset_linear(n,m,phi,k,sigma=True):
    Id = list(range(0, m))
    if sigma:
        center = np.random.permutation(Id).tolist()
    else:
        center = list(range(0, m))
    mallows = Mallows(center, phi)
    votes = []
    for _ in range(n):
        r = mallows.sample_a_permutation()
        partial_list = get_partial_order(m,(m-k),r)
        pairwise = []
        for j in range(len(partial_list)-1):
            pairwise.append((partial_list[j],partial_list[j+1]))
        votes.append(pairwise)
    return votes
    


def save_datasets(candidates,datasetName):
    n = 100000
    for c in candidates:
        f =  open("C:\\Users\\Theo Delemazure\\Documents\\StageNY\\datasets\\"+datasetName+"-"+str(n)+"-"+str(c)+".txt",'a')
        print(c,"candidats")
        for i in range(10):
            #k = (i%int(np.sqrt(c))+1)
            #v = create_dataset_split(n,c,0.3,k)
            for j in range(len(v)):
                for k in range(len(v[j])):
                    f.write(str(v[j][k]))
                    if k < len(v[j])-1:
                        f.write('-')
                if j < len(v)-1:
                    f.write("*")
            if i< 9:
                f.write("\n")
        f.close()
        
