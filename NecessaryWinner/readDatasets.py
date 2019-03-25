import tools


def read_dataset(name,n,m,i):
    voters = 100000
    f = open("C:\\Users\\Theo Delemazure\\Documents\\StageNY\\datasets\\"+name+"-"+str(voters)+"-"+str(m)+".txt",'r')
    l1 = f.read().split('\n')
    votes = []
    l2 = l1[i].split('*')
    for i in range(n):
        e2 = l2[i]
        if e2 != "":
            ssvotes = []
            l3 = e2.split('-')
            for e3 in l3:
                ssvotes.append(tuple(map(int,e3.strip("()").split(","))))
            votes.append(ssvotes)
        else:
            votes.append([])
    f.close()
    return votes
