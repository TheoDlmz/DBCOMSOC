import tools
import NWBorda
import detect
import NWBordaSubranking
import NWBordaMultipleSubrankings
import NWBordaOnlySplits
import NWBordaOnlyMerges
import readDatasets

def testBorda(n,m,dataset):
    print("START")
    s1,s2=0,0
    for i in range(10):
        votes = read_dataset(dataset,n,m,i)
        a = time.time()
        d = detect(votes,m)
        b = time.time()
        if d == 0:
            c = time.time()
            nw = isThereNcW_borda(votes,m)
            d = time.time()
        elif d == 1:
            c = time.time()
            nw = isThereNcW_borda_sub(votes,m)
            d = time.time()
        elif d == 2:
            c = time.time()
            nw = isThereNcW_borda_mulsub(votes,m)
            d = time.time()
        elif d == 3:
            c = time.time()
            nw = isThereNcW_borda_merge(votes,m)
            d = time.time()
        if d == 4:
            c = time.time()
            nw = isThereNcW_borda_split(votes,m)
            d = time.time()
        s2 += d-c
        s1 += b-a
        print(i,".",nw,"(",(d-c),")")
    print(m," x ",n)
    print("Detection : ",s1/10)
    print("Computation : ",s2/10)
    print("END")