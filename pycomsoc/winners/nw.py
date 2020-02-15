# Library pyCOMSOC for computational social choice
# New York University 2019
# All the function below are based on the paper by Xia and Conitzer
# Algorithms for Necessary Winner computation on partial orders


## Imports

import numpy as np
import random
from multiprocessing import Pool
import time

## Tools

def __intersect__(a, b):
    return len(list(set(a) & set(b))) - 1 
    
def __incrementMaxScore__(upLength,rule):
    # Borda
    if rule == 'b':
        return upLength-1
    # K-approval
    elif rule[0] == 'k':
        kIndex = int(rule[1:])
        if upLength > kIndex:                  
            return 1
        else:
            return 0
    else:
        # General positional scoring rule
        return rule[0]-rule[upLength-1]
    
## Step 1 : Compute Up and Down for every candidate and every partial orders

#Compute "Up" for the general case
def __upGeneral(children,parents,roots,m,maxScore,rule='b'): 
    # Initialization
    up = [[i] for i in range(m)]
    parentsToSee = [len(p) for p in parents]
    # Initialize the queue with the roots of the graph
    # for which the up set only contains themselves
    queue = roots.copy()
    
    # True if some candidate has >1 children in the preference graph
    isMerge = (np.max([len(c) for c in children]) > 1)
    
    # True if some candidate has >1 parents in the preference graph
    isSplit = (np.max([len(p) for p in parents]) > 1)    
              
    # BFS algorithm to compute "Up" for every candidate
    if isSplit and isMerge:                         #If the graph is not a tree...
        while queue != []: 
            u = queue.pop() 
            up[u] = list(set(up[u]))                #...then we have to check for duplicate candidates
            
            # Update maxscore depending on the rule
            maxScore[u] += __incrementMaxScore__(len(up[u]),rule)
                
            for e in children[u]:
                up[e].extend(up[u]) 
                parentsToSee[e] -= 1
                if parentsToSee[e] == 0:
                    queue.append(e)
    else:
        while queue != []: 
            u = queue.pop() 
            maxScore[u] += __incrementMaxScore__(len(up[u]),rule)
            for e in children[u]: 
                up[e].extend(up[u]) 
                parentsToSee[e] -= 1
                if parentsToSee[e] == 0:
                    queue.append(e)
    
    return up 


#Compute "Up" for the Partitioned case
def __upPartitioned(children,roots,m,maxScore,rule='b'): 
    #Initialization
    blockNumber = [-1 for i in range(m)]            #Block's number of the ith candidate
                                                    #Block "-1" means not ranked
    ranksBlocks = [0]                               #Minimum rank of ith block
    currentRank = 0
    queue = roots.copy()                            #Initialize the queue with the root candidates
    
    #Algorithm
    while queue != []:
        candidateInBlock = 0
        for candidate in queue:
            blockNumber[candidate] = currentRank
            
            # Update maxscore depending on the rule
            maxScore[candidate] += __incrementMaxScore__(ranksBlocks[-1]+1,rule)
            
            candidateInBlock += 1                   #Count the number of candidate in the block 
        candidate0 = queue[0]
        queue = children[candidate0].copy()         #Get the next block
        ranksBlocks.append(ranksBlocks[-1]+candidateInBlock)
        currentRank += 1
        
    return blockNumber,ranksBlocks


#Compute "Up" for the Linear case
def __upLinear(children,roots,m,maxScore,rule='b'):
    #Initialization
    candRank = [-1 for i in range(m)]               #Rank of ith candidatz
                                                    #Rank "-1" means not ranked
    currentRank = 0
    candidate = roots[0]
    
    #Algorithm
    while candidate != -1:
        candRank[candidate] = currentRank     
               
        # Update maxscore depending on the rule
        maxScore[candidate] += __incrementMaxScore__(currentRank+1,rule)
        
        currentRank +=1
        if len(children[candidate]) == 0:                #Stop when no children in the graph
            candidate = -1
        else:
            candidate = children[candidate][0]
    candRank.append(currentRank)                    #Add the length of the order at the end of candRank
    
    return candRank
    
        
#Compute "Up" for the Multilinear case
def upMultilinear(children,roots,m,maxScore,rule='b'):
    #Initialization
    candRank = [(-1,-1) for i in range(m)]          #(orderNumber,rankInLinearOrder)
    orderLength = []                                #Length of the ith linear order     
    
    #Algorithm                                
    for i in range(len(roots)):                     #Apply the upBordaLinear on each linear order
        currentRank = 0
        candidate = roots[i]
        while candidate != -1:
            candRank[candidate] = (i,currentRank)    
                       
            # Update maxscore depending on the rule
            maxScore[candidate] += __incrementMaxScore__(currentRank+1,rule)
            
            currentRank +=1
            if len(children[candidate]) == 0:            #If it is a leaf...
                candidate = -1                           #...then stop
            else:
                candidate = children[candidate][0]
        orderLength.append(currentRank)
        
    return candRank,orderLength

#General algorithm for the computation of "Up" and "Down"
def upDown(population,m,rule='b',verbose=False,optimUpActivated=True,optimCandActivated=True):
    #population : array with all the partial orders
    #m : nb of candidates
    #optimUpActivated : is the computation of Up and Down optimized for linear, multilinear and partitioned preferences
    #optimCandActivated : is the list of candidate to test optimized
       
    n = len(population)
    
    maxScore = [0 for i in range(m)]                #maximum score of every cand
    
    blockNumberP = []                               #Informations for partitioned preferences
    ranksBlocksP = []
    candRankL = []                                  #Informations for linear orders
    candRankM = []                                  #Informations for multilinear orders
    orderLengthM = []
    upList = []                                     #Informations for general orders
    generalCaseOrders = []
    countEmpty = 0
    
    #Loop over every partial preferences
    for p in range(n):
        pairs = population[p]
        
        #Empty preferences = Empty linear order
        if pairs == []:
            candRankL.append([-1]*m)
            countEmpty += 1
            
        else:
            #We get parents and children of every candidate in the preference graph
            parents = [[] for i in range(m)]
            children = [[] for i in range(m)]      
            for (a,b) in pairs:
                parents[b].append(a)
                children[a].append(b)
                
            
            isLinear = True                         #Is the order linear ?
            roots = []                              #Roots of preference graph
            leaves = []                             #Leaves of preference graph
                
            #Iterate over candidates
            for i in range(m):
                #If some candidate have > 1 parents or > 1 children then
                #this is not a linear order
                if len(parents[i]) > 1 or len(children[i]) > 1:
                    isLinear = False

                #If candidate i does not have children then it is a leaf
                if len(children[i]) == 0  and len(parents[i]) > 0:
                    leaves.append(i)
                #If candidat i does not have parents then it is a children
                elif len(parents[i]) == 0  and len(children[i]) > 0:
                    roots.append(i)
            
            #If the first optimization is disabled, then use the general case
            #algorithm for every order
            if not(optimUpActivated):
                upList_i = __upGeneral(children,parents,roots,m,maxScore,rule)
                upList.append(upList_i)
                generalCaseOrders.append(pairs)
            
            #Otherwise, if the order is linear or multilinear, 
            #use special algorithms
            elif len(roots) == len(leaves) and isLinear:
                #Linear case
                if len(roots) == 1:
                    candRankL_i = __upLinear(children,roots,m,maxScore,rule)
                    candRankL.append(candRankL_i)
                    
                #MultiLinear case
                else:
                    candRankM_i,orderLengthM_i = upMultilinear(children,roots,m,maxScore,rule)
                    candRankM.append(candRankM_i)
                    orderLengthM.append(orderLengthM_i)
            
            #Then test if this is partitioned preferences
            else:
                
                blockNumber = [-1 for i in range(m)]    #Block number of ith candidate
                currentBlockNb = 0                      #Number of the current block
                currentBlock = roots.copy()             #Candidates in the curent block
                
                sum = 0
                last = 0
                while currentBlock != []:

                    temp = len(currentBlock)
                    sum += last*temp
                    last = temp
                    #Set block number of candidates
                    for cand in currentBlock:
                        blockNumber[cand] = currentBlockNb
                    #Increment block number
                    currentBlockNb += 1
                    #Get next block
                    currentBlock = children[currentBlock[0]].copy()
               
                #The order is partitioned iff :
                #   1. every candidate of ith block is connected to every candidate
                #      of i+1th block (ie sum = len(pairs))
                #   2. every children of candidate in ith block is in i+1th block
                
                cont = False
                if sum == (len(pairs)):
                    cont = True
                    for cand in range(m):
                        for x in children[cand]:
                            if blockNumber[x] != blockNumber[cand] + 1:
                                cont = False
                                break
                            if not(cont):
                                break
                
                #Use partitioned preferences algorithm
                if cont:
                    blockNumberP_i,ranksBlocksP_i = __upPartitioned(children,roots,m,maxScore,rule)
                    blockNumberP.append(blockNumberP_i)
                    ranksBlocksP.append(ranksBlocksP_i)
                
                #Use general case algorithm
                else:
                    upList_i = __upGeneral(children,parents,roots,m,maxScore,rule)
                    upList.append(upList_i)
                    generalCaseOrders.append(pairs)
    
    #Optimize the list of candidate we want to test on next step (__competitions)
    if optimCandActivated:
        #TH : Only candidate(s) with the best maximal score can be necessary winner
        #     (we take the min because we saved m-bestScore insted of bestScore)
        bestScore = min(maxScore)
        candToTest = []
        for i in range(m):
            if maxScore[i] == bestScore:
                candToTest.append(i)
        nbToTest = len(candToTest)
        
        #We compute the down only for candidate to test
        downList = [[] for i in range(nbToTest)]
        
        #We need to compute down only for orders in the general case
        for pairs in generalCaseOrders:
            
            #get children of every candidate
            children = [[] for i in range(m)]
            for (a,b) in pairs:
                children[a].append(b)
            
            #We look to the down set of every candidate to test
            #This is the same algorithm than for Up
            for j in range(nbToTest):
                cand = candToTest[j]
                visited = [False for i in range(m)]
                downCand = [cand]
                queue = children[cand].copy()
                while queue != []:
                    newCand = queue.pop()
                    downCand.append(newCand)
                    for newCandChild in children[newCand]:
                        if not(visited[newCandChild]):
                            visited[newCandChild] = True
                            queue.append(newCandChild)
                downList[j].append(downCand)
    else:
        #If we don't optimize, then we compute the down set of every candidate
        #usign their up set.
        candToTest = []
        downList = [[] for i in range(m)]
        for i in range(len(upList)):
            upList_i = upList[i]
            for j_1 in range(m):
                downList[j_1].append([])
            for j_1 in range(m):
                for j_2 in upList_i[j_1]:
                    downList[j_2][i].append(j_1)
    
    #Print if verbose activated
    if verbose:
        print("Empty : "+str(countEmpty)+"\nLinear : "+str(len(candRankL)-countEmpty)+"\nMultilinear : "+str(len(candRankM))+"\nPartitioned : "+str(len(blockNumberP))+"\nGeneral case : "+str(len(upList)))
        
    #Return informations useful to the next step :
    #   1. Informations on partial orders
    #   2. List of candidates to test
    #   3. The best score of each candidate.

    return [[upList,downList],[blockNumberP,ranksBlocksP],[candRankL],[candRankM,orderLengthM]],candToTest,maxScore


## Parallelized version of the Step 1 (computation of Up and Down)


# We can speed up the above algorithm by parallelizing the computation of Up and Down (step 1)
# Below is the code with the parallelized version of the algorithm.

def __upParallelized(pairs,m,indice,rule):
    parents = [[] for i in range(m)]
    children = [[] for i in range(m)]     
     
    for (a,b) in pairs:
        parents[b].append(a)
        children[a].append(b)    
                    
    maxScore = [0 for i in range(m)]
    isLinear = True
    roots = []
    leaves = []
    for i in range(m):
        if len(parents[i]) > 1 or len(children[i]) > 1:
            isLinear = False
        if len(children[i]) == 0  and len(parents[i]) > 0:
            leaves.append(i)
        elif len(parents[i]) == 0  and len(children[i]) > 0:
            roots.append(i)

    if len(roots) == len(leaves) and isLinear:
        if len(roots) == 1:
            candRankL_i = __upLinear(children,roots,m,maxScore,rule)
            return candRankL_i,0,maxScore
        
        else:
            candRankM_i,orderLengthM_i = upMultilinear(children,roots,m,maxScore,rule)
            return (candRankM_i,orderLengthM_i),1,maxScore
    
    else:
        blockNumber = [-1 for i in range(m)]
        currentBlockNb = 0
        currentBlock = roots.copy()
        sum = 0
        last =0
        while currentBlock != []:
            temp = len(currentBlock)
            sum += last*temp
            last = temp
            for cand in currentBlock:
                blockNumber[cand] = currentBlockNb
            currentBlockNb += 1
            currentBlock = children[currentBlock[0]].copy()
        cont = False
        if sum == (len(pairs)):
            cont = True
            for cand in range(m):
                for x in children[cand]:
                    if blockNumber[x] != blockNumber[cand] + 1:
                        cont = False
                        break
                    if not(cont):
                        break
        if cont:
            blockNumberP_i,ranksBlocksP_i = __upPartitioned(children,roots,m,maxScore,rule)
            return (blockNumberP_i,ranksBlocksP_i),2,maxScore
        else:
            U_i = __upGeneral(C,P,roots,m,maxScore,rule)
            # We convert U into a tuple array otherwise it takes too much time to be transfered between processes.
            a = [tuple(ui) for ui in U_i]
            ind = indice
            return (a,ind),3,maxScore
            
            
def __upParallelizedConcat(list):
    #We concatenate output from elements of one chunk
    out = []
    for (pair,m,ind,rule) in list:
        out.append(__upParallelized(pair,m,ind,rule))
    return out
            
  
def upDownParallelized(population,m,rule='b',verbose=False,process=4,chunksize=1,chunks=10):
    n = len(population)
    
    # Initialization
    maxScore = [0 for i in range(m)]
    upList = []
    blockNumberP = []
    ranksBlocksP = []
    candRankL = []
    candRankM = []
    orderLengthM = []
    pairsGeneralCase = []
    
    if process <= 0:
        raise ValueError("Number of processes should be > 0")
    
    pairs_mb = [(pair,m,i,rule) for i,pair in enumerate(Population)]
    # We divide the population into process*chunks blocks so each process work on chunks blocks
    pairs_mb_concat = [pairs_mb[(i*n)//(process*chunks):((i+1)*n)//(process*chunks)] for i in range(process*chunks)]
    with Pool(process) as p:
        out = p.map(__upParallelizedConcat,pairs_mb_concat,chunksize=chunksize)
            
    # We gather all the results together after the parallelized part
    for out_el in out:
        for out_i in out_el:
            maxScore_i = out_i[2]
            for j in range(m):
                maxScore[j] += maxScore_i[j]
            category = out_i[1]
            if category == 0:
                candRankL.append(out_i[0])
            elif category == 1:
                candRankM.append(out_i[0][0])
                orderLengthM.append(out_i[0][1])
            elif category == 2:
                blockNumberP.append(out_i[0][0])
                ranksBlocksP.append(out_i[0][1])
            else:
                upList.append(list(out_i[0][0]))
                pairsGeneralCase.append(population[out_i[0][1]])
    
    # We optimize the candidate to test
    bestScore = min(maxScore)
    candToTest = []
    for i in range(m):
        if maxScore[i] == bestScore:
            candToTest.append(i)
    nbToTest = len(candToTest)
    
    # We compute the downList for those candidates and the 
    # partial orders in the general case.
    downList = [[] for i in range(nbToTest)]
    
    for pairs in pairsGeneralCase:
        
        children = [[] for i in range(m)]
        for (a,b) in pairs:
            children[a].append(b)
            
        for j in range(nbToTest):
            cand = candToTest[j]
            visited = [False for i in range(m)]
            downCand = [cand]
            queue = children[cand].copy()
            while queue != []:
                newCand = queue.pop()
                downCand.append(newCand)
                for newCandChild in children[newCand]:
                    if not(visited[newCandChild]):
                        visited[newCandChild] = True
                        queue.append(newCandChild)
            downList[j].append(downCand)
    
    if verbose:
        print("Linear : "+str(len(candRankL))+"\nMultilinear : "+str(len(candRankM))+"\nPartitoned : "+str(len(blockNumberP))+"\nGeneral case : "+str(len(upList)))
        
    return [[upList,downList],[blockNumberP,ranksBlocksP],[candRankL],[candRankM,orderLengthM]],candToTest,maxScore




## STEP 2 : __competition, with Borda rule.

#A borda __competition between two candidates "candTested" and "opponent"
#as described in Xia and Conitzer paper (in the general case)
def __competitionBordaGeneral(candTested,opponent,upList,downList,m): 
    n = len(upList)
    scoreOpponent = 0                           #Init scores
    scoreCandTested = 0
    
    for i in range(n):
        # If candTested > opponent, then minimize their difference
        if candTested in upList[i][opponent]:
            blockSize = __intersect__(downList[i],upList[i][opponent])
            scoreCandTested += blockSize
        
        # Otherwise, set opponent > candTested and maximize their difference
        else:
            scoreOpponent += m-len(upList[i][opponent])
            scoreCandTested += len(downList[i])-1

    # Return score of the two candidates
    return scoreOpponent,scoreCandTested

#A borda __competition in the case of partitioned preferences
def __competitionBordaPartitioned(candTested,opponent,blockNumberP,ranksBlocksP,m): 
    n = len(blockNumberP)
    scoreOpponent = 0
    scoreCandTested = 0                         #Init scores
    
    for i in range(n): 
        # If block(candTested) > block(opponent) then minimize their difference
        if (blockNumberP[i][candTested] >= 0) and (blockNumberP[i][candTested] < blockNumberP[i][opponent]): 
            blockSize = ranksBlocksP[i][blockNumberP[i][opponent]] - ranksBlocksP[i][blockNumberP[i][candTested]+1] + 1
            scoreCandTested += blockSize
            
        # Otherwise
        else:
            # If candTested ranked, maximize the difference the difference with opponent
            if blockNumberP[i][candTested] != -1:
                scoreCandTested += ranksBlocksP[i][-1]-ranksBlocksP[i][blockNumberP[i][candTested]+1]
            
            # If candTested not ranked, then put it at the bottom and maximize score of opponent
            scoreOpponent += m-1-ranksBlocksP[i][max(0,blockNumberP[i][opponent])]
        
    # Return score of the two candidates
    return scoreOpponent,scoreCandTested

#A borda __competition in the case of linear preferences
def __competitionBordaLinear(candTested,opponent,candRankL,m): 
    n = len(candRankL)
    scoreOpponent = 0                           #Init scores
    scoreCandTested = 0
    
    for i in range(n):
        # If candTested > opponent, then minimize their difference
        if (candRankL[i][candTested] >= 0) and (candRankL[i][candTested] < candRankL[i][opponent]): 
            blockSize = candRankL[i][opponent] - candRankL[i][candTested] 
            scoreCandTested += blockSize
            
        # Otherwise
        else:
            #If canTested ranked, put every "free" candidate between it and the opponent
            if candRankL[i][candTested] != -1:
                scoreCandTested += candRankL[i][-1]-candRankL[i][candTested]-1
            #If candTested not ranked, put it at the bottom
            scoreOpponent += m-1-max(candRankL[i][opponent],0)
    
    # Return score of the two candidates
    return scoreOpponent,scoreCandTested


# A borda __competition on the multilinear case
def __competitionBordaMultilinear(candTested,opponent,candRankM,orderLengthM,m): 
    n = len(candRankM)
    scoreOpponent = 0                           #Init scores
    scoreCandTested = 0
    
    for i in range(n): 
        (orderNumberCandTested,rankCandTested) = candRankM[i][candTested]
        (orderNumberOpponent,rankOpponent) = candRankM[i][opponent]
        
        # If the two candidates are in the same suborder and 
        # candTested > opponent, then minimize their difference
        if (rankCandTested >= 0) and (rankCandTested < rankOpponent) and (orderNumberOpponent == orderNumberCandTested): 
            blockSize = rankOpponent - rankCandTested 
            scoreCandTested += blockSize
            
        # Otherwise
        else:
            #If canTested ranked, put every candidate not in its suborder between it and the opponent
            if orderNumberCandTested != -1:
                scoreCandTested += orderLengthM[i][orderNumberCandTested]-rankCandTested-1
            #If candTested not ranked, put it at the bottom
            scoreOpponent += m-1-max(rankOpponent,0)
            
    # Return score of the two candidates
    return scoreOpponent,scoreCandTested

# A complete Borda __competition between two candidate
def __competitionBorda(candTested,opponent,ordersInfos,candIndex,m,verbose=False):
    #Get orders infos
    [upList,downList] = ordersInfos[0]
    [blockNumberP,ranksBlocksP] = ordersInfos[1]
    [candRankL] = ordersInfos[2]
    [candRankM,orderLengthM] = ordersInfos[3]
    
    #Compute subscore for each special case
    scoreOpponentGeneralCase,scoreCandTestedGeneralCase = __competitionBordaGeneral(candTested,opponent,upList,downList[candIndex],m)
    scoreOpponentPartitioned,scoreCandTestedPartitioned = __competitionBordaPartitioned(candTested,opponent,blockNumberP,ranksBlocksP,m)
    scoreOpponentLinear,scoreCandTestedLinear = __competitionBordaLinear(candTested,opponent,candRankL,m)
    scoreOpponentMultilinear,scoreCandTestedMultilinear = __competitionBordaMultilinear(candTested,opponent,candRankM,orderLengthM,m)
    
    #Compute total scores
    scoreOpponent = scoreOpponentGeneralCase + scoreOpponentPartitioned + scoreOpponentLinear + scoreOpponentMultilinear
    scoreCandTested = scoreCandTestedGeneralCase + scoreCandTestedPartitioned + scoreCandTestedLinear + scoreCandTestedMultilinear
    
    #Print if verbose
    if verbose:
        print("Test "+str(candTested)+" ("+str(scoreCandTested)+") against "+str(opponent)+" ("+str(scoreOpponent)+")")
    
    # Return True iff candTested is always better than its opponent
    return scoreCandTested >= scoreOpponent

## General Algorithm : Borda

# The algorithm for NW and Borda rule
def borda(population,m,verbose=False,optimUpActivated=True,optimCandActivated=True,parallelized=False,process=4,chunksize=1,chunks=10):
    #Step 1 : get Up and Down (or similar order informations)
    if parallelized:
        ordersInfos,candToTest,maxScore = upDownParallelized(population,m,'b',verbose,process,chunksize,chunks)
    else:
        ordersInfos,candToTest,maxScore = upDown(population,m,'b',verbose,optimUpActivated,optimCandActivated)
    
    #Step 2 : do __competitions between candidates
    NW = []                                                 #Init list of NW
    
    #If optimization on __competitions, order the candidates so we test those which are
    #more likely to be NW first
    if optimCandActivated:
        order = np.argsort(maxScore)
        
    #Otherwise, we test every candidate
    else:
        candToTest = [i for i in range(m)]
        order = [i for i in range(m)]

    #Test ith candidate
    for i in range(len(candToTest)):
        isaNW = True
        for j in range(m):                                  #For every opponent != candTested
            if candToTest[i] != order[j]:
                if not(__competitionBorda(candToTest[i],order[j],ordersInfos,i,m,verbose=verbose)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(candToTest[i])
    
    #Return list of NW
    return NW


## Step 2 :

#A k-approval __competition between two candidates "candTested" and "opponent"
#as described in Xia and Conitzer paper (in the general case)
def __competitionKappGeneral(k,candTested,opponent,upList,downList,m): 
    n = len(upList)
    scoreOpponent = 0
    scoreCandTested = 0
    for i in range(n):
        # We compute the best ranking for the opponent
        minposOpponent = len(upList[i][opponent])
        # We compute the worst ranking for thhe candidate tested
        maxposCandTested = m-len(downList[i])+1
        if candTested in upList[i][opponent]:
            # If candTested > oponnent and rank(candTested) <= k and rank(opponent) > k,
            # Then the tested candidate win a point but not the opponent
            if maxposCandTested <= k and minposOpponent > k:
                scoreCandTested += 1
        else:
            # Otherwise, we minimize the rank of the opponent and maximize
            # the one of the candidate being tested
            if maxposCandTested <= k:
                scoreCandTested += 1
            if minposOpponent <= k:
                scoreOpponent += 1
                
    return scoreOpponent,scoreCandTested

# A k-approval __competition with partitioned preferences
def __competitionKappPartitioned(k,candTested,opponent,blockNumberP,ranksBlocksP,m): 
    n = len(blockNumberP)
    scoreOpponent = 0
    scoreCandTested = 0
    
    for i in range(n): 
        # We compute the best ranking for the opponent
        minposOpponent = ranksBlocksP[i][blockNumberP[i][opponent]] + 1
        # We compute the best ranking for the opponent
        maxposCandTested = (m-ranksBlocksP[i][-1])+ranksBlocksP[i][blockNumberP[i][candTested]+1]
        
        if blockNumberP[i][candTested] >= 0 and (blockNumberP[i][candTested] < blockNumberP[i][opponent]):
            # If candTested > oponnent and rank(candTested) <= k and rank(opponent) > k,
            # Then the tested candidate win a point but not the opponent
            if maxposCandTested <= k and minposOpponent > k:
                scoreCandTested += 1
        else:
            # Otherwise, we minimize the rank of the opponent and maximize
            # the one of the candidate being tested
            if maxposCandTested <= k and blockNumberP[i][candTested] >=0:
                scoreCandTested += 1
            if minposOpponent <= k or blockNumberP[i][opponent] < 0:
                scoreOpponent += 1
                
    return scoreOpponent,scoreCandTested
    
# A k-approval __competition with linear preferences
def __competitionKappLinear(k,candTested,opponent,candRankL,m): 
    n = len(candRankL)
    scoreOpponent = 0
    scoreCandTested = 0
    
    for i in range(n): 
        # We compute the best ranking for the opponent
        minposOpponent = max(candRankL[i][opponent],0) + 1
        # We compute the best ranking for the opponent
        maxposCandTested = (m-candRankL[i][-1])+candRankL[i][candTested]+1
        
        if candRankL[i][candTested] >= 0 and candRankL[i][candTested] < candRankL[i][opponent]:
            # If candTested > oponnent and rank(candTested) <= k and rank(opponent) > k,
            # Then the tested candidate win a point but not the opponent
            if minposOpponent > k and maxposCandTested <= k:
                scoreCandTested += 1
        else:
            # Otherwise, we minimize the rank of the opponent and maximize
            # the one of the candidate being tested
            if maxposCandTested <= k and candRankL[i][candTested] >= 0:
                scoreCandTested += 1
            if minposOpponent <= k or candRankL[i][opponent] < 0:
                scoreOpponent += 1
                
    return scoreOpponent,scoreCandTested
    


# A k-approval __competition with multilinear preferences
def __competitionKappMultilinear(k,candTested,opponent,candRankM,orderLengthM,m): 
    n = len(candRankM)
    scoreOpponent = 0
    scoreCandTested = 0
    for i in range(n): 
        (orderNumberCandTested,rankCandTested) = candRankM[i][candTested]
        (orderNumberOpponent,rankOpponent) = candRankM[i][opponent]
        
        # We compute the best ranking for the opponent
        minposOpponent = rankOpponent + 1
        # We compute the best ranking for the opponent
        maxposCandTested = m-(orderLengthM[i][orderNumberCandTested] - rankCandTested)
      
        if rankCandTested > 0 and (rankCandTested < rankOpponent and orderNumberOpponent == orderNumberCandTested):
            # If candTested > oponnent and rank(candTested) <= k and rank(opponent) > k,
            # Then the tested candidate win a point but not the opponent
            if minposOpponent > k and maxposCandTested <= k:
                scoreCandTested += 1
        else:
            # Otherwise, we minimize the rank of the opponent and maximize
            # the one of the candidate being tested
            if rankCandTested != -1 and maxposCandTested <= k:
                scoreCandTested += 1
            if rankOpponent == -1 or minposOpponent <= k:
                scoreOpponent += 1
                
    return scoreOpponent,scoreCandTested
    
# Gather results of all k-approval sub__competition between two candidates
def __competitionKapp(k,candTested,opponent,ordersInfos,candIndex,m,verbose=False):
    #Get orders infos
    [upList,downList] = ordersInfos[0]
    [blockNumberP,ranksBlocksP] = ordersInfos[1]
    [candRankL] = ordersInfos[2]
    [candRankM,orderLengthM] = ordersInfos[3]
    
    #Compute subscore for each special case
    scoreOpponentGeneralCase,scoreCandTestedGeneralCase = __competitionKappGeneral(k,candTested,opponent,upList,downList[candIndex],m)
    scoreOpponentPartitioned,scoreCandTestedPartitioned = __competitionKappPartitioned(k,candTested,opponent,blockNumberP,ranksBlocksP,m)
    scoreOpponentLinear,scoreCandTestedLinear = __competitionKappLinear(k,candTested,opponent,candRankL,m)
    scoreOpponentMultilinear,scoreCandTestedMultilinear = __competitionKappMultilinear(k,candTested,opponent,candRankM,orderLengthM,m)
    
    #Compute total scores
    scoreOpponent = scoreOpponentGeneralCase + scoreOpponentPartitioned + scoreOpponentLinear + scoreOpponentMultilinear
    scoreCandTested = scoreCandTestedGeneralCase + scoreCandTestedPartitioned + scoreCandTestedLinear + scoreCandTestedMultilinear
    
    #Print if verbose
    if verbose:
        print("Test "+str(candTested)+" ("+str(scoreCandTested)+") against "+str(opponent)+" ("+str(scoreOpponent)+")")
    
    # Return True iff candTested is always better than its opponent
    return scoreCandTested >= scoreOpponent


## General Algorithm : k-approval, plurality and veto

def kapp(population,m,k,verbose=False,optimUpActivated=True,optimCandActivated=True,parallelized=False,process=4,chunksize=1,chunks=10):
    #Step 1 : get Up and Down (or similar order informations)
    if parallelized:
        ordersInfos,candToTest,maxScore = upDownParallelized(population,m,"k"+str(k),verbose,process,chunksize,chunks)
    else:
        ordersInfos,candToTest,maxScore = upDown(population,m,"k"+str(k),verbose,optimUpActivated,optimCandActivated)
    
    #Step 2 : do __competitions between candidates
    NW = []
    if optimCandActivated:
        order = np.argsort(maxScore)
    else:
        candToTest = [i for i in range(m)]
        order = [i for i in range(m)]
        
    #Test all candidates in candToTest
    for i in range(len(candToTest)):
        isaNW = True
        for j in range(m):
            if candToTest[i] != order[j]:
                if not(__competitionKapp(k,candToTest[i],order[j],ordersInfos,i,m,verbose=verbose)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(candToTest[i])
    
    return NW
    
 
def plurality(population,m,verbose=False,optimUpActivated=False,optimCandActivated=True,parallelized=False,process=4,chunksize=1,chunks=10):
    return kapp(population,m,1,verbose,optimUpActivated,optimCandActivated,parallelized,process,chunksize,chunks)


def veto(population,m,verbose=False,optimUpActivated=False,optimCandActivated=True,parallelized=False,process=4,chunksize=1,chunks=10):
    return kapp(population,m,m-1,verbose,optimUpActivated,optimCandActivated,parallelized,process,chunksize,chunks)


## Step 2 : Any positional scoring rule

# This function __precompute the score difference for two element :
# M[i,j,k] is the score difference for block of size k whith a top of minimum rank i and a bottom of maximum rank j
# It uses dynamic programming

def precomputeScore(rule,m):
    M = np.zeros((m-1,m-1,m-1)) 
    for i in range(m):
        for j in range(i+1,m):
            M[i][j-1][j-i-1] = rule[i] - rule[j]
    for k in range(1,m):
        for i in range(m):
            for j in range(i+k+1,m):
                M[i][j-1][j-i-k-1] = min(rule[i+k]-rule[j],M[i][j-2][j-i-k-1])
    return M
    

    
def __competitionPositionalScoringRuleGeneral(rule,M,candTested,opponent,upList,downList,m,optimPreprocessing=True): 
    n = len(upList)
    scoreOpponent = 0
    scoreCandTested = 0
    
    for i in range(n):
        if candTested in upList[i][opponent]:
            # If candTested > opponent, then compute the minimal score difference between them.
            blockSize = __intersect__(downList[i],upList[i][opponent])
            if blockSize == 0:
                raise ValueError("Block size = 0")
            
            # If we use preprocessing, then use the M matrix
            if optimPreprocessing:
                M_i = len(upList[i][opponent])-blockSize-1
                M_j = m-len(downList[i])+blockSize-1
                scoreCandTested += M[M_i,M_j,blockSize-1]
            
            # Otherwise, try every position of the block between the two candidates
            else:
                start = len(upList[i][opponent])-blockSize-1
                end = m-len(downList[i])+1
                minDiff = rule[start] - rule[start+blockSize]
                for it in range(start+1,end):
                    if rule[it] - rule[it+blockSize] < minDiff:
                        minDiff = rule[it] - rule[it+blockSize]
                scoreCandTested += minDiff
        else:
            # Otherwise, minimize score of candTested and maximize score of its oponnent
            scoreOpponent += rule[len(upList[i][opponent])-1]
            scoreCandTested += rule[m - len(downList[i])]
    
    return scoreOpponent,scoreCandTested
    
def __competitionPositionalScoringRulePartitioned(rule,M,candTested,opponent,blockNumberP,ranksBlocksP,m,optimPreprocessing=True): 
    n = len(blockNumberP)
    scoreOpponent = 0
    scoreCandTested = 0
    for i in range(n): 
        if (blockNumberP[i][candTested] >= 0) and (blockNumberP[i][candTested] < blockNumberP[i][opponent]):
            # If candTested > opponent, then compute the minimal score difference between them.
            blockSize = ranksBlocksP[i][blockNumberP[i][opponent]]+1 - ranksBlocksP[i][blockNumberP[i][candTested]+1]
            
            if blockSize == 0:
                raise ValueError("Block size = 0")
            
            # If we use preprocessing, then use the M matrix
            if optimPreprocessing:
                M_i = ranksBlocksP[i][blockNumberP[i][opponent]]-blockSize
                M_j = m-(ranksBlocksP[i][-1]-ranksBlocksP[i][blockNumberP[i][candTested]+1] + 1)+blockSize-1
                scoreCandTested += M[M_i,M_j,blockSize-1]
                
            # Otherwise, try every position of the block between the two candidates
            else:
                start = ranksBlocksP[i][blockNumberP[i][opponent]]-blockSize
                end = m-(ranksBlocksP[i][-1]-ranksBlocksP[i][blockNumberP[i][candTested]+1])
                minDiff = rule[start]-rule[start+blockSize]
                for it in range(start+1,end):
                    if rule[it]-rule[it+blockSize] < minDiff:
                        minDiff = rule[it]-rule[it+blockSize]
                scoreCandTested += minDiff
        else:
            # Otherwise, minimize score of candTested and maximize score of its oponnent
            scoreOpponent += rule[ranksBlocksP[i][max(0,blockNumberP[i][opponent])]]
            if blockNumberP[i][candTested] == -1:
                scoreCandTested += rule[-1]
            else:
                scoreCandTested += rule[m - (ranksBlocksP[i][-1]-ranksBlocksP[i][blockNumberP[i][candTested]+1] + 1)]
    
    return scoreOpponent,scoreCandTested
    

    
def __competitionPositionalScoringRuleLinear(rule,M,candTested,opponent,candRankL,m,optimPreprocessing=True): 
    n = len(candRankL)
    scoreOpponent = 0
    scoreCandTested = 0
    
    for i in range(n): 
        if candRankL[i][candTested] >= 0 and candRankL[i][candTested] < candRankL[i][opponent]:
             # If candTested > opponent, then compute the minimal score difference between them.
            blockSize = candRankL[i][opponent] - candRankL[i][candTested]
            
            if blockSize == 0:
                raise ValueError("Block size = 0")
                
            # If we use preprocessing, then use the M matrix
            if optimPreprocessing:
                M_i = candRankL[i][candTested]
                M_j = m-1-(candRankL[i][-1]-candRankL[i][opponent])
                scoreCandTested += M[M_i,M_j,blockSize-1]
                
            # Otherwise, try every position of the block between the two candidates
            else:
                start = candRankL[i][candTested]
                end = (m - candRankL[i][-1])+candRankL[i][candTested]+1
                minDiff = rule[start]-rule[start+blockSize]
                for i in range(start+1,end):
                    if rule[i] - rule[i+blockSize] < minDiff:
                        minDiff = rule[i] - rule[i+blockSize]
                scoreCandTested += minDiff
        else:
            # Otherwise, minimize score of candTested and maximize score of its oponnent
            scoreOpponent += rule[max(candRankL[i][opponent],0)]
            if candRankL[i][candTested] == -1:
                scoreCandTested += rule[-1]
            else:
                scoreCandTested += rule[(m - candRankL[i][-1])+candRankL[i][candTested]]
    
    return scoreOpponent,scoreCandTested
    
def __competitionPositionalScoringRuleMultilinear(rule,M,candTested,opponent,candRankM,orderLengthM,m,optimPreprocessing=True): 
    n = len(candRankM)
    scoreOpponent = 0
    scoreCandTested = 0
    
    for i in range(n): 
        (orderNumberCandTested,rankCandTested) = candRankM[i][candTested]
        (orderNumberOpponent,rankOpponent) = candRankM[i][opponent]
        if rankCandTested >=0 and (rankCandTested < rankOpponent) and orderNumberOpponent == orderNumberCandTested:
            # If candTested > opponent, then compute the minimal score difference between them.
            blockSize = rankOpponent-rankCandTested
            
            if blockSize == 0:
                raise ValueError("Block size = 0")      
                          
            # If we use preprocessing, then use the M matrix
            if optimPreprocessing:
                M_i = rankCandTested-1
                M_j = m-1-(orderLengthM[i][orderNumberOpponent]-rankOpponent)
                scoreCandTested += M[M_i,M_j,blockSize-1] 
                               
            # Otherwise, try every position of the block between the two candidates
            else:
                start = rankCandTested
                end = m - orderLengthM[i][orderNumberCandTested]+rankCandTested+1
                minDiff = rule[start] - rule[start+blockSize]
                for i in range(start+1,end):
                    if rule[i] - rule[i+blockSize] < minDiff:
                        minDiff = rule[i] - rule[i+blockSize]
                scoreCandTested += minDiff
        else:
            # Otherwise, minimize score of candTested and maximize score of its oponnent
            scoreOpponent += rule[max(rankOpponent,0)]
            if rankCandTested == -1:
                scoreCandTested += rule[-1]
            else:
                scoreCandTested += rule[m - (orderLengthM[i][orderNumberCandTested]-rankCandTested)]

    return scoreOpponent,scoreCandTested
    
def __competitionPositionalScoringRule(rule,M,candTested,opponent,ordersInfos,candIndex,m,verbose=False,optimPreprocessing=True):
    #Get orders infos
    [upList,downList] = ordersInfos[0]
    [blockNumberP,ranksBlocksP] = ordersInfos[1]
    [candRankL] = ordersInfos[2]
    [candRankM,orderLengthM] = ordersInfos[3]
    
    #Compute subscore for each special case
    scoreOpponentGeneralCase,scoreCandTestedGeneralCase = __competitionPositionalScoringRuleGeneral(rule,M,candTested,opponent,upList,downList[candIndex],m,optimPreprocessing)
    scoreOpponentPartitioned,scoreCandTestedPartitioned = __competitionPositionalScoringRulePartitioned(rule,M,candTested,opponent,blockNumberP,ranksBlocksP,m,optimPreprocessing)
    scoreOpponentLinear,scoreCandTestedLinear = __competitionPositionalScoringRuleLinear(rule,M,candTested,opponent,candRankL,m,optimPreprocessing)
    scoreOpponentMultilinear,scoreCandTestedMultilinear = __competitionPositionalScoringRuleMultilinear(rule,M,candTested,opponent,candRankM,orderLengthM,m,optimPreprocessing)
    
    #Compute total scores
    scoreOpponent = scoreOpponentGeneralCase + scoreOpponentPartitioned + scoreOpponentLinear + scoreOpponentMultilinear
    scoreCandTested = scoreCandTestedGeneralCase + scoreCandTestedPartitioned + scoreCandTestedLinear + scoreCandTestedMultilinear
    
    #Print if verbose
    if verbose:
        print("Test "+str(candTested)+" ("+str(scoreCandTested)+") against "+str(opponent)+" ("+str(scoreOpponent)+")")
    
    # Return True iff candTested is always better than its opponent
    return scoreCandTested >= scoreOpponent

  
## General algorithm : Positional scoring rule

def positionalScoringRule(population,m,rule,verbose=False,optimUpActivated=True,optimCandActivated=True,optimPreprocessing=True,parallelized=False,process=4,chunksize=1,chunks=10):
    #Step 1 : get Up and Down (or similar order informations)
    if parallelized:
        ordersInfos,candToTest,maxScore = upDownParallelized(population,m,rule,verbose,process,chunksize,chunks)
    else:
        ordersInfos,candToTest,maxScore = upDown(population,m,rule,verbose,optimUpActivated,optimCandActivated)
    
    # The optimization at preprocessing step has complexity m**3 while
    # the unoptimized algorithm has complexity m*n so we check wether
    # m**2 > n or not to know if we do preprocessing
    n = len(population)
    if m*m > n:
        optimPreprocessing = False
    if optimPreprocessing:
        M = precomputeScore(rule,m)
    else:
        M = []
        
    #Step 2 : do __competitions between candidates
    NW = []
    if not(optimCandActivated):
        candToTest = [i for i in range(m)]
        order = [i for i in range(m)]
    else:
        order = np.argsort(maxScore)
        
    #Test all candidates in candToTest
    for i in range(len(candToTest)):
        isaNW = True
        for j in range(m):
            if candToTest[i] != order[j]:
                if not(__competitionPositionalScoringRule(rule,M,candToTest[i],order[j],ordersInfos,i,m,verbose,optimPreprocessing)):
                    isaNW = False
                    break
        if isaNW:
            NW.append(candToTest[i])

    return NW