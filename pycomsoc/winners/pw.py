# Library pyCOMSOC for computational social choice
# New York University 2019
# Algorithms for Possible Winner computation on partial orders


## Imports

from multiprocessing import Pool
import multiprocessing
import numpy as np
import maxflow as mf
import random
import time
import sys
#sys.path.append( '/home/vishal/gurobi811/linux64/lib/python3.5_utf32')
from gurobipy import *
from . import nw

## Possible Winner with Plurality

# Return list of roots of orders in a population.
# rootsList[i,j] = 1 iff j is a root candidate of voter i.

def getRoots(population,m):
    n = len(population)
    rootsList = np.ones((n,m))
    for i in range(n):
        order = population[i]
        for (a,b) in order:
            rootsList[i,b] = 0
    return rootsList.tolist()

# Aggregate similar partial orders for plurality (i.e. partial
# orders with the same roots list)

def __aggregatePlurality(rootsList,m):
    dictRoots = dict()
    tabRoots = []
    countRoots = []
    i = 0
    
    for roots_i in rootsList:
        if str(roots_i) in dictRoots.keys():
            # Increment the number of voters with the same roots list
            countRoots[dictRoots[str(roots_i)]] += 1
        else:
            # Add a new kind of voters
            countRoots.append(1)
            tabRoots.append(roots_i)
            dictRoots[str(roots_i)] = i
            i+=1
    
    return tabRoots,countRoots
        
# The following code build a graph for the PW plurality algorithm
# described by Betzler and Dorn.

def __buildGraphPlurality(graph,score_c,countRoots,rootsList,m,c,blocked=[]):
    nbVotersNodes = len(countRoots)
    size = nbVotersNodes + m - 1
    
    #Initialize the graph
    nodes = graph.add_nodes(size)
    
    for i in range(nbVotersNodes):
        # Add an edge between the source and every voter
        graph.add_tedge(i,countRoots[i],0)
        
        for j in range(m-1):
            if rootsList[i][j] > 0:
                # Add an edge between a voter and a candidate if
                # the candidate is a root for the voter
                graph.add_edge(i,nbVotersNodes+j,countRoots[i],0)
    
    for i in range(m-1):
        # We add edge for m-1 candidate because we removed the candidate
        # being tested "c". If "i" is blocked then it cannot be
        # a co-winner with "c".
        
        if (i < c and i in blocked) or (i >= c and (i+1) in blocked):
            graph.add_tedge(nbVotersNodes+i,0,score_c-1)
        else:
            graph.add_tedge(nbVotersNodes+i,0,score_c)
            

# The following code try to build a possible world in which "c" is
# a winner.

def __tryApproxPlurality(score_c,c,countRoots,rootsList,m,blocked=[]):
    n = len(countRoots)
    
    # Every candidate can earn "score_c" votes but not more
    votesRemaining = [score_c for i in range(m-1)]
    
    # If a candidate is blocked in can earn at most "score_c - 1" votes
    for cand in blocked:
        if cand <c :
            votesRemaining[cand] -= 1
        elif cand > c:
            votesRemaining[cand-1] -= 1
    
    for i in range(n):
        scoreList = list(np.argsort(votesRemaining))
        scoreList.reverse()
        # We initialize the number of voters remaining
        # with the number of voters in the ith bucket
        nbVotersRemaining = countRoots[i]
        
        # We look at the candidate from the one with the lowest
        # score so far to the one with the highest score so far and
        # we give the maximum number of point each time until there
        # is no voters remaining
        for k in range(m-1):
            j = scoreList[k]
            if rootsList[i][j] == 1:
                nbVotes = min(votesRemaining[j],nbVotersRemaining)
                nbVotersRemaining -= nbVotes
                votesRemaining[j] -= nbVotes
        if nbVotersRemaining > 0:
            # If voters cannot vote without making "c" lose, stop
            return False
    
    # If every voters voted and "c" still win then return True
    return True
    
        
# The algorithm to determine if one candidate is a PW with Plurality

def pluralityOneCandidate(rootsList,countRoots,m,c,verbose=False,blocked=[]):
    rootsList_without_c = []
    score_c = 0
    countRoots_without_c = []
    mawflowWanted = 0
    
    # Compute the maximum score of c, the maxflow needed, and
    # isolate voters who cannot vote for c.
    for i in range(len(rootsList)):
        if rootsList[i][c] == 1:
            score_c += countRoots[i]
        else:
            copyRoots = rootsList[i].copy()
            copyRoots.pop(c)
            mawflowWanted += countRoots[i]
            countRoots_without_c.append(countRoots[i])
            rootsList_without_c.append(copyRoots)
            
    # Take care of trivial cases
    if score_c > mawflowWanted:
        if verbose:
            print(str(c)+" : Default winner ("+str(score_c)+")")
        return True
        
    if score_c < (score_c+mawflowWanted)/m:
        if verbose:
            print(str(c)+" : Default loser ("+str(score_c)+")")
        return False
    
    # Try to build a possible world in which c is a winner
    if __tryApproxPlurality(score_c,c,countRoots_without_c,rootsList_without_c,m,blocked=blocked):
        if verbose:
            print(str(c)+" : Winner with approx")
        return True
        
    # If none of the above worked, use Betzler and Dorn algorithm
    graph = mf.GraphInt()
    __buildGraphPlurality(graph,score_c,countRoots_without_c,rootsList_without_c,m,c,blocked=blocked)
    maxflow = graph.maxflow()
    
    if maxflow >= mawflowWanted:
        if verbose:
            print(str(c)+" : Winner with graph")
        return True
    else:
        if verbose:
            print(str(c)+" : Loser ("+str(maxflow)+"/"+str(mawflowWanted)+")")
        return False


# The general algorithm for plurality
def plurality(population,m,verbose=False):
    rootsList_net = getRoots(population,m)
    rootsList,countRoots = __aggregatePlurality(rootsList_net,m)
    
    # Test every candidate
    winners = []
    for c in range(m):
        if pluralityOneCandidate(rootsList,countRoots,m,c,verbose=verbose):
            winners.append(c)
            
    return winners
    

## Veto


# Return list of leaves of orders in a population.
# rootsList[i,j] = 1 iff j is a leaf candidate of voter i.
def getLeaves(population,m):
    n = len(population)
    leavesList = np.ones((n,m))
    for i in range(n):
        pairs = population[i]
        for (a,b) in pairs:
            leavesList[i,a] = 0
    return leavesList.tolist()

# Aggregate similar partial orders for plurality (i.e. partial
# orders with the same roots list)
def __aggregateVeto(leavesList,m):
    dictLeaves = dict()
    tabLeaves = []
    countLeaves = []
    i = 0
    
    for leaves_i in leavesList:
        if str(leaves_i) in dictLeaves.keys():
            countLeaves[dictLeaves[str(leaves_i)]] += 1
        else:
            countLeaves.append(1)
            tabLeaves.append(leaves_i)
            dictLeaves[str(leaves_i)] = i
            i+=1
    
    return tabLeaves,countLeaves
        

# The following code build a graph for the PW veto algorithm
# described by Betzler and Dorn.

def __buildGraphVeto(graph,zero_c,countLeaves,leavesList,m,c,blocked=[]):
    nbVotersNodes = len(countLeaves)
    size = nbVotersNodes + m - 1
    
    #Initialize the graph
    nodes = graph.add_nodes(size)
    
    for i in range(nbVotersNodes):
        
        # Add an edge between the source and every voter
        graph.add_tedge(i,countLeaves[i],0)
        
        for j in range(m-1):
            if leavesList[i][j] >0:
                # Add an edge between a voter and a candidate if
                # the candidate is a root for the voter
                graph.add_edge(i,nbVotersNodes+j,countLeaves[i],0)
    for i in range(m-1):
        # We add edge for m-1 candidate because we removed the candidate
        # being tested "c". If "i" is blocked then it cannot be
        # a co-winner with "c".
        
        if (i < c and i in blocked) or (i >= c and (i+1) in blocked):
            graph.add_tedge(nbVotersNodes+i,0,zero_c+1)
        else:
            graph.add_tedge(nbVotersNodes+i,0,zero_c)
    

    
# The following code try to build a possible world in which "c" is
# not a loser.

def __tryApproxVeto(zero_c,c,countLeaves,leavesList,m,blocked=[]):
    n = len(countLeaves)
    
    # Every candidate must obtain at least "zero_c" vetos
    vetosRemaining = [zero_c for i in range(m-1)]
    
    # If a candidate is blocked it must obtain more than "zero_c" vetos
    for cand in blocked:
        if cand < c:
            vetosRemaining[cand] += 1
        else:
            vetosRemaining[cand-1] += 1
    
    for i in range(n):
        scoreList = list(np.argsort(vetosRemaining))
        scoreList.reverse()
        
        # If every other candidate have more vetos than "c", stop
        if np.max(vetosRemaining) <= 0:
            return True
            
        # We initialize the number of voters remaining
        # with the number of voters in the ith bucket
        nbVotersRemaining = countLeaves[i]
        for k in range(m-1):
            j = scoreList[k]
            if leavesList[i][j] == 1:
                nbVetos = min(vetosRemaining[j],nbVotersRemaining)
                nbVotersRemaining -= nbVetos
                vetosRemaining[j] -= nbVetos
                
    if np.max(vetosRemaining) <=0:
        return True
    else:
        return False
    

# The algorithm to determine if one candidate is a PW with Veto

def vetoOneCandidate(leavesList,countLeaves,m,c,n,verbose=False,blocked=[]):
    leavesList_without_c = []
    zero_c = 0
    countLeaves_without_c = []
    
    # We compute the minimal number of Veto on 'c' and the Maxflow needed
    # for the graph.
    for i in range(len(leavesList)):
        if leavesList[i][c] == 1 and np.sum(leavesList[i]) == 1:
            zero_c += countLeaves[i]
        else:
            l = leavesList[i].copy()
            l.pop(c)
            countLeaves_without_c.append(countLeaves[i])
            leavesList_without_c.append(l)
    mawflowWanted = zero_c*(m-1)
    
    # We take care of the trivial case
    if zero_c > n/2:
        if verbose:
            print(str(c)+" : Default loser ("+str(zero_c)+")")
        return False
    
    # We try to build a possible world in which c is not a loser
    if __tryApproxVeto(zero_c,c,countLeaves_without_c,leavesList_without_c,m,blocked=blocked):
        if verbose:
            print(str(c)+" : Winner with approx")
        return True
    
    # We use Betzler and Dorn algorithm and Build a Graph
    graph = mf.GraphInt()
    __buildGraphVeto(graph,zero_c,countLeaves_without_c,leavesList_without_c,m,c,blocked=blocked)
    maxflow = graph.maxflow()
    
    if maxflow >= mawflowWanted:
        if verbose:
            print(str(c)+" : Winner with graph")
        return True
    else:
        if verbose:
            print(str(c)+" : Loser ("+str(maxflow)+"/"+str(mawflowWanted)+")")
        return False

# The PW for Veto

def veto(population,m,verbose=False):
    leavesList_net = getLeaves(population,m)
    leavesList,countLeaves = __aggregateVeto(leavesList_net,m)
    winners = []
    
    for c in range(m):
        if vetoOneCandidate(leavesList,countLeaves,m,c,len(population),verbose=verbose):
            winners.append(c)
    
    return winners
    
##K-approval in the Partitioned Preferences case

# This function convert partial order to partitioned preferences
def __posetToPartitioned(poset,m):
    # Get roots of the preferences
    children = [[] for i in range(m)]
    isRoot = np.ones(m)
    for (a,b) in poset:
        children[a].append(b)
        isRoot[b] = 0
    roots = []
    for i in range(m):
        if isRoot[i] == 1:
            roots.append(i)
    
    # Create partitioned preferences
    current_rank = 0
    queue = roots.copy()
    partitionedPreferences = []
    while queue != []:
        partitionedPreferences.append(queue)
        c0 = queue[0]
        queue = children[c0].copy()
        
    return partitionedPreferences

# This function compute the set of candidate for which we must decide if we give a point or not
def __remainingCandidate(partitioned,k,scores,m):
    candidatseSeen = len(partitioned[0])
    j = 0
    while candidatseSeen <= k:
        # Add 1 to candidate in top-k positions
        for cand in partitioned[j]:
            scores[cand] += 1
        
        j +=1
        candidatseSeen += len(partitioned[j])
    
    # Return the number of candidate in the block for whom we
    # will decrease their score
    n_remaining = len(partitioned[j])-(candidatseSeen-k)
    return n_remaining,partitioned[j]

# This function solve the PW problem for K-approval in case of partitioned preferneces
def kappPartitionedOneCand(c,graphsInfos,minScores,m):
    n = len(graphsInfos)
    score_c = minScores[c]
    maxflowNeeded = 0

    graph = mf.GraphInt()
    graph.add_nodes(n+m)
    
    for i in range(n):
        n_remaining,setRemaining = graphsInfos[i]
        # If we can vote for 'c' we vote for it
        if c in setRemaining and n_remaining>0:
            score_c += 1
            n_remaining -= 1
        
        maxflowNeeded += n_remaining
        
        # We add an edge between the source and the voter
        graph.add_tedge(i,n_remaining,0)
        
        # We add an edge between the voter and candidates
        # to which it can give a point
        for cand in setRemaining:
            if cand != c:
                graph.add_edge(i,n+cand,1,0)
    
    for cand in range(m):
        if cand != c:
            # We add an edge between each candidate different
            # than c and the target
            graph.add_tedge(n+cand,0,score_c-minScores[cand])
            
    # Run maxflow algorithm
    maxflow = graph.maxflow()
    
    return maxflow == maxflowNeeded
    

# The algorithm for PW for partitioned preferences and k-approval
def kappPartitioned(P,m,k,list=[]):
    minScores = [0]*m
    graphsInfos = []
    
    # Extract information from poset
    for poset in P:
        partitioned = __posetToPartitioned(poset,m)
        graphsInfos.append(__remainingCandidate(partitioned,k,minScores,m))
    
    # Get every PW
    if list == []:
        list = [i for i in range(m)]
    pw = []
    for cand in list:
        if kappPartitionedOneCand(cand,graphsInfos,minScores,m):
            pw.append(cand)
    return pw

## Approximate set of PW for Borda

def __maxRankApprox(upList,downList,m,c,rule,danger=[],danger_weights=[],verbose=False,blocked=[]):
    n = len(upList)
    
    # If there is no weights for dangerous opponent, give them all a weight of 1
    if danger_weights ==[]:
        danger_weights = [1]*len(danger)
    
    # Initialize all the scores
    score = np.zeros(m)
    
    # Initialize the order in which we see the voters
    indexOrder = [i for i in range(n)]
    np.random.shuffle(indexOrder)
  
    for i in range(n):
        newIndex = indexOrder[i]
        
        # number of points given to ith candidate
        pointsGiven = [0 for i in range(m)]
        
        # Which candidate are in Up(c), Down(c) and Independent(c)
        up_c = []
        down_c = []
        independent_c = []
        independentScore = []
        for j in range(m):
            if j!=c:
                if j in upList[newIndex][c]:
                    up_c.append(j)
                elif j in downList[c][newIndex]:
                    down_c.append(j)
                else:
                    independent_c.append(j)
                    independentScore.append(score[j])
        
        remainingSpace = len(independent_c)
        
        # Look at dangerous candidates
        for j in range(len(danger)):
            dangerousCandidate = danger[j]

            if dangerousCandidate in independent_c:
                # spaceNeeded is the number of candidate in Down(danger)
                # and not in Down(c)
                spaceNeeded = 0
                for child_w in downList[dangerousCandidate][newIndex]:
                    if child_w not in down_c:
                        spaceNeeded += 1
                    
                # If we can, we push the dangerous candidate and every
                # candidate < to it in Down(c)
                if remainingSpace - spaceNeeded >=0 :
                    for child_w in downList[dangerousCandidate][newIndex]:
                        if child_w not in down_c:
                            down_c.append(child_w)
                    remainingSpace -= spaceNeeded
        
        # Get the minimum rank of c
        rank_c = len(upList[newIndex][c])-1
        
        # We maximize the rank of c but maximize in the same time its score (useful for k-approval for instance)
        freedSpace = 0
        while rank_c < (m-len(downList[c][newIndex])-(len(independent_c)-remainingSpace)) and rule[rank_c] == rule[rank_c+1]:
            rank_c += 1
            freedSpace += 1
        remainingSpace = remainingSpace-freedSpace
            
        # Get the score of c
        pointsGiven[c] = rule[rank_c]
        
        # Sort candidates in Independent(c) depending on their score so far
        # The most dangerous candidates appear first
        independentArgsorted = np.argsort(independentScore)[::-1]
        
        # Determine wether we put independent candidate before or after c
        for j in range(len(independent_c)):
            independantCand = independent_c[independentArgsorted[j]]
            # If there is no space left, then put the candidate in Up
            if remainingSpace == 0 and independantCand not in down_c:
                up_c.append(independantCand)
            # Otherwise, we look if we can put it and its children in Down(c)
            elif independantCand not in down_c:
                spaceNeeded = 0
                for child_w in downList[independantCand][newIndex]:
                    if child_w not in down_c:
                        spaceNeeded += 1
                if remainingSpace -spaceNeeded >=0 :
                    for child_w in downList[independantCand][newIndex]:
                        if child_w not in down_c:
                            down_c.append(child_w)
                    remainingSpace -= spaceNeeded
                else:
                    up_c.append(independent_c[independentArgsorted[j]])
        
        # We keep in mind children of dangerous candidate and of which candidate
        # they are a child.
        dangerousParent = [[] for i in range(m)]
        for dangerousCandidate in danger:
            for dangerousChild in downList[dangerousCandidate][newIndex]:
                dangerousParent[dangerousChild].append(dangerousCandidate)
        
        # We compute the list of parents of candidates in Down(c) because
        # we are starting from c and going down in the tree and 
        # we compute the list of children of candidates in Up(c) because
        # we are going up in the tree.
        parents = [[] for i in range(m)]
        children = [[] for i in range(m)]
        parentsCount = [0 for i in range(m)]
        childrenCount = [0 for i in range(m)]
        for j in range(m):
            if j in up_c:
                for elem in downList[j][newIndex]:
                    if elem in up_c and elem != j:
                        parents[elem].append(j)
                        childrenCount[j] += 1
            elif j in down_c:
                for elem in upList[newIndex][j]:
                    if elem in down_c and elem != j:
                        children[elem].append(j) 
                        parentsCount[j] += 1
        
        # We compute the list of candidate without parents
        # which are in Down(c)
        orphanDown = []
        for j in down_c:
            if parentsCount[j] == 0:
                orphanDown.append(j)
        
        # We compute the list of candidate without children
        # which are in Up(c)
        orphanUp = []
        for j in up_c:
            if childrenCount[j] == 0:
                orphanUp.append(j)
        
        # We assign ranks for candidates < c, i.e. candidates in Down(c)
        # We start with candidate with no parents in Down(c) and we go down in
        # the tree. At each step, we consider candidates in Down(c) such that all
        # their parents in down(c) have already been ranked.
        queueDown = orphanDown
        currentRank = rank_c+1
        while queueDown != []:
            if len(danger) == 0:
                # If there is no dangerous candidate, we select
                # the candidate with the minimal score so far
                scoreDown = [score[i] for i in queueDown]
                candMin = np.argmin(scoreDown)
                
            else:
                # Otherwise, we select in priority candidates
                # which seems not dangerous.
                scoreDownNotDangerous = [(score[i],index) for index,i in enumerate(queueDown) if i not in danger]
                if len(scoreDownNotDangerous) == 0:
                    scoreDown = [score[i] for i in queueDown]
                    candMin = np.argmin(scoreDown)
                else:
                    candMin = min(scoreDownNotDangerous)[1]
            
            # We get the candidate, remove it from the queue and continue
            selectedCandidate = queueDown[candMin]
            queueDown.pop(candMin)
            pointsGiven[selectedCandidate] = rule[currentRank]
            currentRank += 1
            
            # We add candidates in the queue if all their parents in Down(c)
            # are already ranked
            for child_w in children[selectedCandidate]:
                parentsCount[child_w] -= 1
                if parentsCount[child_w] == 0:
                    queueDown.append(child_w)
               
        # We assign ranks for candidates > c, i.e. candidates in Up(c)
        # We start with candidate with no children in Up(c) and we go up in
        # the tree. At each step, we consider candidates in Up(c) such that all
        # their children in Upf(c) have already been ranked.     
        queueUp = orphanUp
        rankUp = rank_c-1
        while queueUp != []:
            if len(danger) == 0:
                # If there is no dangerous candidate, we select
                # the candidate with the maximal score so far
                scoreUp = [score[i] for i in queueUp]
                candMax = np.argmax(scoreUp)
            else:
                # Otherwise, we select in priority candidates
                # which seems dangerous.
                scoreUpDangerous = [(score[i],index) for index,i in enumerate(queueUp) if i in danger]
                if len(scoreUpDangerous) > 0:
                    candMax = max(scoreUpDangerous)[1]
                else:
                    scoreUp = [(len(dangerousParent[i]),np.sum(np.array(score)[dangerousParent[i]]),score[i],index) for index,i in enumerate(queueUp)]
                    candMax = max(scoreUp)[-1]
                    
            
             # We get the candidate, remove it from the queue and continue
            w_max = queueUp[candMax]
            queueUp.pop(candMax)
            pointsGiven[w_max] = rule[rankUp]
            rankUp -= 1
            
            # We add candidates in the queue if all their children in Up(c)
            # are already ranked
            for parents_w in parents[w_max]:
                childrenCount[parents_w] -= 1
                if childrenCount[parents_w] == 0:
                    queueUp.append(parents_w)   
                    
        # We increment the score of all the candidates
        score += pointsGiven
    
    # We get the maximum score and check if c has the best score, i.e. is a winner
    maxScore = np.max(score)
    if score[c] == maxScore:
        # We check if blocked candidate have no more than c
        for candBlocked in blocked:
            if score[candBlocked] == maxScore:
                # c is not a winner in this situation, because of candidate blocked
                return False,candBlocked
        
        # We count the number of candidate with the same score
        countMax = 0
        for i in range(m):
            if score[i] == maxScore:
                countMax += 1
        # c is a winner, with countMax - 1 cowinners
        return True,countMax
    
    # If c not a winner, print the winner and its score (if verbose), and return
    # the candidate with the maximum score
    if verbose:
        print("The maximum is not "+str(c)+" ("+str(score[c])+") but "+str(np.argmax(score))+" ("+str(np.max(score))+")")
        if len(danger) > 0:
            print("Were minimized : "+str(danger))
    
    return False,np.argmax(score) 
    

# This function simulate a competition between one candidate (score mximized) and
# 2 opponents (score minimized), but we try to minimize the minimal score of 
# the opponent and we count the number of 
def __competitionKapp1v2(k,candTested_list,opponent,upList,downList,m): 
    [cand1,cand2] = candTested_list
    n = len(upList)
    pointsFilled = 0
    for i in range(n):
        minposOpponent = len(upList[i][opponent])
        maxposCand1 = m-len(downList[cand1][i])+1
        maxposCand2 = m-len(downList[cand2][i])+1
        maxposSCand = m-len(set(downList[cand1][i]+downList[cand2][i]))+  1
        isCand1Up = cand1 in upList[i][opponent]
        isCand2Up = cand2 in upList[i][opponent]
        if isCand1Up or isCand2Up:
            if minposOpponent > k and maxposCand1 > k and maxposCand2 > k and maxposSCand <= k:
                pointsFilled += 1
        else:
            if maxposCand1 > k and maxposCand2 > k and maxposSCand <= k:
                pointsFilled += 1
    return pointsFilled
    
    
def buildPossibleWorld(precompute,rule,candidatesToTest,shuffle=1,verbose=False,maxTries=10,listQ=[],blocked=[],maxDiff=False):
    # Initialization
    (upList,downList,_,m) = precompute
    singleWinners = []
    surePW = []
    possiblePW = []
    countWorld = 0
    
    
    for candidate in candidatesToTest:
        # Test a candidate
        if verbose:
            print("Testing "+str(candidate))
            
        # verified is False until we find a world in which c is a winner
        verified = False
        
        # We try "shuffle" different shuffling of the population
        for i in range(shuffle):
            
            # We initialize the list of dangerous candidate with the empty list
            danger = []
            countTries = 0
            
            while (verified == False):
                
                # Build a new world
                countWorld += 1
                isAWinner,winner = __maxRankApprox(upList,downList,m,candidate,rule,danger=danger,verbose=verbose,blocked=blocked)
                
                # If c is a winner, then ok
                if isAWinner:
                    if verbose:
                        print("Is a Possible winner!")
                    verified = True
                    
                    # If it is a winner alone, then remember it
                    if maxDiff and winner == 1:
                        singleWinners.append(candidate)
                    
                    # Stop
                    break
                else:
                    # If c is not a winner, but the winner was already considered dangerous, then
                    # stop trying to find a possible world
                    if winner in danger:
                        break 
                    else:
                        # Otherwise, add this candidate to the list of dangerous candidate and try
                        # again (if there are tries remaining)
                        danger.append(winner)
                        countTries += 1
                        if countTries == maxTries:
                            break
            # If c is a winner stop
            if verified:
                break
        
        # If c is a PW, add it to the list surePW and otherwise possiblePW
        if verified:
            if listQ == []:
                surePW.append(candidate)
            else:
                return True,[]
        else:
            possiblePW.append(candidate)
    
    if maxDiff:
        return surePW,possiblePW,singleWinners
    else:
        return surePW,possiblePW,countWorld


    
def pruningPW(population,m,rule,kapproval=False,verbose=False,maxCompetition=10):
    n = len(population)
    # Is it k-approval rule
    if kapproval:
        kindex = int(np.sum(rule))
    
    # Are we using preprocessing
    if m*m > n:
        optimPreprocessing = False
        M = []
    else:
        optimPreprocessing = True
        M = nw.precomputeScore(rule,m)
        
    # Compute Up and Down
    maxScore = [0 for i in range(m)]
    upList = []
    for i in range(len(population)):
        # Compute parents, children
        parents = [[] for i in range(m)]
        children = [[] for i in range(m)]
        for (a,b) in population[i]:
            parents[b].append(a)
            children[a].append(b)
        
        # Compute roots
        roots = []
        for j in range(m):
            if len(parents[j]) == 0:
                roots.append(j)
                
        # Compute up
        upList_i = nw.__upGeneral(children,parents,roots,m,maxScore,rule)
        upList.append(upList_i)
        
    # Compute down
    downList = [[] for i in range(m)]
    for i in range(len(upList)):
        upList_i = upList[i]
        for j_1 in range(m):
            downList[j_1].append([])
        for j_1 in range(m):
            for j_2 in upList_i[j_1]:
                downList[j_2][i].append(j_1)
    argsortMaxScore = np.argsort(maxScore)
    
    # Search for PW
    totalScore = np.sum(rule)*len(population)
    maximumScore = np.max(rule)*len(population)
    
    possibleWinners = []
    defaultWinners = []
    
    # Test every candidate
    for i in range(m):
        candidate = argsortMaxScore[i]
        bestScoreCand = maximumScore-maxScore[candidate]
        # Trivial case : best score too low
        if bestScoreCand < (totalScore-bestScoreCand)/(m-1): 
            if verbose:
                print(str(candidate)+" : Default Loser ("+str(bestScoreCand)+")")
        
        # Trivial case : best score high enough
        elif bestScoreCand > (totalScore)/2:
            if verbose:
                print(str(candidate)+" : Default Winner ("+str(bestScoreCand)+")")
            defaultWinners.append(candidate)
            possibleWinners.append(candidate)
        
        # Trivial case : highest best score
        elif bestScoreCand == maximumScore-maxScore[argsortMaxScore[0]]:
            if verbose:
                print(str(candidate)+" : Default Winner ("+str(bestScoreCand)+")")
            defaultWinners.append(candidate)
            possibleWinners.append(candidate)
            
        # Hard case
        else:
            isPW = True
            
            # Step 1 : Prune candidates which are not PW with NW algo
            countCompetition = 0
            scoreDiff = np.ones(m)*np.inf
            
            for opponent in possibleWinners:
                # Stop if we've reach the maximum number of competition
                if countCompetition == maxCompetition:
                    break
                else:
                    countCompetition += 1
                
                # Note that opponent and candidate tested are inverted because we want to find PW not NW
                scoreCandidate,scoreOpponent = nw.__competitionPositionalScoringRuleGeneral(rule,M,opponent,candidate,upList,downList[opponent],m,optimPreprocessing)
                scoreDiff[opponent] = scoreCandidate-scoreOpponent
                
                if verbose:
                    print("Test "+str(opponent)+" ("+str(scoreOpponent)+") against "+str(candidate)+" ("+str(scoreCandidate)+")")
                
                # If the opponent is better, then the candidate is not a PW
                if scoreCandidate < scoreOpponent:
                    isPW = False
                    break
            
            # Step 2 : for k-approval, try NW algorithm with multiple opponents at the same time
            if len(possibleWinners) > 1 and kapproval and isPW:
                argsortOpponents = np.argsort(scoreDiff)
                
                # get the two most dangerous candidate and all candidate with the 
                # same score difference than the second most dangerous candidate
                opponentList = list(argsortOpponents[:2])
                opponent2 = opponentList[1]
                j = 2
                while scoreDiff[argsortOpponents[j]] == scoreDiff[opponent2]:
                    opponentList.append(argsortOpponents[j])
                    j += 1
                
                # Do a competition between the candidate being tested and
                # 2 opponent at the same time
                for opponent_i1 in range(len(opponentList)):
                    for opponent_i2 in range(opponent_i1+1,len(opponentList)):
                        opponent_1 = opponentList[opponent_i1]
                        opponent_2 = opponentList[opponent_i2]
                        
                        pointsToFill = scoreDiff[opponent_1]+scoreDiff[opponent_2]
                        pointsFilled = __competitionKapp1v2(kindex,[opponent_1,opponent_2],candidate,upList,downList,m)
                        
                        if verbose:
                            print(str(candidate)+" vs "+" : "+str(opponent_1)+","+str(opponent_2)+" --> ("+str(pointsFilled)+"/"+str(pointsToFill)+")")
                        
                        if pointsFilled > pointsToFill:
                            isPW = False
    
            if isPW:
                # There is at least 2 default winners
                if len(defaultWinners) == 1:
                    defaultWinners.append(candidate)
                possibleWinners.append(candidate)
    
    # We want to test every candidate which is not a default winner.
    candidatesToTest = []
    for cand in possibleWinners:
        if cand not in defaultWinners:
            candidatesToTest.append(cand)
    
    return defaultWinners,candidatesToTest,(upList,downList,maxScore,m)
    
 
# Global algorithm for the approximation
# kapproval : is the rule a k-approval ?
# shuffle : number of shuffling we are trying
# maxTries : number of Tries for each shuffling (i.e. number of dangerous opponent)
# listQ : if not empty, we want to know if an element of the list is a PW. return True, [] if yes and False, [..] otherwise
# blocked : candidates which cannot be co winners with candidates in listQ
# maxDiff : 
# maxCompetition : maximum number of competition for each candidate in step 1 (pruning)
# reutrnNumber : return only the number of PW found (or not) at each step

def approx(population,m,rule,shuffle=1,verbose=False,maxTries=10,listQ=[],blocked=[],maxDiff=False,maxCompetition=1000,returnNumber=False,kapproval=False):
    # Step 1 : pruning
    defaultWinners,candidatesToTest,precompute = pruningPW(population,m,rule,kapproval,verbose,maxCompetition)
    if verbose:
        print("step 1:",defaultWinners,"/",candidatesToTest)
    step1 = m - len(candidatesToTest)
    
    # Determine which candidate we will test
    # listQ != [] for possible queries
    if listQ == []:
        if maxDiff:
            surePW = []
            candidatesToTest = defaultWinners+candidatesToTest
    else:
        surePW = []
        candidatesToTestOld = candidatesToTest.copy()
        candidatesToTest = []
        for candidate in listQ:
            if candidate in candidatesToTestOld:
                candidatesToTest.append(candidate)
        
    # Step 2 : build a possible world
    winners,possiblePW,usefulInfo= buildPossibleWorld(precompute,rule,candidatesToTest,shuffle,verbose,maxTries,listQ,blocked,maxDiff)
    if verbose:
        print("step2:",winners,"/",possiblePW)
    step2 = m-step1-len(possiblePW)
    step3 = len(possiblePW)
    
    # Return all the PW
    if listQ == []:
        if returnNumber:
            return step1,step2,step3
        elif maxDiff:
            # return all possible PW and list of possible single winners (with not cowinners)
            return winners+possiblePW+defaultWinners,usefulInfo
        else:
            # return all winners for sure, possible PW (which need to be tested) and number of world visited
            return winners+defaultWinners,possiblePW,usefulInfo
    else:
        return False,possiblePW
            

# Specific algorithm for borda and kapproval
def approxBorda(population,m,shuffle=1,verbose=False,maxTries=10,listQ=[],blocked=[],maxDiff=False,maxCompetition=1000,returnNumber=False):
    rule = [m-1-i for i in range(m)]
    return approx(population,m,rule,shuffle,verbose,maxTries,listQ,blocked,maxDiff,maxCompetition,returnNumber,kapproval=False)
    
def approxKapproval(population,m,k,shuffle=1,verbose=False,maxTries=10,listQ=[],blocked=[],maxDiff=False,maxCompetition=1000,returnNumber=False):
    rule =[1]*k+[0]*(m-k)
    return approx(population,m,rule,shuffle,verbose,maxTries,listQ,blocked,maxDiff,maxCompetition,returnNumber,kapproval=True)
    
    
## Winner set for Plurality
# A set of candidate is a winner set if there is an instance in which they all win together (with the same score)

# The following function compute the maximum score of a set of candidate if it wins together
def __getMaxScoreSet(dicoScore,rootsList,rootsCount,m,candidateSet,n_voters):
    strCand = str(sorted(candidateSet))
    
    # If we've already seen this set, strop
    if strCand in dicoScore.keys():
        score = dicoScore[strCand]
        return score
        
    # If there is one candidate, we returns the number of voters who vote for it
    nbCandidate = len(candidateSet)
    if nbCandidate == 1:
        dicoScore[strCand] = n_voters
        return n_voters
     
    else:
        # Otherwise, for every candidate in the set :
        for i in range(nbCandidate):
            # Step 1 : Compute the rootsList of voters who can vote for i but
            # who can also vote for another candidate in the set
            cand_i = candidateSet[i]
            rootsList_i = []
            rootsCount_i = []
            n_voters_i = 0
            for j in range(len(rootsList)):
                canChangeVote = False
                k = 0
                while k < nbCandidate and not(canChangeVote):
                    if candidateSet[k] != cand_i and rootsList[j][candidateSet[k]] == 1:
                        canChangeVote = True 
                    k += 1
                if canChangeVote:
                    n_voters_i += rootsCount[j]
                    rootsList_i.append(rootsList[j])
                    rootsCount_i.append(rootsCount[j])
            complem_n_voters_i = n_voters - n_voters_i
            
            # We remove the ith candidate of the set and compute recursively the best score
            set_i = candidateSet.copy()
            set_i.pop(i)
            maxScoreSet_i = __getMaxScoreSet(dicoScore,rootsList_i,rootsCount_i,m,set_i,n_voters_i)
            maxScoreSet_i = min(maxScoreSet_i,n_voters//nbCandidate)
            # If min Score(i) >= maxscore(set-i) then we need to lower the score of i and
            # we know what is the best score
            if complem_n_voters_i >= maxScoreSet_i:
                dicoScore[strCand] = maxScoreSet_i
                return maxScoreSet_i
            
            # If the score of i is not sufficient but there might be remaining candidates (the remaining
            # part of the division), we can still obtain mascore(set-i) and to check that we use
            # a graph
            elif complem_n_voters_i + (n_voters_i-maxScoreSet_i*(nbCandidate-1)) >= maxScoreSet_i:
                g = mf.GraphInt()
                
                # one node for each voters and each candidate in the set
                g.add_nodes(len(rootsCount)+nbCandidate)
                
                # one edge between a candidate and a voter who can vote for it
                for j in range(len(rootsCount)):
                    g.add_tedge(j,rootsCount[j],0)
                    for k in range(nbCandidate):
                        if rootsList[j][candidateSet[k]] == 1:
                            g.add_edge(j,len(rootsCount)+k,rootsCount[j],0)
                            
                # one edge between each candidate and the target with weight maxscore(set-i)
                for k in range(nbCandidate):
                    g.add_tedge(len(rootsCount)+k,0,maxScoreSet_i)
                
                # If every candidate obtain at least maxscore(set-i), then it is the best score for sure
                maxflow = g.maxflow()
                if maxflow == maxScoreSet_i*nbCandidate:
                    dicoScore[strCand] = maxScoreSet_i
                    return maxScoreSet_i

# This function build a graph such that the maximum score of every candidate is "maxscore(set)"
def __buildGraphPluralitySet(g,score,rootsList,rootsCount,m):
    g.add_nodes(len(rootsCount)+m)
    for i in range(len(rootsCount)):
        g.add_tedge(i,rootsCount[i],0)
        for j in range(m):
            if rootsList[i][j] == 1:
                g.add_edge(i,len(rootsCount)+j,rootsCount[i],0)
    for j in range(m):
        g.add_tedge(len(rootsCount)+j,0,score)
            


def setPluralityPW(dicoScore,rootsList,rootsCount,m,candidateSet,n_total,verbose=False):
    # Compute the number of voter with favorite candidate in the set
    n_set = 0
    for i in range(len(rootsList)):
        ok = False
        for j in candidateSet:
            if rootsList[i][j] == 1:
                ok = True
                break
        if ok:
            n_set += rootsCount[i]
            
    # Compute the maximum score of the set
    if verbose:
        print("Compute maximum score...")
    score = __getMaxScoreSet(dicoScore,rootsList,rootsCount,m,candidateSet,n_set)
    if verbose:
        print("Maximum score : ",score)
    # Obvious loser : the maximum score is lower than nbVoters/nbCandidates
    if score < n_total/m:
        if verbose:
            print("Obvious loser (",score,"/",n_total//m,")")
        return False
    
    # Otherwise we build a graph and check if there is an instance in which every candidate
    # obtain a score < maxscore(set)
    g = mf.GraphInt()
    maxwanted = __buildGraphPluralitySet(g,score,rootsList,rootsCount,m)
    maxflow = g.maxflow()
    if maxflow == n_total:
        if verbose:
            print("Winners !")
        return True
    else:
        if verbose:
            print("Losers (",maxflow,"/",n_total,")")
        return False
    
def plurality_set(population,m,candidateSet,verbose=False):
    n_total = len(population)
    
    # get roots
    roots = getRoots(population,m)
    rootsList,rootsCount = __aggregatePlurality(roots,m)
    
    # compute if PW
    dicoScore =dict()
    return setPluralityPW(dicoScore,rootsList,rootsCount,m,candidateSet,n_total,verbose)
    
## Winner set Veto




        
def __getMinVetoSet(dicoZero,leavesList,leavesCount,m,candidateSet,n_voters):
    strCand = str(sorted(candidateSet))
    
    # If we've already computed this value
    if strCand in dicoZero.keys():
        zero = dicoZero[strCand]
        return zero
    nbCandidates = len(candidateSet)
    # If there is only one candidate
    if nbCandidates == 1:
        dicoZero[strCand] = n_voters
        return n_voters
    else:
        # Otherwise, try this for every candidate in the set
        for i in range(nbCandidates):
            # Step 1 : Look at the voters who can veto i (therefore, they can veto
            # only other candidate of the set)
            cand_i = candidateSet[i]
            leavesList_i = []
            leavesCount_i = []
            n_voters_i = 0
            for j in range(len(leavesList)):
                if leavesList[j][cand_i] == 0:
                    n_voters_i += leavesCount[j]
                    leavesList_i.append(leavesList[j])
                    leavesCount_i.append(leavesCount[j])
                    
            # Step 2 : Compute recursively the number of veto of the set without i
            set_i = candidateSet.copy()
            set_i.pop(i)
            complem_n_voters_i = n_voters - n_voters_i
            zero_set_i = __getMinVetoSet(dicoZero,leavesList_i,leavesCount_i,m,set_i,n_voters_i)
            
            # If it is
            if complem_n_voters_i <= zero_set_i:
                dicoZero[strCand] = zero_set_i
                return zero_set_i
        
        # If none of the above worked, then the min number of veto is nbVoters/nbCandidates
        zeros = np.ceil(n_voters/nbCandidates)
        dicoZero[strCand] = zeros
        return zeros
        
# This function build a graph for PW set determination and Veto's rule. Every candidate
# must have at least "zeros(set)" vetos
def __buildGraphVetoSet(g,zeros,leavesList,leavesCount,m):
    g.add_nodes(len(leavesCount)+m)
    for i in range(len(leavesCount)):
        g.add_tedge(i,leavesCount[i],0)
        for j in range(m):
            if leavesList[i][j] == 1:
                g.add_edge(i,len(leavesCount)+j,leavesCount[i],0)
    for j in range(m):
        g.add_tedge(len(leavesCount)+j,0,zeros)
            
    

def setVetoPW(dicoZero,leavesList,leavesCount,m,candidateSet,n_total,verbose=False):
    n_set = 0
    complementary = [x for x in range(m) if x not in candidateSet]
    leavesList_zero = []
    leavesCount_zero = []
    for i in range(len(leavesList)):
        canBeOut = False
        # We check if the voter can veto a candidate outside the set
        for j in complementary:
            if leavesList[i][j] == 1:
                canBeOut = True
                break
        # If not, then we add it to the list of voters who veto the set
        if not(canBeOut):
            n_set += leavesCount[i]
            leavesList_zero.append(leavesList[i])
            leavesCount_zero.append(leavesCount[i])
            
    if verbose:
        print("Computing min zeros...")
    # We compute the minimum number of vetos for our set
    zeros = __getMinVetoSet(dicoZero,leavesList_zero,leavesCount_zero,m,candidateSet,n_set)
    if verbose:
        print("Min zeros : ",zeros)
        
    # trivial case : too many vetos
    if zeros > n_total/m:
        if verbose:
            print("Obvious losers (",zeros,"/",n_total//m,")")
        return False
    
    # hard case : use maxflow algorithm
    g = mf.GraphInt()
    maxwanted = __buildGraphVetoSet(g,zeros,leavesList,leavesCount,m)
    maxflow = g.maxflow()
    if maxflow == zeros*m:
        if verbose:
            print("Winners !")
        return True
    else:
        if verbose:
            print("Losers (",maxflow,"/",zeros*m,")")
        return False
    
# general algorithm for PW set determination with Veto
def veto_set(population,m,candidateSet,verbose=False):
    n_total = len(population)
    
    # Compute leaves
    leaves = getLeaves(population,m)
    leavesList,leavesCount = __aggregateVeto(leaves,m)
    
    # Compute if PW
    dicoZero =dict()
    return setVetoPW(dicoZero,leavesList,leavesCount,m,candidateSet,n_total,verbose)
     

### Exact solution with Gurobi

def createModel(n,m,population,rule):
    # Initialize empty model
    model = Model("election_pw")
    model.setParam("Seed", 42)
    model.params.presolve = 0
    
    # K-approval
    if rule[0] == 'k':
        k = int(rule[1:])
        # Create decision variables for each x^l_i
        x = model.addVars(n, m, vtype = GRB.BINARY, name = "x" )
        
        # Constraint - k votes per voters
        model.addConstrs(  k == sum(x[l,i] for i in range(m)) for l in range(n))
        
        # Contraint - Preferences pair
        for ind,poset in enumerate(population):
            for (a,b) in poset:
                model.addConstr(0 <= x[ind, a] - x[ind, b])
        
    # Borda model
    elif rule == 'b':
        x = model.addVars(n, m,m, vtype = GRB.BINARY, name = "x" )
        
        # Contraint - i > j xor j > i
        model.addConstrs( (x[l, i, j] + x[l, j, i] == 1) 
                        for l in range(n) 
                        for i in range(m) 
                        for j in range(m) 
                        if i!=j )
        
        # Contraint - if i > j and j > k then i > k
        model.addConstrs( (x[l, i, j] + x[l, j, k] + x[l, k, i] <= 2) 
                        for l in range(n)
                        for i in range(m)
                        for j in range(m)
                        for k in range(m)
                        if i != j and i != k and j != k)
        
        # Contraint - Preferences
        for ind,poset in enumerate(population):
            for (a,b) in poset:
                model.addConstr(x[ind,a,b] == 1)
    else:
        # Create decision variables for each x^l_{i,j}
        # x[l,i,p] = 1 if voters i give rank ith candidate at position m-p
        x = model.addVars(n, m, m, vtype = GRB.BINARY, name = "x" )
        

        # Contraint - One rank per candidate and one candidate per rank
        model.addConstrs(  1 == sum(x[l,i,p] for p in range(m)  ) for l in range(n) for i in range(m) )
        model.addConstrs( 1 == sum(x[l, i, p] for i in range(m) )  for l in range(n) for p in range(m) )
    
        # Constraint - partial profile  
        for ind,poset in enumerate(population):
            for (a,b) in poset:
                model.addConstr( 0 >= sum(p * (x[ind, a,p] - x[ind, b,p]) for p in range(m) ) )
       

    
    #save model file
    model.write('model.mps')
    return True
    
  
  
def checkPW(inputs):
    (m,n,candidateSet,rule,candBlocked) = inputs
    # Initialization
    distinguedCandidate = candidateSet[0]
    print("testing :",distinguedCandidate)
    
    try:
        #Loading common constraints model
        model = read('model.mps')
        model.params.mipFocus = 1
        model.params.preDepRow = 1
        model.params.presolve = 1
        model.params.presparsify = 1
        
        x = model.getVars()
        
        # K-approval
        if rule[0] == 'k':
            # Reshaping variables for easy access
            x = np.array(x).reshape((n,m))
            
            # Score of distinguished candidate 
            winner_sum = sum(x[l, distinguedCandidate] for l in range(n))
            
            # Constraint - PW constraint
            for cand in range(m):
                if cand != distinguedCandidate:
                    if cand in candidateSet:
                        model.addConstr( sum(x[l, cand] for l in range(n) ) == winner_sum)
                    else:
                        if cand in candBlocked:
                            maxScore = winner_sum -1
                        else:
                            maxScore = winner_sum
                        model.addConstr( sum(x[l, cand] for l in range(n) )<= maxScore)
                    
        # Borda rule
        if rule == 'b':
            # Reshaping variables for easy access
            x = np.array(x).reshape((n,m,m))
            
            # Score of distinguished candidate 
            winner_sum = sum(x[l, distinguedCandidate ,j] for l in range(n) for j in range(m) if j != distinguedCandidate)
            
            # Constraint - PW constraint
            for cand in range(m):
                if cand != distinguedCandidate:
                    if cand in candidateSet:
                        model.addConstr(sum(x[l, cand ,j] for l in range(n) for j in range(m) if j != cand) == winner_sum)
                    else:
                        if cand in candBlocked:
                            maxScore = winner_sum -1
                        else:
                            maxScore = winner_sum
                        model.addConstr(sum(x[l, cand ,j] for l in range(n) for j in range(m) if j != cand) <= maxScore)
        
        # Other rules
        else:
            # Reshaping variables for easy access
            x = np.array(x).reshape((n,m,m))
            
            # Score of distinguished candidate 
            winner_sum = sum(rule[p]*x[l, distinguedCandidate ,p] for p in range(m) for l in range(n))
            
            # Constraint - PW constraint
            for cand in range(m):
                if cand != distinguedCandidate:
                    if cand in candidateSet:
                        model.addConstr(sum(rule[p]*x[l, cand ,p] for p in range(m) for l in range(n)) == winner_sum)
                    else:
                        if cand in candBlocked:
                            maxScore = winner_sum -1
                        else:
                            maxScore = winner_sum
                        model.addConstr(sum(rule[p]*x[l, cand ,p] for p in range(m) for l in range(n)) <= maxScore)

        # Run model
        model.optimize()
        
        # Check if PW
        if model.status == GRB.Status.OPTIMAL:
            output = 1
        else:
            output = 0
            
    except Exception as e:
        print("ERROR in checkPW:",e)
        return e
        
    return (distinguedCandidate, output)
    


def PWgurobi(population,m,rule,pwlist,process=5,verbose=True):
    n = len(population)
    PWlist = []
    tot_start = time.time()
    # Creates a set of processes to simultaneously make calculations
    pool = Pool(processes = process)
    
    # Generates the common constraints
    createModel(n,m,population,rule)
    input_gurobi = [(m,n,[cand],rule,[]) for cand in pwlist]
    output = pool.imap(checkPW, input_gurobi)
    while True:
        try:
            retVal = output.next(timeout=2000 - (time.time() - tot_start))
            if verbose:
                print("Result: ", retVal)
            if retVal[1] == 1:
                PWlist.append(retVal[0])
        except StopIteration:
            break
        except multiprocessing.TimeoutError:
            print("Timeout")
            pool.terminate()
            return False
    pool.close()
    pool.join()
    return PWlist
    
    
# General algorithms for Borda, K-approval, and any positional scoring rule
def borda(P,m,verbose=False,shuffle=1,maxTries=4,maxCompetition=10,process=5,setCand=[],listQ=[],blocked=[]):
    if listQ == []:
        PWsure,PWdoubt,_ = approxBorda(P,m,shuffle,verbose,maxTries,listQ,blocked,maxCompetition=maxCompetition)
        if PWdoubt == []:
            return PWsure
        PWgur = PWgurobi(P,m,'b',PWdoubt,process,verbose)
        return PWsure+PWgur
    else:
        success,PWdoubt = approxBorda(P,m,shuffle,verbose,maxTries,listQ,blocked,maxCompetition=maxCompetition)
        if success:
            return True
        if PWdoubt == []:
            return False
        PWgur = PWgurobi(P,m,'b',PWdoubt,process,verbose)
        return PWgur != []
    
def kapp(P,m,k,verbose=False,shuffle=1,maxTries=4,maxCompetition=10,process=5,setCand=[],listQ=[],blocked=[]):
    if listQ == []:
        PWsure,PWdoubt,_ = approxKapproval(P,m,k,shuffle,verbose,maxTries,listQ,blocked,maxCompetition=maxCompetition)
        if PWdoubt == []:
            return PWsure
        PWgur = PWgurobi(P,m,'k'+str(k),PWdoubt,process,verbose)
        return PWsure+PWgur
    else:
        success,PWdoubt = approxKapproval(P,m,k,shuffle,verbose,maxTries,listQ,blocked,maxCompetition=maxCompetition)
        if success:
            return True
        if PWdoubt == []:
            return False
        PWgur = PWgurobi(P,m,'k'+str(k),PWdoubt,process,verbose)
        return PWgur != []
    
def positionalScoringRule(P,m,rule,verbose=False,shuffle=1,maxTries=4,maxCompetition=10,process=5,setCand=[],listQ=[],blocked=[]):
    if listQ == []:
        PWsure,PWdoubt,_ = approx(P,m,rule,shuffle,verbose,maxTries,listQ,blocked,maxCompetition=maxCompetition)
        if PWdoubt == []:
            return PWsure
        PWgur = PWgurobi(P,m,rule,PWdoubt,process,verbose)
        return PWsure+PWgur
    else:
        success,PWdoubt = approx(P,m,rule,shuffle,verbose,maxTries,listQ,blocked,maxCompetition=maxCompetition)
        if success:
            return True
        if PWdoubt == []:
            return False
        PWgur = PWgurobi(P,m,rule,PWdoubt,process,verbose)
        return PWgur != []