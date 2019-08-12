# DBCOMSOC: Implementing The Necessary and Possible Winner Problem

__Notation__

Number of candidates _m_

Number of voters _n_

## Necessary Winner (NW)

## Possible Winner (PW)
We reduce the PW problem to Integer Linear Programming and add some optimisation and heuristics to make it efficient. 

### Experiments
We evaluate our system with two kinds of experiments. 
#### Pruning optimisation and Gurobi
>Step 1.We prune the list of candidates using the `pw_pruning()`
>
>This returns two lists - 
>>a. list of candidates who are PW for sure
>>
>>b. list of candidates who are possibly PW (we run Gurobi on this list of candidates)
>>
>>Note that the lists in (a) and (b) are disjoint. 
>>Moreover, `pw_pruning()` eliminates the candidates which cannot be PW. 
>>These candidates apprear in neither of the lists.

>Step 2. Run Gurobi on the second list of candidates returned by `pw_pruning()`
>
>>To check if a candidate is a PW or not, comprises of the following three steps-
>>```
>>1. Read the input file containing the patial profile
>>2. Create a model
>>    1. Initialise variables
>>    2. Transitivity constraints
>>    3. Antisymetric constraints
>>    4. Partial profile constraints
>>    5. PW definition constraints
>>3. Optimise (solve) the model
>>```
#### Pruning optimisation, heuristics, and Gurobi
TO BE ADDED


---

### Datasets

All experiments willbe run on three synthetically generated datasets

Artificial Datasets
>1. Dataset 1 - Drop Cand (Kunal B. R.)
>2. Dataset 2 - RSM+ (Théo D.)
>3. Dataset 3 - Top k (Théo D.)
Real Datasets
