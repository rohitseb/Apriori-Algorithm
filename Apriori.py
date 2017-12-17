
# coding: utf-8

# In[1]:

#Apriori Frequent Itemset Generation

import numpy as np 
import time
import collections
import copy


def printOutput(output):
    print("\nRules - ")
    x=False
    
    for i in output:
        for item in LHS[i]:
            if x:
                print(",",end="")
                
            print(item, end="")
            x=True
        print(" --> ", end="")
        x=False
        for item in RHS[i]:
            if x:
                print(",",end="")
                
            print(item, end="")
            x=True
        print()
        x=False
    print()
    print("Count - "+str(len(output)))

file_name = 'dm/data/associationruletestdata.txt'
data = np.loadtxt(file_name, dtype='a', delimiter='\t')
unicode_data = data.view(np.chararray).decode('utf-8')
disease_names = unicode_data[:,-1]
input_data = unicode_data

#Enter Support and Confidence
sup = int(input("Enter Support - "))
threshold=int(input("Enter Confidence - "))

threshold=threshold/input_data.shape[0]
support = (sup/100) * input_data.shape[0]
support = int(support)

#SupportPrint
print("Support is set to be "+str(sup)+"%")

mod_input_data = np.zeros(shape=(0, input_data.shape[0]))

count = 1;
for col in input_data.T:
    colVal = "G"+str(count)+"_";
    col = [colVal + c for c in col]
    mod_input_data=np.vstack((mod_input_data,col))
    count +=1
    
tempMap= collections.OrderedDict()
mod_input_data = mod_input_data.T
klist = mod_input_data.T.flatten()

countMap = collections.OrderedDict()
for val in klist:
    countMap[val] = 0
for val in klist:
    countMap[val] += 1

for key,value in countMap.items():
    if value>=support:
        tempMap[key]=value



if len(tempMap)>0:
        print("Number of length-1 frequent itemsets: "+str(len(tempMap)))

isFirstTime = True
level = 2
qualList = []
for key in tempMap.keys():
    qualList.append(key)
ruleMap = tempMap
while isFirstTime or len(tempMap) > 0:
    for key, value in tempMap.items():
        ruleMap[key] = value
    isFirstTime = False
    keySet=tempMap.keys()
    currLength=level
    level += 1

    
    tempMap= collections.OrderedDict()
    validateSet = set()
    
    for i in range(0, len(qualList)):
        for j in range(i + 1, len(qualList)):
            pairs =(qualList[i], qualList[j])
            
            if currLength==2:
                unionSet=set(pairs)
            else:
                inSet = set()
                for xtuple in pairs:
                    if len(inSet) == 0:
                        inSet = set(xtuple)
                    else:
                        tx = set(xtuple)
                        inSet = inSet & tx
                
                target = currLength - 2
                if len(inSet) != target:
                    break
                
                mTuple  = ()
                for xtuple in pairs:
                    mTuple += xtuple
                unionSet = set(mTuple)
            
            tempSet = sorted(unionSet)
            sortedStr = ', '.join(tempSet)
    
            if sortedStr in validateSet:
                continue
            validateSet.add(sortedStr)
            if len(unionSet)==currLength:
                m=np.zeros(shape=(0,mod_input_data.shape[0]))
                for col in unionSet:
                    arr=col.split("_")
                    colNum = ((int)(arr[0][1:]))-1
                    colVal=mod_input_data[:,colNum]
                    m=np.vstack((m,mod_input_data[:,colNum]))
                m=m.T
                supCount=0
                for row in m:
                    if set(row)==unionSet:
                        supCount+=1
                if supCount>=support:
                    tempMap[tuple(unionSet)]=supCount
    if len(tempMap)>0:
        print("Number of length-"+str(currLength)+" frequent itemsets: "+str(len(tempMap)))
    qualList = []
    for key in tempMap.keys():
        qualList.append(key)

ruleMapMod = {}
for k in ruleMap.keys():
    if(isinstance(k, np.str)):
        ruleMapMod[k] = ruleMap[k]
    else:
        unSortedTup = k
        unSortedVal = ruleMap[k];
        sortedTup = tuple(sorted(k))
        ruleMapMod[sortedTup] = unSortedVal
 
    
ruleMap = {}
ruleMap = ruleMapMod

print("Number of all lengths frequent itemsets: "+str(len(ruleMap)))


##############################################################################################################  


#Generating The Association Rules
from itertools import combinations
keyList = []


for key in ruleMap.keys():
    keyList.append(key)
LHS=[]
RHS=[]

for key in keyList:
    tempSup=ruleMap[key]
    keyLen=len(key)
    if(str(type(key))=="<class 'numpy.str_'>"):
        continue
    comb=[]
    
    while(keyLen>1):
        tempComb = [np.str('|'.join(p)) for p in combinations(key,keyLen-1)]
        comb += (tempComb)
        keyLen-=1
            
    for items in comb:
        compareSet = set()
        l=[]
        if('|'in items):
            items=items.split('|')
            for i in items:
                l.append(i)
        else:
            l.append(items)
        
        ruleRHS=set(key).difference(set(l))
        conf = 0
        keyTuple = tuple(l)
        keyTuple = tuple(sorted(keyTuple))
        if (len(keyTuple) == 1):
            conf = ruleMap[tuple(key)]/ruleMap[l[0]]
        else:
            conf = ruleMap[tuple(sorted(key))]
            denom = ruleMap[keyTuple]
            conf /= denom
            
        if(conf>=threshold):
            LHS.append(tuple(l))
            RHS.append(tuple(ruleRHS))

print("\nThe Association Rules Generated Are - ")
printOutput(range(0, len(LHS)))
print("\n")
#################################################################################################################
            
            
#Get Rules for Template 1
def template1(rules):
    count=rules[1]
    itemList=rules[2].split(",")
    resultList=[]
    ruleCount=0
    
    if count=='NONE':
        if(rules[0]=='RULE'):
            visited = [False] * len(LHS)
            for items in itemList:
                for i in range(0,len(LHS)):
                    if items in LHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
                for i in range(0,len(RHS)):
                    if items in RHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
            
        elif (rules[0]=='HEAD'):
            visited = [False] * len(RHS)
            for items in itemList:
                for i in range(0,len(RHS)):
                    if items in RHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
            
        elif (rules[0]=='BODY'):
            visited = [False] * len(LHS)
            for items in itemList:
                for i in range(0,len(LHS)):
                    if items in LHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
            
        for i in range(0, len(visited)):
            if not visited[i]:
                resultList.append(i)
    
    if count=='ANY':
        if(rules[0]=='RULE'):
            visited = [False] * len(LHS)
            for items in itemList:
                for i in range(0,len(LHS)):
                    if items in LHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
                            resultList.append(i)
                for i in range(0,len(RHS)):
                    if items in RHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
                            resultList.append(i)
            
        elif (rules[0]=='HEAD'):
            visited = [False] * len(RHS)
            for items in itemList:
                for i in range(0,len(RHS)):
                    if items in RHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
                            resultList.append(i)
            
        elif (rules[0]=='BODY'):
            visited = [False] * len(LHS)
            for items in itemList:
                for i in range(0,len(LHS)):
                    if items in LHS[i]:
                        if not visited[i]:
                            visited[i]=True
                            ruleCount+=1
                            resultList.append(i)
            
    if count=='1':
        if(rules[0]=='RULE'):
            visited = [0] * len(LHS)
            for items in itemList:
                for i in range(0,len(LHS)):
                    if items in LHS[i]:
                        if visited[i]==0:
                            ruleCount+=1
                            visited[i]=1
                        elif visited[i]==1:
                            ruleCount-=1
                            visited[i]=2
                    if items in RHS[i]:
                        if visited[i]==0:
                            ruleCount+=1
                            visited[i]=1
                        elif visited[i]==1:
                            ruleCount-=1
                            visited[i]=2
            
        elif(rules[0]=='HEAD'):
            visited = [0] * len(RHS)
            for items in itemList:
                for i in range(0,len(RHS)):
                    if items in RHS[i]:
                        if visited[i]==0:
                            ruleCount+=1
                            visited[i]=1
                        elif visited[i]==1:
                            ruleCount-=1
                            visited[i]=2
            
        elif(rules[0]=='BODY'):
            visited = [0] * len(LHS)
            for items in itemList:
                for i in range(0,len(LHS)):
                    if items in LHS[i]:
                        if visited[i]==0:
                            ruleCount+=1
                            visited[i]=1
                        elif visited[i]==1:
                            ruleCount-=1
                            visited[i]=2
                       
        for i in range(0, len(visited)):
            if visited[i] == 1:
                resultList.append(i)
                
    return resultList


#################################################################################################################


#Get Rules for Template2
def template2(rules):
    count=int(rules[1])
    ruleType=rules[0]
    resultList=[]
    ruleCount=0
    if ruleType=='RULE':
        for i in range(0,len(LHS)):
            if (len(LHS[i])+len(RHS[i]))>=count:
                ruleCount+=1
                resultList.append(i)
    elif ruleType=='BODY':
        for i in range(0,len(LHS)):
            if (len(LHS[i])>=count):
                ruleCount+=1
                resultList.append(i)
    elif ruleType=='HEAD':
        for i in range(0,len(RHS)):
            if len(RHS[i])>=count:
                ruleCount+=1
                resultList.append(i)
    return resultList

#################################################################################################################


#Get Rules for Template 3
def template3(rules):
    andOr = rules[0]
    digits = []
    status = ''
    if 'and' in andOr:
        status = 'and'
        digits = andOr.split('and')
    else: 
        status = 'or'
        digits = andOr.split('or')
    
    result1 = []
    result2 = []
    
    if (digits[0] == '1' and digits[1] == '1'):
        tempRules = ['']*3
        tempRules[0] = rules[1]
        tempRules[1] = rules[2]
        tempRules[2] = rules[3]
        
        result1 = template1(tempRules)
        
        tempRules = ['']*3
        tempRules[0] = rules[4]
        tempRules[1] = rules[5]
        tempRules[2] = rules[6]
        
        result2 = template1(tempRules)
        
    elif (digits[0] == '1' and digits[1] == '2'):
        tempRules = ['']*3
        tempRules[0] = rules[1]
        tempRules[1] = rules[2]
        tempRules[2] = rules[3]
        
        result1 = template1(tempRules)
        
        tempRules = ['']*2
        tempRules[0] = rules[4]
        tempRules[1] = rules[5]
        
        result2 = template2(tempRules)
        
    elif (digits[0] == '2' and digits[1] == '1'):
        tempRules = ['']*2
        tempRules[0] = rules[1]
        tempRules[1] = rules[2]
        
        result1 = template2(tempRules)
        
        tempRules = ['']*3
        tempRules[0] = rules[3]
        tempRules[1] = rules[4]
        tempRules[2] = rules[5]
        
        result2 = template1(tempRules)
        
    elif (digits[0] == '2' and digits[1] == '2'):
        tempRules = ['']*2
        tempRules[0] = rules[1]
        tempRules[1] = rules[2]
        
        result1 = template2(tempRules)
        
        tempRules = ['']*2
        tempRules[0] = rules[3]
        tempRules[1] = rules[4]
        
        result2 = template2(tempRules)
        
    set1 = set(result1)
    set2 = set(result2)
    
    if status =='and':
        printOutput(set1.intersection(set2))
    else:
        printOutput(set1.union(set2))
       
        
#Template Counts
ip=""
while(ip!="quit"):
    print()
    ip=input("Enter Rule - ")
    rules=ip.split("/")
    if len(rules)==3:
        printOutput(template1(rules))
    elif len(rules)==2:
        printOutput(template2(rules))
    elif ip == 'quit':
        break
    else:
        template3(rules)
        


