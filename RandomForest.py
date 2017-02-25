import csv
import numpy as np  # http://www.numpy.org
import math



"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of samples
and d is the number of features of each sample

Also, y is assumed to be a vector of n labels
"""

# Enter your name here
myname = "NA"

class RandomForest(object):
    class __DecisionTree(object):

        # class tree + lernToTree + learn:
        #TODO: train decision tree and store it in self.root
        
    
        #build a tree class used to store the tree
        class tree(object):
            def __init__(self,node,value,left=None,right=None):
                self.left=left
                self.right=right
                self.value=value
                self.node=node

        # store the ouput from self.learn (the tree) into a form of class tree
        # input: list of node infomatoin
        # output: tree
        def learnToTree(self,learnOutput):
            root=self.tree(learnOutput[0][0][0][0],learnOutput[0][0][0][1])
            for terminal in learnOutput:
                myPointer=root
                output=terminal[1]
                nodeList=terminal[0]
                for index in range(len(nodeList)-1):
                    aNode=nodeList[index]
                    mySide=aNode[2]
                    if mySide==0:
                        if myPointer.left==None:
                            myPointer.left=self.tree(nodeList[index+1][0],nodeList[index+1][1])
                        myPointer=myPointer.left
                    if mySide==1:
                        if myPointer.right==None:
                            myPointer.right=self.tree(nodeList[index+1][0],nodeList[index+1][1])
                        myPointer=myPointer.right
                    index=index+1
                if nodeList[-1][2]==0:
                    myPointer.left=output
                else:
                    myPointer.right=output
            self.root=root
            return root


        # get the node information of the tree
        # input: X_train, y_train
        # output: list of node information
        def learn(self, X, y):
            #generate boostrap sample
            y_train=y
            bsSampleIndex=np.random.choice(y_train.shape[0], y_train.shape[0],replace=True)
            XFull=X[bsSampleIndex][:]
            yFull=y[bsSampleIndex]
            #select features
            attrFull=range(XFull.shape[1])
            attrFull=np.random.choice(attrFull,6,replace=False) #4
            attrFull=list(attrFull)
            #get the best splitting point
            attrRemain=attrFull[:]
            XRemain=XFull[:]
            yRemain=yFull[:]
            bestAttr,bestValue=self.getSplitAttrPoint(attrRemain,XRemain,yRemain)
            #store the nodes
            store=[]
            currentNodeLeft=(bestAttr,bestValue,0) #x<=value
            currentNodeRight=(bestAttr,bestValue,1)#x>value
            store.append([currentNodeLeft])
            store.append([currentNodeRight])           
            #iterate through all the nodes and record terminal nodes
            terminal=[]
            count=0
            while count<len(store):
                nodeInfo=store[count]
                #update data according to split
                attrRemain=attrFull[:]
                XRemain=XFull[:]
                yRemain=yFull[:]
                for aNode in nodeInfo:
                    attrRemain.remove(aNode[0])
                    Xattribute= XRemain[:,aNode[0]]
                    if aNode[2]==0:
                        IndexRemain=np.where(Xattribute<=aNode[1])[0]
                    elif aNode[2]==1:
                        IndexRemain=np.where(Xattribute>aNode[1])[0]
                    XRemain=XRemain[IndexRemain]
                    yRemain=yRemain[IndexRemain]
		#get the class output
                if len(yRemain)==0:
                    terminal.append([nodeInfo,0])
                elif len(yRemain)<=3:
                    terminal.append([nodeInfo,self.getClassB(yRemain)])
                elif len(attrRemain)==0:
                    terminal.append([nodeInfo,self.getClassB(yRemain)])
                else:   
                    myClass=self.getClass(yRemain)
                    if myClass==None:
                        bestAttr,bestValue=self.getSplitAttrPoint(attrRemain,XRemain,yRemain)
                        aList=nodeInfo[:]
                        aList.append((bestAttr,bestValue,0)) #x<=value
                        bList=nodeInfo[:]
                        bList.append((bestAttr,bestValue,1))#x>value
                        store.append(aList)
                        store.append(bList)
                    else:
                        terminal.append([nodeInfo,myClass])
                count=count+1
            return terminal
            
            



        def getSplitAttrPoint(self,attrs,X,y):
            # input a list of attribution , find the best split attribution and value of that best attribute
            # get the weighted entropy for each attribution and value pair
            weightedEntropyArray=list()#
            for attr in attrs:               
                Xattr=X[:,attr]
                currentWeightedEntropy=list()
                valueList=np.linspace(min(Xattr),max(Xattr),80,endpoint=False)
                valueList=valueList[1:]##
                for value in valueList:
                    #split the data into two set according to the attr and value pair and get the indices
                    indexLeft=np.where(Xattr<=value)[0]
                    indexRight=np.where(Xattr>value)[0]
                    yLeft=y[indexLeft]
                    yRight=y[indexRight]
                    #get entropy for two side after split
                    entropyLeft=self.getEntropy(yLeft)
                    entropyRight=self.getEntropy(yRight)
                    #calculate weighted entropy
                    weightedEntorpy=float(len(indexLeft))/(len(y))*entropyLeft+float(len(indexRight))/(len(y))*entropyRight
                    currentWeightedEntropy.append(weightedEntorpy)
                weightedEntropyArray.append(currentWeightedEntropy)#
            #get minimum weighted entropy
            weightedEntropyArray=np.asarray(weightedEntropyArray)
            minEntropy=weightedEntropyArray.min()
            minEntropyIndex=np.where(weightedEntropyArray == minEntropy)
            bestAttr=attrs[minEntropyIndex[0][0]]
            Xattr=X[:,bestAttr]
            valueList=np.linspace(min(Xattr),max(Xattr),80,endpoint=False)
            valueList=valueList[1:]
            bestValue=valueList[minEntropyIndex[1][0]]
            return bestAttr,bestValue

        def getEntropy(self,y):
            #input a list of y, get the corresponding entropy
            sumY=float(sum(y))
            if (sumY != len(y)) and (sumY != 0):
                entropy=-(sumY/float(len(y)))*math.log(sumY/float(len(y)),2)-(1-sumY/float(len(y)))*math.log(1-sumY/float(len(y)),2)
            else:
                entropy=0
            return entropy

        def getClassB(self,y):
            #input X, get the corresponding predicted y 
            length=len(y)
            numOne=sum(y)
            if float(numOne)/length>0.5:
                return 1
            else:
                return 0

        def getClass(self,y):
            #input X, get the corresponding predicted y
             length=len(y)
             numOne=sum(y)
             if float(numOne)/length>=0.9:
                 return 1
             elif float(numOne)/length<0.1:
                return 0
             else:
                return None           


        # TODO: return predicted label for a single instance using self.root
        def classify(self, test_instance):           
            pointer=self.root
            flag=1
            while flag:
                myNode=pointer.node
                myValue=pointer.value
                if test_instance[myNode]<=myValue:
                    if pointer.left==1:
                        y= 1
                        flag=0
                    elif pointer.left==0 or None:
                        y= 0
                        flag=0
                    else:
                        pointer=pointer.left
                else:
                    if pointer.right==1:
                        y= 1
                        flag=0
                    elif pointer.right==0 or None:
                        y= 0
                        flag=0
                    else:
                        pointer=pointer.right
            return y
        

    decision_trees = []

    def __init__(self, num_trees):
        # TODO: do initialization here, you can change the function signature according to your need
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree()] * num_trees
        
           

    # You MUST NOT change this signature
    # TODO: train `num_trees` decision trees
    def fit(self, X, y):
        self.finalTrees=[]
        for i in range(self.num_trees):
            aLearn=self.decision_trees[i].learn(X,y)
            aTree=self.decision_trees[i].learnToTree(aLearn)
            self.finalTrees.append(aTree)
        
        

    # You MUST NOT change this signature
    def predict(self, X):
        y = np.array([], dtype = int)
        for instance in X:
            votes = np.array([decision_tree.classify(instance)
                              for decision_tree in self.decision_trees])
            counts = np.bincount(votes)
            y = np.append(y, np.argmax(counts))
        return y


def main():
    X = []
    y = []

    # Load data set
    with open("data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])

    X = np.array(X, dtype = float)
    y = np.array(y, dtype = int)

    #Split training/test sets
    # cross validation
    K = 10
    totalAccuracy=0
    for iteration in range(10):
        X_train = np.array([x for i, x in enumerate(X) if i % K != iteration], dtype = float)
        y_train = np.array([z for i, z in enumerate(y) if i % K != iteration], dtype = int)
        X_test  = np.array([x for i, x in enumerate(X) if i % K == iteration], dtype = float)
        y_test  = np.array([z for i, z in enumerate(y) if i % K == iteration], dtype = int)

        randomForest = RandomForest(101)  # Initialize

        randomForest.fit(X_train, y_train)

        y_predicted = randomForest.predict(X_test)

        results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]

        #Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        totalAccuracy=totalAccuracy+accuracy
        print "flod {} accuracy: {:.4f}".format(iteration,accuracy)
    
    print "average accuracy: %.4f"%(totalAccuracy/10)


main()


