import pickle

def saveResults(fileName,*args,**kwargs):
    f=open(fileName,'w')
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return

def saveLog(fileName,log):
    f=open(fileName,'wb')
    pickle.dump(log,f)
    f.close()
    return

def bestInd(toolbox,population,number):
    bestInd=[]
    best=toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd
        

def saveAllResults(randomSeeds,dataSetName,best_ind_va,log,hof,num_features,trainTime,testTime,testResults):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', best_ind_va.fitness, 'Number of features', num_features,
                         'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                         best_ind_va, 'Best individual in each run',*hof[:])

    return
