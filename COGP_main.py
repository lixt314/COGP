#python packages
import random
import time
import operator
import evalGP
import sys
# only for strongly typed GP
import gp_restrict
import scoop
from scoop import futures
import numpy as np
# deap package
from deap import base, creator, tools, gp
# fitness function
from FEVal_norm_fast import evalTest_fromvector as evalTest
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
##from FEVal_norm_fast import evalTest_multi as evalTest
from FEVal_norm_fast import feature_length
##plot tree and save
import saveFile2 as saveFile
import gp_tree
##image Data
from strongGPDataType import ndarray
from strongGPDataType import kernelSize,histdata,filterData
from strongGPDataType import poolingType
# defined by author
import functionSet as fs
import logging
'''finished, can ren experiments'''
##from plot_confusion_matrix import plot_conf_matrix
##import matplotlib.pyplot as plt
##dataSetName=str(sys.argv[1])
#randomSeeds=int(sys.argv[2])
randomSeeds =3
dataSetName = 'f1'
#data_path = '/vol/grid-solar/sgeusers/yingbi/conv_gp/al_madi/data_set/'

def load_data(dataset_name, path=None):
    if path is not None:
        file = path+dataset_name+'/'+dataset_name
    else: file = dataset_name
    x_train = np.load(file+'_train_data.npy')
    y_train = np.load(file+'_train_label.npy')
    x_test = np.load(file+'_test_data.npy')
    y_test = np.load(file+'_test_label.npy')
    return x_train, y_train, x_test, y_test

#x_train, y_train, x_test, y_test = load_data(dataSetName, path = data_path)
x_train, y_train, x_test, y_test = load_data(dataSetName)
print(x_train.shape,y_train.shape, x_test.shape,y_test.shape)

logging.basicConfig(level=logging.INFO, filename=str(randomSeeds) + '_' + dataSetName + '.log',
                    format='%(asctime)-15s  %(message)s')
logging.info('#############Strat#######################################################')
logging.info(dataSetName)
logging.info('Train set shape ' + str(x_train.shape))
logging.info('Test set shape ' + str(x_test.shape))
#parameters:
population=500
generation=50
cxProb=0.5
mutProb=0.49
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=6
maxDepth=8
logging.info('population ' + str(population))
logging.info('generation ' + str(generation))
logging.info('cxProb ' + str(cxProb))
logging.info('mutProb ' + str(mutProb))
logging.info('elitismProb ' + str(elitismProb))
logging.info('initialMinDepth ' + str(initialMinDepth))
logging.info('initialMaxDepth ' + str(initialMaxDepth))
logging.info('maxDepth ' + str(maxDepth))
##GP
pset = gp_tree.PrimitiveSetTyped('MAIN',[filterData], histdata, prefix='Image')
pset.addPrimitive(fs.root_conVector2,[histdata,histdata],histdata,name='Root1')
pset.addPrimitive(fs.root_conVector2,[poolingType, poolingType],histdata,name='Root2')
pset.addPrimitive(fs.root_conVector3,[poolingType, poolingType, poolingType],histdata,name='Root3')
pset.addPrimitive(fs.root_conVector4,[poolingType, poolingType, poolingType, poolingType],histdata,name='Root4')
#pooling
pset.addPrimitive(fs.ZeromaxP,[poolingType, kernelSize, kernelSize],poolingType,name='ZMaxPF')
#aggregation
pset.addPrimitive(fs.mixconadd, [poolingType, float, poolingType, float], poolingType, name='Mix_AddF')
pset.addPrimitive(fs.mixconsub, [poolingType, float, poolingType, float], poolingType, name='Mix_SubF')
pset.addPrimitive(np.abs, [poolingType], poolingType, name='AbsF')
pset.addPrimitive(fs.sqrt, [poolingType], poolingType, name='SqrtF')
pset.addPrimitive(fs.relu, [poolingType], poolingType, name='ReluF')
pset.addPrimitive(fs.conv_filters, [poolingType, ndarray], poolingType, name='ConvF')
#pooling
pset.addPrimitive(fs.maxP,[poolingType, kernelSize, kernelSize], poolingType,name='MaxPF')
pset.addPrimitive(fs.maxP,[filterData, kernelSize, kernelSize], poolingType,name='MaxP')
#aggregation
pset.addPrimitive(fs.mixconadd, [filterData, float, filterData, float], filterData, name='M_Add')
pset.addPrimitive(fs.mixconsub, [filterData, float, filterData, float], filterData, name='M_Sub')
pset.addPrimitive(np.abs, [filterData], filterData, name='Abs')
pset.addPrimitive(fs.sqrt, [filterData], filterData, name='Sqrt')
pset.addPrimitive(fs.relu, [filterData], filterData, name='Relu')
pset.addPrimitive(fs.conv_filters, [filterData, ndarray], filterData, name='Conv')

#Terminals
pset.renameArguments(ARG0='grey')
pset.addEphemeralConstant('randomD',lambda:round(random.random(),3),float)
pset.addEphemeralConstant('filters3',lambda:list(fs.random_filters(3)), ndarray)
pset.addEphemeralConstant('filters5',lambda:list(fs.random_filters(5)), ndarray)
pset.addEphemeralConstant('filters7',lambda:list(fs.random_filters(7)), ndarray)
pset.addEphemeralConstant('kernelSize',lambda:random.randrange(2,5,2),kernelSize)

##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", futures.map)

def evalTrainb(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    lsvm= LinearSVC()
    accuracy = round(100*cross_val_score(lsvm, train_norm, y_train, cv=5).mean(),2)
    return accuracy,

def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(np.asarray(func(x_train[i, :, :])))
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
        lsvm= LinearSVC()
        accuracy = round(100*cross_val_score(lsvm, train_norm, y_train, cv=5).mean(),2)
    except:
        accuracy=0
    return accuracy,

toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament,tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):

    random.seed(randomSeeds)
   
    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    stats_size_feature = tools.Statistics(key= lambda ind: feature_length(ind, x_train[1,:,:], toolbox))
    mstats = tools.MultiStatistics(fitness=stats_fit,size_tree=stats_size_tree, size_feature = stats_size_feature)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(randomSeeds, pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop,log, hof

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    train_tf, test_tf, trainLabel, testL, testResults = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test)
    saveFile.saveLog(str(randomSeeds) + dataSetName + 'all_pop.pickle', pop)
    saveFile.saveLog(str(randomSeeds) + dataSetName + 'best_pop.pickle', hof)
    testTime = time.process_time() - endTime
    logging.info('test results ' + str(testResults))

    print(train_tf.shape, test_tf.shape)
    num_features = train_tf.shape[1]
    saveFile.saveAllResults(randomSeeds, dataSetName, hof[0], log,
                            hof, num_features, trainTime, testTime, testResults)
    logging.info(hof[0])
    logging.info('test results ' + str(testResults))
    logging.info('train time ' + str(trainTime))
    logging.info('test time ' + str(testTime))
    logging.info('Train set shape ' + str(train_tf.shape))
    logging.info('Test set shape ' + str(test_tf.shape))
    logging.info('End')
