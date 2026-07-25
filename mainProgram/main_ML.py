import os
import sys
import matplotlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
matplotlib.use("Agg") 

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

import pandas as pd
import numpy as np
import random
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring
# from MLProcess.Stacking import Stacking   # 已停用 voting/stacking (ΔMCC=0)
# from MLProcess.Voting import Voting        # 已停用 voting/stacking (ΔMCC=0)
from MLProcess.DrawPlot import DrawPlot
# from sklearn.linear_model import LogisticRegression   # 只被 stacking 的 final_estimator 用到，已停用
from collections import Counter



def changeBinaryFeatureInDf(dataDf):
    """
    LightGBM 無法吃 binary feature 以及 int64 type 的 feature，所以人工處理。
    binary feature: 隨機挑 10% 的多數/次多數值來微調 (加減 0.01 或乘 0.99/1.01)
    int64 type feature: 手動轉為 float64
    :param dataDf:
    :return:
    """
    pd.options.mode.chained_assignment = None
    threshold = len(dataDf.index.tolist()) * (95 / 100)
    for column in dataDf.columns.to_list():
        dfUniqueValue = dataDf[column].unique()
        count_class = Counter(dataDf[column])
        dfCounterValue = pd.DataFrame.from_dict(count_class, orient='index', columns=["Count"])
        dfCounterValueSum = dfCounterValue['Count'].nlargest(2).sum()
        if (dfCounterValueSum >= threshold) and (column != 'y'):
            value1 = dfCounterValue['Count'].nlargest(2).index.tolist()[0]
            value2 = dfCounterValue['Count'].nlargest(2).index.tolist()[1]
            if 0 in dfUniqueValue:
                convertMaxValue = value1 - 0.01
                convertMinValue = value2 + 0.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --+0.01-->  ' + str(convertMaxValue))
                print(str(value2) + '  --+0.01-->  ' + str(convertMinValue))
            else:
                convertMaxValue = value1 * 0.99
                convertMinValue = value2 * 1.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --*0.99-->  ' + str(convertMaxValue))
                print(str(value2) + '  --*1.01-->  ' + str(convertMinValue))
            value1Index = list(dataDf.loc[dataDf[column] == value1].index[:])
            value2Index = list(dataDf.loc[dataDf[column] == value2].index[:])
            countMax_int = int(np.ceil(len(value1Index) * 0.1))
            countMin_int = int(np.ceil(len(value2Index) * 0.1))
            randomMaxIndex = random.sample(value1Index, countMax_int)
            randomMinIndex = random.sample(value2Index, countMin_int)
            dataDf[column].loc[randomMaxIndex] = dataDf[column].loc[randomMaxIndex].replace(value1, convertMaxValue)
            dataDf[column].loc[randomMinIndex] = dataDf[column].loc[randomMinIndex].replace(value2, convertMinValue)
        if (dataDf[column].dtype.name == 'int64') and (column != 'y'):
            dataDf[column] = dataDf[column].astype('float64')
            print('Type of ' + str(column) + ' Converted')
            print('int64  ---->  float64')

    return dataDf


# ----- voting() 已停用 (voting/stacking 對此資料 ΔMCC=0，見 complementarityAnalysis 結論) -----
# def voting(dataIndp, loadModelPath, voteNumList, selfTestScoreDf, sortingMeasure):
#     dataIndp_X = dataIndp.drop(["y"], axis=1)
#     dataIndp_y = dataIndp[["y"]]
#     pycVoteObj = PycaretWrapper()
#     vote01_predVectorListIndp = []
#     voteFrac_probVectorListIndp = []
#     voteFrac_predVectorListIndp = []
#     voteModelNameList = []
#     selfTestModelNameList = []
#     selfTestScoreDf = selfTestScoreDf.sort_values(by=sortingMeasure, ascending=False)
#     print('voting model selected by ' + str(sortingMeasure) + '!!!')
#     selfTestModelIndexList = selfTestScoreDf.index.tolist()
#     for selfTestModelIndex in selfTestModelIndexList:
#         selfTestModelName = ''.join(selfTestModelIndex)
#         selfTestModelNameList.append(selfTestModelName)
#     for voteNum in voteNumList:
#         fileNameList = selfTestModelNameList[:voteNum]
#         finalModelList = pycVoteObj.doLoadModel(loadModelPath, b_isFinalizedModel=True, fileNameList=fileNameList)
#         voteObjIndp = Voting(dataX=dataIndp_X, modelList=finalModelList)
#         vote01_predVectorIndp = voteObjIndp.vote_predVector()
#         voteFrac_predVectorIndp, voteFrac_probVectorIndp = voteObjIndp.vote_probVector(cutoff=0.5)
#         vote01_predVectorListIndp.append(vote01_predVectorIndp)
#         voteFrac_predVectorListIndp.append(voteFrac_predVectorIndp)
#         voteFrac_probVectorListIndp.append(voteFrac_probVectorIndp)
#         voteModelNameList.append('voting_' + str(voteNum))
#         print('Voting model with ' + str(voteNum) + ' estimators: ' + str(fileNameList))
#     # 算分 : 使用 [0, 1] 投票的結果,仍需 probVector 來計算 AUC
#     scoreObj = Scoring(predVectorList=vote01_predVectorListIndp, probVectorList=voteFrac_probVectorListIndp,
#                        answerDf=dataIndp_y, modelNameList=voteModelNameList)
#     scoreDf = scoreObj.doScoring(b_optimizedMcc=False, sortColumn='mcc')
#     # 算分 : 使用 prob 投票的結果,仍需 probVector 來計算 AUC
#     scoreProbObj = Scoring(predVectorList=voteFrac_predVectorListIndp, probVectorList=voteFrac_probVectorListIndp,
#                            answerDf=dataIndp_y, modelNameList=voteModelNameList)
#     scoreProbDf = scoreProbObj.doScoring(b_optimizedMcc=False, sortColumn='mcc')
#
#     return scoreDf, scoreProbDf


#####################################################################
#   確定 feature 數目後做 tunedModel & CV                            #
#####################################################################

mlDataPath = "../data/mlData/"  # 內含 feature selection 後的 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
finalModelPath = "../data/finalModel"  # train 好且 finalize 的 model 內含檔案 ex: lr_final.pkl
tuneModelPath = "../data/tuneModel"  # tune 好，但未進行finalize 的 model 內含檔案 ex: lr_tuned.pkl
mlScorePath = "../data/mlScore/"  # 內含 ml model 預測完並算好分的檔案 ex: cvScore.csv, singleModelScore_Indp.csv
# ratioPath = "../data/PeptideProperty/"
# ratioPicPath = "../data/PeptidePropertyPic/"


featNumber = 6

modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']

dataTrainDf = pd.read_csv(mlDataPath + f'/{featNumber}/' + "train_F" + str(featNumber) + ".csv",
                          index_col=[0])  # 例如檔名為train_F190.csv
dataTrainDf = changeBinaryFeatureInDf(dataTrainDf)
pycObj = PycaretWrapper()
setupDf = pycObj.doSetup(trainData=dataTrainDf, sessionID=None)
# 一般情況建議使用 TPE search
pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=5,
                   n_iter=10, early_stopping=False, customGridDict=None)

# 以grid search 做 tune model (通常使用於RBFSVM)
# CList = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
# gammaList = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]
# customGridDict = {'rbfsvm': {'C': CList, 'gamma': gammaList}}
# pycObj.doTuneModel(searchLibrary='scikit-learn', searchAlg='grid', includeModelList=['rbfsvm'], foldNum=5,
#                    n_iter=100, early_stopping=False', customGridDict=customGridDict)

pycObj.doSaveModel(tuneModelPath, b_isFinalizedModel=False)  # 儲存tune好的model
tunedModelList = pycObj.doLoadModel(tuneModelPath, fileNameList=modelNameList,
                                    b_isFinalizedModel=False)  # 讀取tune好的model
tunedModelParamList, scoreRank = pycObj.doCompareModel(fold=5,
                                                       includeModelList=tunedModelList)  # 用tune好的model去做CV,scoreRank即為CV的表格
scoreRank.to_csv(mlScorePath + 'cvScore.csv')

#####################################################################
#     拿FinalizeModel做independent test                              #
#####################################################################

dataIndp = pd.read_csv(mlDataPath + f'/{featNumber}/' + "indp_F" + str(featNumber) + ".csv",
                       index_col=[0])  # 例如檔名為Indp_F190.csv   ##加上test1/2
# dataIndp = changeBinaryFeatureInDf(dataIndp)
dataIndp_X = dataIndp.drop(["y"], axis=1)
dataIndp_y = dataIndp[["y"]]
pycObj.doFinalizeModel()
pycObj.doSaveModel(finalModelPath, b_isFinalizedModel=True)
finalModelList = pycObj.doLoadModel(path=finalModelPath, fileNameList=modelNameList, b_isFinalizedModel=True)
predObjIndp = Predict(dataX=dataIndp_X, modelList=finalModelList)
predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()  # predVectorListIndp：model 產生的原始 0 or 1 預測結果, probVectorListIndp：model 產生的原始 probability 預測結果
# predVectorDf = pd.DataFrame(predVectorListIndp, pepName=modelNameList, columns=dataIndp.pepName).T
probVectorDf = pd.DataFrame(probVectorListIndp, index=modelNameList, columns=dataIndp.index).T
# predVectorDf.to_csv(mlScorePath + 'predVector.csv')
probVectorDf.to_csv(mlScorePath + 'probVector.csv')
scoreObjIndp = Scoring(predVectorList=predVectorListIndp, probVectorList=probVectorListIndp,
                       answerDf=dataIndp_y, modelNameList=modelNameList)
bestPredVectorListIndp = scoreObjIndp.optimizeMcc(cutOffList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # bestPredVectorListIndp：probVectorListIndp 經過最佳 cutoff 轉出的 0 or 1 vector (binary prediction)
                                                  method='mcc',
                                                  bestCutoffJsonPath=f'{paramPath}/bestCutoff.json')
bestPredVectorDf = pd.DataFrame(bestPredVectorListIndp, index=modelNameList, columns=dataIndp.index).T

bestPredVectorDf.to_csv('../data/predResultDf.csv')

scoreDfIndp = scoreObjIndp.doScoring(b_optimizedMcc=True, path=mlScorePath + 'singleModelScore_Indp.csv',
                                     sortColumn='mcc')
#plotPredConfidence
scoreObjIndp.plotPredConfidence(predictionsList=probVectorListIndp, trueLabelsDf=dataIndp_y, numBins=10,
                                modelNameList=modelNameList,
                                outputExcel=mlScorePath + "plotPredConfidence.xlsx",
                                figSave=True,
                                figSavePath=mlScorePath)
drawObj = DrawPlot(answerDf=dataIndp_y, modelList=finalModelList, modelNameList=modelNameList,
                   predArrList=predVectorListIndp, probArrList=probVectorListIndp)
aucDf = drawObj.drawROC(colorList=None, title=False, titleName='Receiver Operating Characteristic', setDpi=True,
                        legendSize=11, labelSize=20, save=True, saveLoc=mlScorePath + 'Multi_Single_Model.png',
                        show=True,
                        dpi=300, figSize=(12, 9))

#####################################################################
#           voting & stacking                                       #
#####################################################################

# ----- 已停用: voting/stacking 對此資料相對最佳單模型 ΔMCC=0 (底層預測太相關)。 -----
# ----- 等新 prompt (S2A/Self_ASK/Contrastive) 讓底層變多樣後再解除註解重評。      -----
# clusterNumList = [3, 4, 5, 6, 7]  # 每個值 = 一個 stacking 模型要用的 base learner 數量
# modelNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf',
#                  'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
# stackObj = Stacking(loadModelPath=tuneModelPath,
#                     modelNameList=modelNameList,
#                     trainDataPath=mlDataPath + f'/{featNumber}/' + "train_F" + str(featNumber) + ".csv",
#                     testDataPath=mlDataPath + f'/{featNumber}/' + "indp_F" + str(featNumber) + ".csv",
#                     clusterNumList=clusterNumList)
# selfTestScoreDf = stackObj.genSelfTestResult(drawPlot=True)
# selfTestScoreDf.to_csv(mlScorePath + 'selfTestScore.csv')
# stkModelList = stackObj.genStkModel(selfTestScoreLabel='mcc', final_estimator=LogisticRegression(), drawPlot=True,
#                                     metric='correlation', linkageType='average')
# stkScoreDf = stackObj.stkModelPredictScoring(scoreCsvPath=mlScorePath + 'stkModelScore.csv', drawPlot=True,
#                                              b_isBinary=True)
#
# voteNumList = [3, 5, 7]  # 只用奇數避免同票
# scoreVoteDfIndp, scoreVoteProbDfIndp = voting(dataIndp=dataIndp,
#                                               loadModelPath=finalModelPath,
#                                               voteNumList=voteNumList,
#                                               selfTestScoreDf=selfTestScoreDf,
#                                               sortingMeasure='mcc')
#
# scoreVoteDfIndp.to_csv(mlScorePath + 'VoteScorePred.csv')
# scoreVoteProbDfIndp.to_csv(mlScorePath + 'VoteScoreProb.csv')