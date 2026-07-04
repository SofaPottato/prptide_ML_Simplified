import os
import pickle

import pandas as pd

from devPackage.Normalization import Normalization
from devPackage.PackageBoruta import BorutaPackage
from MLProcess.PycaretWrapper import PycaretWrapper


class FeatureEngineering:
    def __init__(self):
        self.skipFeatureList = None


    @staticmethod
    def dataNormalization(encodeTrainDf=None, encodeIndpDf=None, normalization='standard',
                          saveNmlzScalerPklPath='./data/', loadNmlzScalerPklPath='./data/', b_loadPkl=False):
        nmlzObj = Normalization(trainDf=encodeTrainDf, testDf=encodeIndpDf)
        if normalization.lower() == 'standard':
            if b_loadPkl:
                indpNmlzedDf = nmlzObj.standardTest(loadParams=True, loadNmlzParamsPklPath=loadNmlzScalerPklPath)
            else:
                trainNmlzedDf, standardSca = nmlzObj.standard()
                with open(saveNmlzScalerPklPath, 'wb') as f:
                    pickle.dump(standardSca, f)
        elif normalization.lower() == 'minmax':
            if b_loadPkl:
                indpNmlzedDf = nmlzObj.minMaxTest(loadParams=True, loadNmlzParamsPklPath=loadNmlzScalerPklPath)
            else:
                trainNmlzedDf, minMaxSca = nmlzObj.minMax()
                with open(saveNmlzScalerPklPath, 'wb') as f:
                    pickle.dump(minMaxSca, f)
        elif normalization.lower() == 'robust':
            if b_loadPkl:
                indpNmlzedDf = nmlzObj.robustTest(loadParams=True, loadNmlzParamsPklPath=loadNmlzScalerPklPath)
            else:
                trainNmlzedDf, robustSca = nmlzObj.robust()
                with open(saveNmlzScalerPklPath, 'wb') as f:
                    pickle.dump(robustSca, f)
        else:
            raise ValueError("normalization must be one of: standard / minmax / robust")

        if b_loadPkl:
            return indpNmlzedDf.round(5)
        return trainNmlzedDf.round(5)


    def dataBoruta(self, trainDf, borutaMethod='XGB', runBoruta=True, featRankPath="./data/", skipFeatureList=None):

        self.skipFeatureList = skipFeatureList if skipFeatureList is not None else []
        if runBoruta:
            brtTrainDf = trainDf.drop(columns=self.skipFeatureList)
            brtObj = BorutaPackage(dataDf=brtTrainDf, modelName=borutaMethod.upper(), runBoruta=runBoruta,
                                   featRankPath=featRankPath + 'Boruta-featureRank-{}.csv'.format(borutaMethod))
        else:
            brtObj = BorutaPackage(dataDf=None, modelName=borutaMethod.upper(), runBoruta=runBoruta,
                                   featRankPath=featRankPath)
        return brtObj


    def dataEvalFeatureNum(self, startNum=2, endNum=19, step=2, featNumScorePath='../data/', saveCsvPath='../data/',
                           trainDf=None, indpDf=None, brtObj=None, foldNum=5, session=None,
                           includeModelList=None):
        if includeModelList is None:
            includeModelList = ['gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et',
                                'lightgbm', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda']
        skipList = self.skipFeatureList if self.skipFeatureList is not None else []
        availableFeatNum = brtObj.numberRanks(10 ** 9)  # Boruta 排名後實際可用的特徵
        maxFeat = len(availableFeatNum)
        # 要求的特徵數若超過 Boruta 排名後實際特徵數，會報錯
        requestedNums = list(range(startNum, endNum, step))
        if requestedNums and max(requestedNums) > maxFeat:
            raise ValueError(
                f"要求的特徵數 {max(requestedNums)} 超過 Boruta 排名後實際可用的特徵數 {maxFeat}，"
                f"請把 endNum 調到 {maxFeat + 1} 以內（或減少 skipFeatureList / 放寬前面的特徵篩選）。")

        bestScoreDf_mcc = pd.DataFrame()
        bestScoreDf_auc = pd.DataFrame()
        for featureNum in range(startNum, endNum, step):
            os.makedirs(saveCsvPath + str(featureNum), exist_ok=True)
            featureList = brtObj.numberRanks(featureNum) + skipList + ["y"]
            trainCSV = trainDf[featureList]
            indpCSV = indpDf[featureList]
            trainCSV.to_csv(saveCsvPath + str(featureNum) + "/train_F" + str(featureNum) + ".csv")
            indpCSV.to_csv(saveCsvPath + str(featureNum) + "/indp_F" + str(featureNum) + ".csv")
            trainData = pd.read_csv(saveCsvPath + str(featureNum) + "/train_F" + str(featureNum) + ".csv",
                                    index_col=[0])
            pycObj = PycaretWrapper()
            pycObj.doSetup(trainData, sessionID=session)
            defaultModelParamList, scoreRank_mcc = pycObj.doCompareModel(fold=foldNum,
                                                                         includeModelList=includeModelList)
            scoreRank_auc = scoreRank_mcc.sort_values(by='AUC', ascending=False)
            bestScoreDf_mcc.insert(loc=len(bestScoreDf_mcc.columns.tolist()), value=scoreRank_mcc.iloc[0],
                                   column=featureNum)
            bestScoreDf_auc.insert(loc=len(bestScoreDf_auc.columns.tolist()), value=scoreRank_auc.iloc[0],
                                   column=featureNum)
            scoreRank_mcc.to_csv(saveCsvPath + str(featureNum) + "/CV_" + str(featureNum) + "_byMcc.csv")
            scoreRank_auc.to_csv(saveCsvPath + str(featureNum) + "/CV_" + str(featureNum) + "_byAuc.csv")
        bestScoreDf_mcc.T.to_csv(featNumScorePath + 'bestScore_mcc.csv')
        bestScoreDf_auc.T.to_csv(featNumScorePath + 'bestScore_auc.csv')


    def dataDecidedFeatureNum(self, featureNum, saveCsvPath="./data/", brtObj=None, trainDf=None, indpDf=None):
        skipList = self.skipFeatureList if self.skipFeatureList is not None else []
        featureList = brtObj.numberRanks(featureNum) + skipList + ['y']
        if trainDf is not None:
            trainDf[featureList].to_csv(saveCsvPath + "/train_F" + str(featureNum) + ".csv")
        indpDf[featureList].to_csv(saveCsvPath + "/indp_F" + str(featureNum) + ".csv")
