import json
import os
import sys
import matplotlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
matplotlib.use("Agg") 

from userPackage.Package_Encode_Simplified import FeatureEngineering
from userPackage.FeatureStat import FeatureStat
from userPackage.convertMLTable import loadMLTable


mlDataPath = "../data/mlData/"        # 內含 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"         # 內含檔案: featureTypeDict.pkl, normalize.pkl
normalizeMethod = 'minmax'
dataName = 'HPRD50'

trainMLTableCsv = "../data/HPRD50trainmlTable.csv"
indpMLTableCsv = "../data/HPRD50testmlTable.csv"


trainDf = loadMLTable(trainMLTableCsv)
indpDf = loadMLTable(indpMLTableCsv)

feObj = FeatureEngineering()

# ====================================================================== #
#  normalization                                                         #
# ====================================================================== #
nmlzScalerPath = paramPath + f'{dataName}_{normalizeMethod}Scaler.pkl'

#正常資料不會出現NAN
trainDf = FeatureStat.delNan(data=trainDf, logPath="../data/mlData/delNanTrain.txt")
indpDf = FeatureStat.delNan(data=indpDf, logPath="../data/mlData/delNanIndp.txt")

trainNmlzDf = feObj.dataNormalization(encodeTrainDf=trainDf,
                                      encodeIndpDf=None,  # train scaler存起來 ，indp 另外做
                                      normalization=normalizeMethod,
                                      saveNmlzScalerPklPath=nmlzScalerPath,
                                      loadNmlzScalerPklPath=None,
                                      b_loadPkl=False) # True: 讀取 NmlzScaler 的 pkl 檔 (loadNmlzScalerPklPath)
# False: 把 NmlzScaler 存至 pkl 檔 (saveNmlzScalerPklPath)

indpNmlzDf = feObj.dataNormalization(encodeTrainDf=None,
                                     encodeIndpDf=indpDf,
                                     normalization=normalizeMethod,
                                     saveNmlzScalerPklPath=None,
                                     loadNmlzScalerPklPath=nmlzScalerPath,
                                     b_loadPkl=True)  # indp 永遠套 train 存好的 scaler

# ================================================================================ #
# 把全部 feature 做完 nmlz 的結果存成 csv 檔 (可檢查 feature cutoff 以及 nmlz 的結果) #
# ================================================================================ #
featureStatPath = "../data/featureStat/"

trainNmlzCsvPath = featureStatPath + f'train_{dataName}_{normalizeMethod}.csv'
indpNmlzCsvPath = featureStatPath + f'indp_{dataName}_{normalizeMethod}.csv'
featureAnalysisXlsxPath = featureStatPath + "featureAnalysis.xlsx"

trainNmlzDf.to_csv(trainNmlzCsvPath)
indpNmlzDf.to_csv(indpNmlzCsvPath)

featureStatObj = FeatureStat(dataPath=trainNmlzCsvPath)
featureStatObj.sdAnalysis(
    saveFigPath=featureStatPath + f"sd_analysis_{dataName}_{normalizeMethod}.jpg")# std deviation 分析結果 output 圖片
featureStatObj.featureValuePct_analysis(saveFinalExcel=featureAnalysisXlsxPath)
#featureStatObj.pepCompositionAnalysis(posFastaPath="../data/HemoPI_1_pos_main80%.fasta",
#                                      negFastaPath="../data/HemoPI_1_neg_main80%.fasta",
#                                      saveXlsxPath=featureStatPath + 'pepCompositionAnalysis.xlsx')  # 找出只含有 1, 2 or 3 個 amino acid 的 peptide
# output xlsx 檔

# filteredTrainNmlzDf 為篩選後的 nmlz dataframe, 跑後續 boruta 用  (紀錄剩下幾個 feature)
filteredTrainNmlzDf, removeList = featureStatObj.processData(xlsxPath=featureAnalysisXlsxPath, columnName='top1percent',
                                                             number='+0.99', protectFeatSubstringList=[])
featureStatObj.processDataLog(logPath='../data/mlData/')
filterTrainNmlzPath = featureStatPath + f'filtered_train_{dataName}_{normalizeMethod}.csv' # 移除完 feature 後新的 nmlz dataset 檔, 跑後續 boruta 用
removeFeatureListPath = featureStatPath + f'remove_feature_list_{dataName}_{normalizeMethod}.json'# 移除掉的 feature 會存在這個文字檔裡
filteredTrainNmlzDf.to_csv(filterTrainNmlzPath)
with open(removeFeatureListPath, 'w') as f:
    json.dump(removeList, f)
# ====================================================================== #
#  Boruta 
# ====================================================================== #


brtObj = feObj.dataBoruta(borutaMethod='XGB', runBoruta=True, featRankPath=mlDataPath,
                          trainDf=filteredTrainNmlzDf)

feObj.dataEvalFeatureNum(startNum=2, endNum=40, step=2,
                         featNumScorePath=mlDataPath, saveCsvPath=mlDataPath,
                         trainDf=filteredTrainNmlzDf, indpDf=indpNmlzDf, brtObj=brtObj, foldNum=5, session=None)#sessionID可修改成任意整數，ex:1,4,10,15...
#startNum=5, endNum=50, step=5
# 決定好 feature 數字請開這個
# 原版：feObj.dataDecidedFeatureNum(featureNum=390, saveCsvPath=mlDataPath,
#                                       trainDf=filteredTrainNmlzDf, indpDf=indpNmlzDf, brtObj=brtObj)


