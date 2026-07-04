import numpy as np
import statistics
from collections import Counter
import pandas as pd
from userPackage.LoadDataset import *
import matplotlib.pyplot as plt
import os
import math


class FeatureStat:
    def __init__(self, dataDf=None, dataPath=None):
        if dataPath is None and dataDf is not None:
            self.data = dataDf
        elif dataPath is not None and dataDf is None:
            self.data = pd.read_csv(dataPath, index_col=[0])
        self.stdNanList = []
        self.newDataDict = {}
        self.newTrainDf = None
        self.removeList = None
        # 加excel

    def sdAnalysis(self, saveFigPath):  # 分析各feature區間
        """
        分析各feature區間並繪製在圖像中分析斜率，並算出各區間累積和獨立的比數即占比
        :return:
        """
        pepTypeList = []
        sd = []
        in_one = 0    # 0~1
        in_two = 0    # 1~2
        in_three = 0  # 2~3
        in_four = 0  # 3~4
        above_four = 0  # 4<
        x = 0
        columns1 = self.data.columns.tolist()
        for column in columns1:  # 計算各feature標準差
            columnList = self.data[column]
            x = x + 1
            st_dev = statistics.pstdev(columnList)
            if math.isnan(st_dev):
                print(f'Found NaN In Std List -- {column}!!!')
                print(f'Found NaN In Std List -- {column}!!!')
                print(f'Found NaN In Std List -- {column}!!!')
                self.stdNanList.append(column)
            else:
                pepTypeList.append(column)
                sd.append(round(st_dev, 6))
                if int(st_dev) <= 1:
                    in_one = in_one+1
                elif int(st_dev) <= 2:
                    in_two = in_two+1
                elif int(st_dev) <= 3:
                    in_three = in_three+1
                elif int(st_dev) <= 4:
                    in_four = in_four + 1
                else:
                    above_four = above_four + 1
        self.newDataDict["pepTypeList"] = pepTypeList
        self.newDataDict["standard deviation"] = sd
        print(f"\n|  [0~1]: {in_one}--[{round(in_one/x,6)}%] [1~2]: {in_two}--[{round(in_two/x,6)}%] [2~3]: {in_three}--[{round(in_three/x,6)}%] [3~4]: {in_four}--[{round(in_four/x,6)}%] [4<]: {above_four}--[{round(above_four/x,6)}%]  |\n|  [1sum]: {in_one}--[{round(in_one/x,6)}%] [2sum]: {in_one+in_two}--[{round((in_one+in_two)/x,6)}%] [3sum]: {in_one+in_two+in_three}--[{round((in_one+in_two+in_three)/x,6)}%] [4sum]: {in_one+in_two+in_three+in_four}--[{round((in_one+in_two+in_three+in_four)/x,6)}%] [above4sum]: {in_one+in_two+in_three+in_four+above_four}--[{round((in_one+in_two+in_three+in_four+above_four)/x,6)}%]  |")

        y1List = [in_one, in_two, in_three, in_four, above_four]
        y2List = [round(in_one / x, 6), round((in_one + in_two) / x, 6), round((in_one + in_two + in_three) / x, 6),
                  round((in_one + in_two + in_three + in_four) / x, 6), round((in_one + in_two + in_three + in_four + above_four) / x, 6)]
        #  雙 y 軸圖寫在下面
        x1StickList = [1, 2, 3, 4, 5]
        fig, ax1 = plt.subplots(figsize=(12, 9), dpi=300)
        ax1.bar(x=x1StickList, height=y1List, label='cutoff of feature', color='lightblue', alpha=0.8)
        ax2 = plt.twinx()
        ax2.plot(x1StickList, y2List, color='red', linestyle="-", linewidth="2", markersize="16", marker=".", label="Accumulative Ratio")
        plt.xticks(x1StickList, ['<1', '1-2', '2-3', '3-4', '>4'])  # 設定 total 軸座標範圍
        ax2.set_ylim([0, 1.1])  # 設定 y 軸座標範圍
        ax1.set_xlabel('S.D. Range', fontsize="20")  # 設定 total 軸標題內容及大小
        ax1.set_ylabel('Feature Number', fontsize="20")   # 設定 y 軸標題內容及大小
        ax2.set_ylabel('Accumulative Ratio', fontsize="20")
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        for x1, y1, y2 in zip(x1StickList, y1List, y2List):  # 刻度fontsize要改大
            ax1.text(x1, y1 + 1, y1, ha='center', va='bottom', fontsize="15")
            ax2.text(x1+0.15, y2-0.02, round(y2, 3), ha='center', va='bottom', fontsize="15")

        # plt.title('sd_analysis', fontsize="18")  # 設定圖表標題內容及大小
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.savefig(saveFigPath)
        plt.show()

    def featureValuePct_analysis(self, saveFinalExcel):
        """
                 feature取出出現前一二多次的data種類，並計算R值綜合stander deviation建立一個csv檔案
        :return:
        """

        columns1 = self.data.columns.tolist()
        for stdNanColumn in self.stdNanList:
            columns1.remove(stdNanColumn)
        Rvalue = []
        top2 = []
        top1 = []
        top12percent = []
        top1percent = []
        n = 0
        above = []
        for column in columns1:
            columnList = self.data[column]
            numLen = self.data[column].shape[0]
            dic = dict(Counter(columnList))
            sortDic = sorted(dic.items(), key=lambda d: d[1], reverse=True)

            if len(sortDic) > 1:
                top2.append(sortDic[1])
            else:
                top2.append('NaN')
            top1.append(sortDic[0])

            if len(sortDic[0:2]) == 1:
                top1percent.append(round(sortDic[0][1] / numLen, 6))
                top12percent.append(round(sortDic[0][1] / numLen, 6))
            else:
                top1percent.append(round(sortDic[0][1] / numLen, 6))
                top12percent.append(round((sortDic[0][1] + sortDic[1][1]) / numLen, 6))
            if sortDic[0][1] == numLen:
                r = sortDic[0][1] / numLen
            else:
                r = (sortDic[0][1] + sortDic[1][1]) / numLen

            Rvalue.append(r)
            if r >= 0.7:
                above.append(column)
            n = n + 1
        self.newDataDict["Rank1"] = top1
        self.newDataDict["Rank2"] = top2
        self.newDataDict["top1percent"] = top1percent
        self.newDataDict["top12percent"] = top12percent
        finalDF = pd.DataFrame(self.newDataDict)
        finalDF = finalDF.sort_values(by="top12percent", ascending=False)
        # finalDF.to_csv(saveFinalExcel)
        finalDF.to_excel(saveFinalExcel)

    @staticmethod
    def topThree(cv, saveXlsxPath):  # 歸類只有一二三種氨基酸
        for label in [0, 1]:
            oneTypeList = []
            twoTypesList = []
            threeTypesList = []
            for pepKey, pepValue in cv[label].items():
                same = {}
                for i in range(len(pepValue)):
                    if pepValue.count(pepValue[i]) > 1:
                        if pepValue[i] in same:
                            same[pepValue[i]] = int(same[pepValue[i]]) + 1
                        else:
                            same[pepValue[i]] = 1
                    else:
                        same[pepValue[i]] = 1
                if len(same) == 1:
                    # oneTypeList.append(f"[{pepKey}]: {same}")
                    oneTypeList.append({pepKey: pepValue})
                elif len(same) == 2:
                    # twoTypesList.append(f"[{pepKey}]: {same}")
                    twoTypesList.append({pepKey: pepValue})
                elif len(same) == 3:
                    # threeTypesList.append(f"[{pepKey}]: {same}")
                    threeTypesList.append({pepKey: pepValue})
                else:
                    pass

            typesDfList = []
            if label == 0:
                sheetNameList = ['One amino acid_neg', 'Two amino acids_neg', 'Three amino acids_neg']
            elif label == 1:
                sheetNameList = ['One amino acid_pos', 'Two amino acids_pos', 'Three amino acids_pos']
            for (typeList, sheetName) in zip([oneTypeList, twoTypesList, threeTypesList], sheetNameList):
                if len(typeList) != 0:
                    typesKeys = []
                    typesValues = []
                    for i in typeList:
                        key, val = next(iter(i.items()))
                        typesKeys.append(key)
                        typesValues.append(val)
                    typesDf = pd.DataFrame(typesValues, index=typesKeys, columns=['sequence'])
                    if os.path.exists(saveXlsxPath):
                        print(f'pepCompositionAnalysis.xlsx 檔案裡的{sheetName}已經存在，已覆蓋之前的資料。')
                        writer = pd.ExcelWriter(saveXlsxPath, engine='openpyxl', mode='a', if_sheet_exists='replace')
                    else:
                        writer = pd.ExcelWriter(saveXlsxPath, engine='openpyxl', mode='w')
                    typesDf.to_excel(writer, sheet_name=sheetName)
                    typesDfList.append(typesDf)
                    writer.close()


    @staticmethod
    def pepCompositionAnalysis(posFastaPath, negFastaPath, saveXlsxPath='../data/featureStat/pepCompositionAnalysis.xlsx'):  # 寫入 positive negative csv檔
        loadObj = LoadDataset()
        cvPos = loadObj.readFasta(posFastaPath)
        cvNeg = loadObj.readFasta(negFastaPath)
        cvDict = {0: cvNeg, 1: cvPos}
        FeatureStat.topThree(cv=cvDict, saveXlsxPath=saveXlsxPath)

    def processData(self, xlsxPath, columnName, number, protectFeatSubstringList):
        featureAnalysisDf = pd.read_excel(xlsxPath, index_col=[1], sheet_name='Sheet1', keep_default_na=False,
                                          na_values=['nan'])
        featureAnalysisDf_X = featureAnalysisDf.drop(['y'])
        unProtectTFlist = []  # T:不保護 F:保護

        if float(number) > 0:
            removeDf = featureAnalysisDf_X.loc[featureAnalysisDf_X[columnName] > float(number)]
            dfIndexList = removeDf.index.to_list()
            for dfIndex in dfIndexList:
                if any(protect in dfIndex for protect in protectFeatSubstringList):  # j
                    unProtectTFlist.append(False)  # 欄位名稱包含保護的字串，就把unProtect設成False，意思就是保護此feature
                else:
                    unProtectTFlist.append(True)  # 欄位名稱不包含保護的字串，就把unProtect設成True，意思就是不保護此feature
            removeDf = removeDf[unProtectTFlist]
        else:
            removeDf = featureAnalysisDf_X.loc[featureAnalysisDf_X[columnName] < float(number)]
        self.removeList = removeDf.index.to_list()
        self.newTrainDf = self.data.drop(columns=self.removeList, axis=1)
        return self.newTrainDf, self.removeList

    @staticmethod
    def delNan(data, logPath="../data/mlData/"):
        processedData = data.replace('NA', np.nan)
        path = logPath
        file = open(path, 'w')
        featureNum = len(processedData.columns)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        if processedData.isnull().any().any():
            nanNumSer = processedData.isnull().sum()
            numDf = pd.DataFrame(nanNumSer, columns=['NAN數量'])
            nanNumDf = numDf.loc[~(numDf == 0).all(axis=1)]
            nanNum = len(nanNumDf.index)
            print("=========!!!!!Warning!!!!!=========", file=file)
            print("NaN Value Is In Your Data!", file=file)
            print('===================================', file=file)
            print(nanNumDf, file=file)
            print('===================================', file=file)
            print("Deleted feature = " + str(nanNum), file=file)
            print("After Deleted, Total feature = " + str(featureNum - nanNum), file=file)
            delNanDf = processedData.dropna(axis=1)
            print("Already Deleted NaN!", file=file)
        else:
            print("No NaN Value in your Data", file=file)
            delNanDf = processedData
        file.close()
        return delNanDf

    def processDataLog(self, logPath="../data/mlData/"):
        """
        :param logPath: input the path of Nan log file, save as .txt
        :return: if NaN value in input data, it will return the data of deleted Nan.
                if not, it will return original data.
        """
        path = logPath + 'processLog_.txt'
        file = open(path, 'w')
        print(f'Origin Data Feature Number = {len(self.data.columns.tolist())}', file=file)
        print(f'Remove Feature Number = {len(self.removeList)}', file=file)
        print(f"Remained Data Feature Number = {len(self.newTrainDf.columns.tolist())}", file=file)

