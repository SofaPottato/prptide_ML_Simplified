import os
import re
import sys
import pandas as pd


class LoadDataset:
    def __init__(self, minSeqLength=5):
        self.minSeqLength = minSeqLength
        pass

    def readFasta(self, File):
        if not os.path.exists(File):
            print('Error: "' + File + '" does not exist.')
            sys.exit(1)
        with open(File) as f:
            records = f.read()
        if re.search('>', records) is None:
            print('The input file seems not in fasta format.')
            sys.exit(1)
        records = records.split('>')[1:]
        SeqDict = {}
        for fasta in records:
            array = fasta.split('\n')
            name = array[0].split()[0]
            sequence = re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
            if len(sequence) >= self.minSeqLength:
                SeqDict[name] = sequence
        return SeqDict

    def LoadToxicity_ATSE(self, trainfile, testfile):
        traindf = pd.read_csv(trainfile, index_col=[0])
        trainposSeqDf = traindf[traindf["label"] == 1]
        trainnegSeqDf = traindf[traindf["label"] == 0]
        trainposSeqDf = trainposSeqDf.drop(labels=["label"], axis=1)
        trainnegSeqDf = trainnegSeqDf.drop(labels=["label"], axis=1)
        trainposSeqDict = {}
        trainnegSeqDict = {}
        trainposSeq = trainposSeqDf.T.to_dict('list')
        trainnegSeq = trainnegSeqDf.T.to_dict('list')
        for i in trainposSeq:
            for j in trainposSeq[i]:
                trainposSeqDict[str(i)] = j
        for i in trainnegSeq:
            for j in trainnegSeq[i]:
                trainnegSeqDict[str(i)] = j

        testdf = pd.read_csv(testfile, index_col=[0])
        testposSeqDf = testdf[testdf["label"] == 1]
        testnegSeqDf = testdf[testdf["label"] == 0]
        testposSeqDf = testposSeqDf.drop(labels=["label"], axis=1)
        testnegSeqDf = testnegSeqDf.drop(labels=["label"], axis=1)
        testposSeqDict = {}
        testnegSeqDict = {}
        testposSeq = testposSeqDf.T.to_dict('list')
        testnegSeq = testnegSeqDf.T.to_dict('list')
        for i in testposSeq:
            for j in testposSeq[i]:
                testposSeqDict[str(i)] = j
        for i in testnegSeq:
            for j in testnegSeq[i]:
                testnegSeqDict[str(i)] = j

        return trainposSeqDict, trainnegSeqDict, testposSeqDict, testnegSeqDict
            #key:每個peptide的序號，value:peptideSeq


    def LoadACP_MHCNN(self, file):
        if os.path.exists(file) == False:
            print('Error: "' + file + '" does not exist.')
            sys.exit(1)

        with open(file) as f:
            records = f.read()

        if re.search('>', records) == None:
            print('The input file seems not in fasta format.')
            sys.exit(1)

        records = records.split('>')[1:]
        SeqDict = {}

        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
            SeqDict[name] = sequence

        posSeqDict = {}
        negSeqDict = {}
        seqslist = list(SeqDict.values())
        seqskeys = list(SeqDict.keys())
        n = 0
        while True:
            if (seqskeys[n])[-1] == '0':
                posSeqDict[seqskeys[n]] = seqslist[n]
            elif seqskeys[n][-1] == '1':
                negSeqDict[seqskeys[n]] = seqslist[n]
            n += 1
            if n == len(SeqDict):
                break
        return posSeqDict, negSeqDict

    def LoadAntiCPLoad(self, file):
        if os.path.exists(file) == False:
            print('Error: "' + file + '" does not exist.')
            sys.exit(1)

        with open(file) as f:
            records = f.read()
            records = records.strip()
            myList = {}
            records = records.split('\n')[1:]
            for fasta in records:
                name, sequence = fasta, re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(fasta[0:]).upper())
                myList[name] = sequence
            return myList



