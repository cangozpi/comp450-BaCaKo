import numpy.random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorboard
# import keras
import matplotlib as plt
import os


class Env:

    def __init__(self):  # initialize the env
        # env variables
        self.finishTime = 18.50
        self.beginTime = 10.01
        self.trainTime = 5
        self.budgetState = [100, 0]
        self.executionFee = 4 / 10 ** 4
        self.stateSpaceStDict = dict()
        self.stateSpaceLtDict = dict()
        self.LtStepSize = 300
        self.StStepSize = 3000
        self.step2 = 0
        self.reward_range = [0]



        df = pd.read_csv("shortTerm5min.csv", sep=",")
        df.drop('<TICKER>', inplace=True, axis=1)
        df.drop('<PER>', inplace=True, axis=1)
        df.dropna()
        dateDf = df[['<DATE>']]
        timeDf = df[['<TIME>']]
        df.drop('<DATE>', inplace=True, axis=1)
        df.drop('<TIME>', inplace=True, axis=1)

        for i in range(len(dateDf)):
            date = dateDf.iloc[i]
            time = timeDf.iloc[i]

            if date[0] > 20200110:  # skip the first date since no data is present earlier to process
                self.stateSpaceStDict[(date[0], time[0])] = df.iloc[i - 105: i].copy(deep=True).reset_index()

        df1 = pd.read_csv("longTermData.csv", sep=",")
        df1.drop('<TICKER>', inplace=True, axis=1)
        df1.drop('<PER>', inplace=True, axis=1)
        # df1.dropna()
        dateDf1 = df1[['<DATE>']]
        timeDf1 = df1[['<TIME>']]
        df1.drop('<DATE>', inplace=True, axis=1)
        df1.drop('<TIME>', inplace=True, axis=1)
        for i in range(len(dateDf1)):
            date1 = dateDf1.iloc[i]
            time1 = timeDf1.iloc[i]

            if date1[0] > 20200110:  # skip the first date since no data is present earlier to process
                self.stateSpaceLtDict[(date1[0], time1[0])] = df1.iloc[i - 40: i].copy(deep=True).reset_index()

        # print(self.stateSpaceLt[(20210415, 1500)])
        # normalize the columns of stateSpace

        # save to file
        self.stateSpaceStDict = self.scalePrices(self.stateSpaceStDict)
        self.stateSpaceLtDict = self.scalePrices(self.stateSpaceLtDict)

        self.dateListSt = list(self.stateSpaceStDict)

        self.dateListLt = list(self.stateSpaceLtDict)


        # self.stateSpaceSt.to_Frame().to_save("st", sep=',')



    def getReward(self, oldBudgetState, newBudgetState, action, type):

        #if type == 2:
        #    pass
        #elif type == 3:
        #    pass
        #else:
        return newBudgetState[0] - oldBudgetState[0] + newBudgetState[1] - oldBudgetState[1]
        # type = 0 no intrinsic short term
        # type = 1 no intrinsic long term
        # type = 2 short term model comparison with long term model
        # type = 3 long term model comparison with short term model

    def getInstrinsicReward(self, date, budget, type):
        pass


    def scalePrices(self, stateSpace):
        # normalize columns for each frame stored in stateSpace
        length = len(stateSpace[(20200113, 1500)])
        for key, value in stateSpace.items():
            # extract values to normalize wrt
            # value['<LOW>'][len(value['<LOW>']) - 1]

            openNorm = value.iloc[length - 1]['<OPEN>']

            highNorm = value['<OPEN>'][length - 1]

            volNorm = value['<VOL>'][length - 1]

            for column in value:
                if column == '<VOL>':
                    value[column] = value[column] / volNorm
                elif column == 'index':
                    pass
                else:
                    value[column] = value[column] / openNorm

        return stateSpace

    def takeAction(self, budgetState, acts):

        selectedAction = acts[0]
        buyAmount = acts[1]
        hold = acts[2]
        sellAmount = acts[3]
        if (selectedAction == hold): # hold
            return budgetState
        elif (selectedAction == buyAmount):
            return self.buy(budgetState, buyAmount)
        elif (selectedAction == sellAmount):
            return self.sell(budgetState, sellAmount)

    def buy(self, budgetState, percent):
        buyAmount = budgetState[0] * (percent)

        budgetState[0] -= buyAmount * ( 1 + self.executionFee)
        budgetState[1] += buyAmount
        return budgetState

    def sell(self, budgetState, percent):
        sellAmount = budgetState[1] * (percent)
        budgetState[0] += sellAmount * ( 1 - self.executionFee)
        budgetState[1] -= sellAmount
        return budgetState

    def startState(self, type):
        cashBudget = np.random.rand()
        stockBudget = 1 - cashBudget


# before 4089 lt validation , 4089 sonrası validation 20210504 saat 10
        #yine aynı tarih st 476901 bu ve sonrası validation.
        if type == 0 or type == 2:  # short term
            date = self.dateListSt[numpy.random.randint(47601-self.StStepSize)]
            df = self.stateSpaceStDict.get(date)
        else:  # long term
            date = self.dateListLt[numpy.random.randint(4089-self.LtStepSize)]
            df = self.stateSpaceLtDict.get(date)

        arrayVersion = [date]
        for index, row in df.iterrows():
            i = 0
            for element in row:
                if i != 0:
                    arrayVersion.append(element)
                i += 1

        arrayVersion += [cashBudget, stockBudget]

        return arrayVersion

    def getNextState(self, date, newBudget, type):
        nextDate = -1
        isDoneFlag = False
        if type == 0 or type == 2:  # short term
            if self.dateListSt.index(date) >= len(self.dateListSt)-2:
                isDoneFlag = True
            nextDate = self.dateListSt[self.dateListSt.index(date) + 1]
            df = self.stateSpaceStDict.get(nextDate)
        else:  # long term
            if self.dateListLt.index(date) >= len(self.dateListLt)-2:
                isDoneFlag = True
            nextDate = self.dateListLt[self.dateListLt.index(date) + 1]
            df = self.stateSpaceLtDict.get(nextDate)

        arrayVersion = [nextDate]
        for index, row in df.iterrows():
            i = 0
            for element in row:
                if i != 0:
                    arrayVersion.append(element)
                i += 1

        newBudget[1] *= 1/arrayVersion[-10]

        arrayVersion += [newBudget[0], newBudget[1]]

        return isDoneFlag, arrayVersion




    def step(self, state, action, type):

        isDone = False
        reward = 0
        self.step2 += 1
        currentDate = state[0]
        oldBudgetState = state[-2:]
        newBudgetState = self.takeAction(oldBudgetState, action)

        isDone, nextState = self.getNextState(currentDate, newBudgetState, type)
        if type%2 == 1:
            if self.step2 % self.LtStepSize == 0:
                isDone = True
        elif type%2 == 0:
            if self.step2 % self.StStepSize == 0:
                isDone = True

        reward = self.getReward(state[-2:], nextState[-2:], action, type)

        return nextState, reward, isDone




if __name__ == '__main__':
    env = Env()
