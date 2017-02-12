
# coding: utf-8

# # Tic-Tac-Toe

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
np.random.seed(228)


# In[2]:

def checkPosition(position):
        if checkVertical(position, 'x') or checkHorizontal(position, 'x')         or checkDiagonal(position, 'x'):
            return 'w'
        elif checkVertical(position, 'o') or checkHorizontal(position, 'o')         or checkDiagonal(position, 'o'):
            return 'l'
        elif position.find('.') == -1:
            return 'd'
        else:
            return 'n'       

def checkVertical(position, c):
    for i in range(3, 6):
        if position[i-3] == c and position[i] == c and position[i+3] == c:
            return True
    return False

def checkHorizontal(position, c):
    for i in range(1, 8, 3):
        if position[i-1] == c and position[i] == c and position[i+1] == c:
            return True
    return False

def checkDiagonal(position, c):
    if (position[0] == c and position[4] == c and position[8] == c) or     (position[2] == c and position[4] == c and position[6] == c):
        return True
    return False

def printPosition(position):
    clear_output()
    for i in range(0, 9, 3):
        print(position[i : i + 3])


# In[3]:

class mdpPlayer(object):
    def __init__(self, player, mode, epsilon=0.1, alpha=0.2, discount=1):
        self.state2num = {}
        self.num2state = {}
        self.S = 0
        self.curPosition = "........."
        self.player=player
        self.__generateStates__(self.curPosition)
        self.Q = np.zeros((9, self.S))
        self.eps = epsilon
        self.alpha = alpha
        self.discount = discount
        self.lastA = None
        self.lastS = None
        self.learn = 1
        self.mode = mode
    
    def takeResponse(self, new_position, reward=None):
        if not self.learn:
            return
        
        self.curPosition = new_position
        newS = self.state2num[self.curPosition]
        if reward == None:
            self.lastS = newS
            return
        if self.mode == 'sarsa':
            newA = self.__chooseAction__(self.curPosition)
        elif self.mode == 'q':
            epsilon = self.eps
            self.eps = 0
            newA = self.__chooseAction__(self.curPosition)
            self.eps = epsilon

        self.Q[self.lastA, self.lastS] = (1 - self.alpha) * self.Q[self.lastA, self.lastS] +         self.alpha * (reward + self.discount * self.Q[newA, newS])

        self.lastS = newS 
        
    def takeAction(self, position):
        self.lastA = self.__chooseAction__(position)
        return self.lastA
    
    def __chooseAction__(self, position):
        if position.find('.') == -1:
            return 0
        state = self.state2num[position]
        actions = []
        for i in range(9):
            if position[i] == '.':
                actions.append(i)
        actions = np.array(actions, copy=False)
        
        p = np.random.random()
        if p < self.eps:
            return np.random.choice(actions)
        else:
            values = np.zeros(actions.shape)
            for a in range(actions.shape[0]):
                values[a] = self.Q[actions[a], state]
            best_value = np.max(values)
            return actions[np.random.choice(np.where(values == best_value)[0])]
        
    def __generateStates__(self, position, player=0):
        pos_type = checkPosition(position)
        if player == self.player or pos_type != 'n':
            if self.state2num.get(position) is None:
                self.state2num[position] = self.S
                self.num2state[self.S] = position
                self.S += 1
            else:
                return
        
        if pos_type == 'n':
            for i in range(9):
                if position[i] == '.':
                    new_position = position[:i]
                    new_position += 'x' if not player else 'o'
                    new_position += position[i + 1:]                        
                    self.__generateStates__(new_position, 1 - player)
        return
    
    def playMode(self):
        self.eps = 0
        self.learn = 0
    
    def randomMode(self):
        self.eps = 1
        self.learn = 0
    
    def learnMode(self, epsilon):
        self.eps = epsilon
        self.learn = 1


# In[4]:

class Match(object):
    def __init__(self):
        self.position = '.........'
        self.player = 0
        self.crosseswin = 0
        self.noughtswin = 0
        self.draw = 0
    
    def __train__(self, player0, player1):
        finished = 0
        beginning = 1
        while not finished:
            if self.player == 0:
                move = player0.takeAction(self.position)
                self.position = self.position[: move] + 'x' + self.position[move + 1 :]
                pos_type = checkPosition(self.position)
                if pos_type == 'n':
                    if beginning:
                        player1.takeResponse(new_position=self.position, reward=None)
                    else:
                        player1.takeResponse(new_position=self.position, reward=0)
                elif pos_type == 'd':
                    player0.takeResponse(new_position=self.position, reward=1)
                    player1.takeResponse(new_position=self.position, reward=1)
                    self.draw += 1
                    finished = 1
                elif pos_type == 'w':
                    player0.takeResponse(new_position=self.position, reward=10)
                    player1.takeResponse(new_position=self.position, reward=-10)
                    self.crosseswin += 1
                    finished = 1
            else:
                move = player1.takeAction(self.position)
                self.position = self.position[: move] + 'o' + self.position[move + 1 :]
                pos_type = checkPosition(self.position)
                if pos_type == 'n':
                    player0.takeResponse(new_position=self.position, reward=0)
                elif pos_type == 'l':
                    player0.takeResponse(new_position=self.position, reward=-10)
                    player1.takeResponse(new_position=self.position, reward=10)
                    self.noughtswin += 1
                    finished = 1
            self.player = 1 - self.player
            beginning = 0
            
        self.position = '.........'
        self.player = 0
    
    def __test__(self, player0, player1, verbose=0):
        finished = 0
        while not finished:
            if self.player == 0:
                move = player0.takeAction(self.position)
                self.position = self.position[: move] + 'x' + self.position[move + 1 :]
                pos_type = checkPosition(self.position)
                if pos_type == 'd':
                    self.draw += 1
                    if verbose == 2:
                        printPosition(self.position)
                        print('\nDraw')
                        if type(player0) == humanPlayer or type(player1) == humanPlayer:
                            time.sleep(5)
                    finished = 1
                elif pos_type == 'w':
                    self.crosseswin += 1
                    if verbose == 2:
                        printPosition(self.position)
                        print('\nCrosses win')
                        if type(player0) == humanPlayer or type(player1) == humanPlayer:
                            time.sleep(5)
                    finished = 1
            else:
                move = player1.takeAction(self.position)
                self.position = self.position[: move] + 'o' + self.position[move + 1 :]
                pos_type = checkPosition(self.position)
                if pos_type == 'l':
                    self.noughtswin += 1
                    if verbose == 2:
                        printPosition(self.position)
                        print('\nNoughts win')
                        if type(player0) == humanPlayer or type(player1) == humanPlayer:
                            time.sleep(5)
                    finished = 1
            self.player = 1 - self.player
        self.position = '.........'
        self.player = 0
    
    def trainSet(self, number, player0, player1):
        time0 = time.time()
        self.__init__()
        for i in range(1, number+1):
            self.__train__(player0, player1)
        print('\t' + str(time.time() - time0))
        
    
    def testSet(self, number, player0, player1, verbose=0):
        self.__init__()
        for i in range(1, number+1):
            self.__test__(player0, player1, verbose)
            if verbose == 1 and i % 100 == 0:
                print('\r\tCrosses: ' + str(self.crosseswin) + ', Noughts: ' + str(self.noughtswin) +                       ', Draw: ' + str(self.draw), end=' ')
                time.sleep(0.0001)
        print('')


# In[5]:

class humanPlayer(object):
    def __init__(self):
        pass
    
    def takeResponse(self, new_position, reward):
        pass
    
    def takeAction(self, position):
        printPosition(position)
        action = int(input())
        return action


# In[6]:

def trainNtest(mtch, cr, nt, explore, exploit):
    cr.learnMode(0.85)
    nt.learnMode(0.85)
    print('Exploration learning time, s:')
    mtch.trainSet(explore, cr, nt)
    cr.learnMode(0.15)
    nt.learnMode(0.15)
    print('\nExploitation learning time, s:')
    mtch.trainSet(exploit, cr, nt)
    cr.playMode()
    nt.playMode()
    print('\nTrained crosses vs Trained noughts:')
    mtch.testSet(1000, cr, nt, verbose=1)
    cr.playMode()
    nt.randomMode()
    print('\nTrained crosses vs Random noughts:')
    mtch.testSet(100000, cr, nt, verbose=1)
    cr.randomMode()
    nt.playMode()
    print('\nRandom crosses vs Trained noughts:')
    mtch.testSet(100000, cr, nt, verbose=1)


# In[7]:

mtch = Match()
crS = mdpPlayer(player=0, mode='sarsa', alpha=0.15, epsilon=0.85, discount=0.85)
ntS = mdpPlayer(player=1, mode='sarsa', alpha=0.15, epsilon=0.85, discount=0.85)
trainNtest(mtch, crS, ntS, 100000, 50000)


# In[8]:

crQ = mdpPlayer(player=0, mode='q', alpha=0.15, epsilon=0.85, discount=0.85)
ntQ = mdpPlayer(player=1, mode='q', alpha=0.15, epsilon=0.85, discount=0.85)
trainNtest(mtch, crQ, ntQ, 100000, 50000)


# In[8]:

hp = humanPlayer()


# In[19]:

crS.learnMode(0.15)
ntS.playMode()
mtch.trainSet(50000, crS, ntS)


# In[ ]:



