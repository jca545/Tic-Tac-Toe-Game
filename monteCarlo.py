
import copy
import random
import time
import sys
import math
from collections import namedtuple
#import numpy as np

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

# MonteCarlo Tree Search support

class MCTS: #Monte Carlo Tree Search implementation
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild



    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def isTerminalState(self, utility, moves):
        return utility != 0 or len(moves) == 0
    def monteCarloPlayer(self, timelimit = 4):
        """Entry point for Monte Carlo tree search"""
        start = time.perf_counter()
        end = start + timelimit
        """
        Use time.perf_counter() above to apply iterative deepening strategy.
         At each iteration we perform 4 stages of MCTS: 
         SELECT, EXPEND, SIMULATE, and BACKUP. Once time is up
        we use getChildWithMaxScore() to pick the node to move to
        """
        # print("MCTS: your code goes here. -10")

        while time.perf_counter() < end:
            # select node
            node = self.selectNode(self.root)

            # check adn expand child if it's not terminate
            if not self.isTerminalState(node.state.utility, node.state.moves):
                self.expandNode(node)

            # simulate random play
            if len(node.children) > 0:
                node = random.choice(node.children)

            result = self.simulateRandomPlay(node)

            # update backward
            self.backPropagation(node, result)

        winnerNode = self.root.getChildWithMaxScore()
        assert(winnerNode is not None)
        return winnerNode.state.move


    """SELECT stage function. walks down the tree using findBestNodeWithUCT()"""
    def selectNode(self, nd):
        node = nd
        # print("Your code goes here -5pt")
        # Loop through cur node's children
        while len(node.children)>0:
            node = self.findBestNodeWithUCT(node)
        return node

    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. Parse nd's children and use uctValue() to collect uct's for the
        children....."""
        childUCT = []
        # print("Your code goes here -2pt")

        # Loop through cur node's children
        for child in nd.children:
            childUCT.append(self.uctValue(nd.visitCount, child.winScore, child.visitCount))

        # Find the child with the maximum UCT value
        max_uct_value = max(childUCT)
        best_index = childUCT.index(max_uct_value)
        return nd.children[best_index]


    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """compute Upper Confidence Value for a node"""
        # print("Your code goes here -3pt")
        # Det if prioritize
        if nodeVisit == 0:
            if self.exploreFactor == 0:
                return 0
            else:
                return sys.maxsize
        else:
            # The function: UCBI(n)
            return (nodeScore / nodeVisit) + self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)

    """EXPAND stage function. """
    def expandNode(self, nd):
        """generate all the possible child nodes and append them to nd's children"""
        stat = nd.state
        tempState = GameState(to_move=stat.to_move, move=stat.move, utility=stat.utility, board=stat.board, moves=stat.moves)
        for a in self.game.actions(tempState):
            childNode = self.Node(self.game.result(tempState, a), nd)
            nd.children.append(childNode)

    """SIMULATE stage function"""
    def simulateRandomPlay(self, nd):
        # first use compute_utility() to check win possibility for the current node. IF so, return the winner's symbol X, O or N representing tie
        """If player wins with this move, return k if player is 'X' and -k if 'O' else return 0."""
        winStatus = self.game.compute_utility(nd.state.board, nd.state.move, nd.state.board[nd.state.move])
        # print("your code goes here -5pt")

        # For non root nodes
        if nd.parent is not None:
            # Winning for Player 'X':
            if(winStatus == self.game.k and nd.state.to_move == 'X') or (winStatus == -self.game.k and nd.state.to_move == 'O'):
                nd.parent.winScore = sys.maxsize
                # ret player who won
                return nd.state.to_move
            # Winning for Player 'O':
            elif(winStatus == self.game.k and nd.state.to_move == 'O') or (winStatus == -self.game.k and nd.state.to_move == 'X'):
                nd.parent.winScore = -sys.maxsize
                return nd.state.to_move

        """now roll out a random play down to a terminating state. """

        tempState = copy.deepcopy(nd.state) # to be used in the following random playout
        to_move = tempState.to_move
        # print("Your code goes here -5pt")

        # Get to terminal state
        while not self.game.terminal_test(tempState):
            # get list of all possible legal moves
            possibleMoves = self.game.actions(tempState)
            # random move
            move = random.choice(possibleMoves)
            tempState = self.game.result(tempState, move)

        # Check if a Draw
        if tempState.to_move not in tempState.board:
            return 'N'

        winStatus = self.game.compute_utility(tempState.board, tempState.to_move,
                                                  tempState.board[tempState.to_move])

        return ('X' if winStatus > 0 else 'O' if winStatus < 0 else 'N') # 'N' means tie


    def backPropagation(self, nd, winningPlayer):
        """propagate upword to update score and visit count from
        the current leaf node to the root node."""
        tempNode = nd
        # print("Your code goes here -5pt")

        # loop backward, child to parent till root
        while tempNode is not None:
            # update visit count
            tempNode.visitCount += 1
            # check not a draw
            if winningPlayer != 'N':
                # cur player wins the simu
                if tempNode.state.to_move == winningPlayer:
                    tempNode.winScore += sys.maxsize
                # cur's opponent wins
                else:
                    tempNode.winScore -= sys.maxsize
            # update to parent
            tempNode = tempNode.parent


