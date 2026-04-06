# Tic-Tac-Toe-Game
Assignment for CMPT 310 - Introduction to Artificial Intelligence

This project implements adversarial search algorithms for an extended n × n TicTacToe game. The goal is to design intelligent agents that can optimally play the game using Minimax, Alpha-Beta pruning, and Monte Carlo Tree Search (MCTS).


## Table of Contents
1. [Navigation Guide](#1-navigation-guide)
2. [Installation](#install)
3. [Getting Started](#start)
4. [Features](#features)



<a name="navigation"></a>

## 1. Navigation Guide

```bash
repository
├── games.py            ## main logic + Minimax + AlphaBeta
├── monteCarlo.py       ## MCTS implementations
├── tic-tac-toe.py      ## main application script
├── utils.py            ## helper functions
```



<a name="install"></a>

## 2. Installation

This project requires using Python and the following Python libraries are used:
- pygame
- numpy

You can install the required libraries using the following command::
```bash
pip install pygame numpy
```


<a name="start"></a>

## 3. Getting Started

Suggestion: Use PyCharm to efficiently switch between algorithms.

#### 3.1. Clone and Navigate
```bash
# 1. Clone this repo to your local machine
git clone $THISREPO
# 2. Navigate into the repository directory
cd $THISREPO
```

#### 3.2. Run project
1. Open tic-tac-toe.py_ file (Make sure you are in the repository directory)
2. (opitonal) Select an grid
3. (optional) Select an algorithm, cutoff depth
4. Start playing tic-tac-toe



<a name="features"></a>

## 4. Features
### Grid
There are three n x n grid to choose from:
- 3 x 3
- 4 x 4
- 5 x 5

#### Search Algorithm
This tic-tac-toe Agent supports three different algorithms:
- Minimax
- Minimax with Alpha-Beta Pruning
Monte Carlo Tree Search (MCTS)

#### timer
Determines the cutoff depth for agent's evaluation function (default -1 = no limit).

