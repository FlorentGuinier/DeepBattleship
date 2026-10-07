import numpy as np
import torch

FIREDCHANNEL = 0
BOATCHANNEL = 2

def LoadGameStates(states_path='../Data/gameStates.npy', board_index_path='../Data/boardIndex.npy'):
    return np.load(states_path), np.load(board_index_path)

def MakeBatch(states, indices):
    fullGameStates = states[indices].permute(0, 3, 1, 2).float() / 255

    # only boat state
    boatStates = fullGameStates[:, BOATCHANNEL:BOATCHANNEL+1].clone()

    # remove un-hit boat definition
    playerKnownStates = fullGameStates
    playerKnownStates[:, BOATCHANNEL] = playerKnownStates[:, BOATCHANNEL] * playerKnownStates[:, FIREDCHANNEL]

    return playerKnownStates, boatStates
