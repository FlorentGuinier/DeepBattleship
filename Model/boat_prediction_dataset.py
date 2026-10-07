import numpy as np
import torch
from torch.utils.data import Dataset

FIREDCHANNEL = 0
BOATCHANNEL = 2

class BoatPredictionDataset(Dataset):

    def __init__(self, states_path='../Data/gameStates.npy'):
        self.states = np.load(states_path)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        fullGameState = torch.from_numpy(self.states[idx]).permute(2, 0, 1).float() / 255

        # only boat state
        boatState = fullGameState[BOATCHANNEL].clone().reshape((1, 10, 10))

        # remove un-hit boat definition
        playerKnownState = fullGameState.clone()
        playerKnownState[BOATCHANNEL] = playerKnownState[BOATCHANNEL] * playerKnownState[FIREDCHANNEL]

        return {
            'X': playerKnownState,
            'Y': boatState
        }
