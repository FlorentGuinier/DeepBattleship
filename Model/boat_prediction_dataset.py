import os
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset

BOATCHANNEL = 0
HITCHANNEL = 2

class BoatPredictionDataset(Dataset):

    def __init__(self, main_dir='..\Data\GameStates'):
        self.main_dir = main_dir
        self.to_tensor = ToTensor()
        self.images_names = sorted(os.listdir(main_dir))

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.main_dir, self.images_names[idx])
        image = Image.open(image_path)
        fullGameState = self.to_tensor(image)

        # only boat state
        boatState = fullGameState[BOATCHANNEL, :, :].detach().clone().reshape((1,10,10))

        # remove un-hit boat definition
        playerKnownState = fullGameState.detach().clone()
        playerKnownState[BOATCHANNEL, :, :] = playerKnownState[BOATCHANNEL, :, :] * playerKnownState[HITCHANNEL, :, :]


        return {
            'X': playerKnownState,
            'Y': boatState
        }