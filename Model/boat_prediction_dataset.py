import os
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset

class BoatPredictionDataset(Dataset):

    def __init__(self, main_dir='..\Data\GameStates'):
        self.main_dir = main_dir
        self.to_tensor = ToTensor()
        self.images_names = sorted(os.listdir(main_dir))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image_path = os.path.join(self.main_dir, self.images_names[idx])
        image = Image.open(image_path)
        Y = self.to_tensor(image)

        # remove un hit boat definition from X (that is what we want to predict)
        X = Y.detach().clone()
        X[0, :, :] = X[0, :, :] * X[2, :, :]

        sample = [X, Y]
        return sample

#test code
if True:
    dataset = BoatPredictionDataset()
    X, Y = dataset[42]
    result1 = to_pil_image(X)
    result2 = to_pil_image(Y)
    result1.show()
    result2.show()