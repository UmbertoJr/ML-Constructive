from InOut.tools import DirManager
from torch.utils.data import DataLoader
from InOut.image_manager import DatasetHandler


def test_images_created(settings):
    generator = DataLoader(DatasetHandler(settings), settings.bs)
    for data in generator:
        x, y = data["X"], data["Y"]
        print(x.shape)
        print("the dataloader for the training is working!")
        break
