import shutil
from torch.utils.tensorboard import SummaryWriter
import os

class TensorBoardManager():

    def __init__(self, name):

        if os.path.exists('runs/{}'.format(name)):
            shutil.rmtree('runs/{}'.format(name))
        self.writer = SummaryWriter('runs/{}'.format(name))


    def log_loss(self, label, loss_value, x_index):
        self.writer.add_scalar(label,
                               loss_value,
                               x_index)

    def log_losses(self, main_label, list_of_losses, list_of_names, epoch):
        loss_dict = self.build_loss_dict(list_of_losses = list_of_losses, 
                                         list_of_names = list_of_names)

        self.writer.add_scalars(main_tag = main_label, 
                                tag_scalar_dict = loss_dict, 
                                global_step=epoch)
    
    def build_loss_dict(self, list_of_losses, list_of_names):
        return {n: l for n, l in zip(list_of_names, list_of_losses)}

    def set_name(self, name):
        self.run_name = name 


