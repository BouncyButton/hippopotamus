import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm

PATH = 'trained_models/'


class Experiment:
    r"""This class handles:
            - Interactions with the experiment folder
            - Loading / Saving experiments
            - Datasets
    """

    def __init__(self, config: dict, dataset: Dataset, model: nn.Module, agent) -> None:
        # Backward compatibility
        if isinstance(config, list):
            config = config[0]

        self.projectConfig = config
        self.add_required_to_config()
        self.config = self.projectConfig
        self.dataset = dataset
        self.model = model
        self.agent = agent
        self.storage = {}
        self.model_state = "train"
        self.general()
        if (os.path.isdir(os.path.join(self.config['model_path'], 'models'))):
            self.reload()
        else:
            self.setup()
            # Load pretrained model
            if 'pretrained' in self.config and self.currentStep == 0:
                self.load_model()

        # self.initializeFID()
        self.currentStep = self.currentStep + 1
        # self.set_current_config()

    def add_required_to_config(self) -> None:
        r"""Fills config with basic setup if not defined otherwise
        """
        if 'Persistence' not in self.projectConfig:
            self.projectConfig['Persistence'] = False
        if 'batch_duplication' not in self.projectConfig:
            self.projectConfig['batch_duplication'] = 1
        if 'keep_original_scale' not in self.projectConfig:
            self.projectConfig['keep_original_scale'] = False
        if 'rescale' not in self.projectConfig:
            self.projectConfig['rescale'] = True
        if 'channel_n' not in self.projectConfig:
            self.projectConfig['channel_n'] = 16
        if 'cell_fire_rate' not in self.projectConfig:
            self.projectConfig['cell_fire_rate'] = 0.5
        if 'output_channels' not in self.projectConfig:
            self.projectConfig['output_channels'] = 1
        if 'description' not in self.projectConfig:
            self.projectConfig['description'] = "None"

        # Basic Configs
        if 'model_path' not in self.projectConfig:
            self.projectConfig['model_path'] = os.path.join(PATH, 'Experiments',
                                                            self.projectConfig['name'] + "_" + self.projectConfig[
                                                                'description'])
        if 'generate_path' not in self.projectConfig:
            self.projectConfig['generate_path'] = os.path.join(PATH, 'Experiments', self.projectConfig['name'],
                                                               'Generated')

    def set_loss_function(self, loss_function) -> None:
        self.loss_function = loss_function

    def set_data_loader(self, data_loader) -> None:
        self.data_loader = data_loader

    def setup(self) -> None:
        r"""Initial experiment setup when first started
        """
        # Create dirs
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(os.path.join(self.config['model_path'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.config['model_path'], 'inference'), exist_ok=True)

        # Create basic configuration
        self.data_split = self.new_datasplit()
        self.set_model_state("train")

    def new_datasplit(self) -> 'DataSplit':
        return DataSplit(self.config['img_path'], self.config['label_path'], data_split=self.config['data_split'],
                         dataset=self.dataset)

    def temporarly_overwrite_config(self, config: dict):
        r"""This function is useful for evaluation purposes where you want to change the config, e.g. data paths or similar.
            It does not save the config and should NEVER be used during training.
        """
        print("WARNING: NEVER USE \'temporarly_overwrite_config\' FUNCTION DURING TRAINING.")
        self.projectConfig = config
        # self.set_current_config()
        self.data_split = self.new_datasplit()
        self.set_size()

    def get_max_steps(self) -> int:
        r"""Get max defined training steps of experiment
        """
        return self.projectConfig['n_epoch']

    def load_model(self) -> None:
        if 'pretrained' in self.config and self.currentStep == 0:
            print('>>>>>> Load Pretrained Model <<<<<<')
            pretrained_path = os.path.join(PATH, 'Experiments',
                                           self.config['pretrained'] + "_" + self.projectConfig['description'])
            pretrained_step = self.current_step(
                os.path.join(pretrained_path, 'models'))  # os.path.join(self.config['model_path'], 'models')
            model_path = os.path.join(pretrained_path, 'models', 'epoch_' + str(pretrained_step))
        else:
            model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep))
        print(model_path)
        if os.path.exists(model_path):
            print("Reload State " + str(self.currentStep))
            self.agent.load_state(model_path)  # is not True:
            #    raise Exception("Model could not be loaded. Check if folder contains weights and architecture is identical.")

    def set_size(self) -> None:
        if isinstance(self.config['input_size'][0], tuple):
            self.dataset.set_size(self.config['input_size'][-1])
        else:
            print(self.config['input_size'])
            self.dataset.set_size(self.config['input_size'])

    def general(self) -> None:
        r"""General experiment configurations needed after setup or loading
        """
        self.currentStep = self.current_step()
        self.set_size()
        # self.writer = SummaryWriter(log_dir=os.path.join(self.get_from_config('model_path'), 'tensorboard', os.path.basename(self.get_from_config('model_path'))))
        # self.set_current_config()
        self.agent.set_exp(self)
        self.fid = None
        self.kid = None
        self.dataset.set_experiment(self)
        # if self.currentStep == 0:
        #    self.write_text('config', str(self.projectConfig), 0)

        if self.get_from_config('unlock_CPU') is None or self.get_from_config('unlock_CPU') is False:
            print(
                'In basic configuration threads are limited to 1 to limit CPU usage on shared Server. Add \'unlock_CPU:True\' to config to disable that.')
            torch.set_num_threads(4)

    def bufferData(self) -> None:
        self.set_model_state("train")
        dataloader_fid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
        for i, data in tqdm(enumerate(dataloader_fid)):
            continue
        self.set_model_state("val")
        dataloader_fid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
        for i, data in tqdm(enumerate(dataloader_fid)):
            continue
        self.set_model_state("test")
        dataloader_fid = torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=2048)
        for i, data in tqdm(enumerate(dataloader_fid)):
            continue

    def reload_model(self) -> None:
        r"""Reload model
            TODO: Move to a more logical position. Probably to the model and then call directly from the agent
        """
        if 'pretrained' in self.config and self.current_step == 0:
            print('Load Pretrained Model')
            pretrained_path = os.path.join(PATH, 'Experiments',
                                           self.config['pretrained'] + "_" + self.projectConfig['description'])
            pretrained_step = self.current_step(model_path=os.path.join(self.config['pretrained_path'],
                                                                        'models'))  # os.path.join(self.config['model_path'], 'models')
            model_path = os.path.join(pretrained_path, 'models', 'epoch_' + str(pretrained_step), 'model.pth')
        else:
            model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep),
                                      'model.pth')
        if os.path.exists(model_path):
            self.agent.load_model(model_path)

    def save_model(self) -> None:
        r"""TODO: Same as for reload -> move to better location
        """
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep + 1))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))

    def current_step(self, model_path: str = None) -> int:
        r"""Find out the initial epoch by checking the saved models"""
        if model_path is None:
            model_path = os.path.join(self.config['model_path'], 'models')
        if os.path.exists(model_path):
            dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            if dirs:
                maxDir = max([int(d.split('_')[1]) for d in dirs])
                return maxDir
        return 0

    def set_model_state(self, state: str) -> None:
        r"""TODO: remove? """
        self.model_state = state
        self.dataset.setPaths(self.config['img_path'], self.data_split.get_images(state), self.config['label_path'],
                              self.data_split.get_labels(state))
        self.dataset.setState(state)

        models = [self.model] if not isinstance(self.model, list) else self.model
        for m in models:
            if self.model_state == "train":
                m.train()
            else:
                m.eval()

    def get_from_config(self, tag: str) -> any:
        r"""Get from config
            #Args
                tag (String): Key of requested value
        """
        if tag in self.config.keys():
            return self.config[tag]
        else:
            return None

    @DeprecationWarning
    def set_current_config(self) -> None:
        r"""Set current config. This can change during training and will always
            overwrite previous settings, but keep everything else
        """
        self.config = {}
        for i in range(0, len(self.projectConfig)):
            for k in self.projectConfig[i].keys():
                self.config[k] = self.projectConfig[i][k]
            if self.projectConfig[i]['n_epoch'] > self.currentStep:
                return

    def increase_epoch(self) -> None:
        r"""Increase current epoch
        """
        self.currentStep = self.currentStep + 1
        # self.set_current_config()

    def get_current_config(self) -> dict:
        r"""TODO: remove?"""
        return self.config


class DataSplit():
    r"""Handles the splitting of data
    """

    def __init__(self, path_image: str, path_label: str, data_split: dict, dataset: Dataset):
        self.images = self.split_files(self.getFilesInFolder(path_image, dataset), data_split)
        self.labels = self.split_files(self.getFilesInFolder(path_label, dataset), data_split)

    def get_images(self, state: str) -> dict:
        r"""#Returns the images of selected state
            #Args
                state (String): Can be 'train', 'val', 'test'
        """
        return self.get_data(self.images[state])

    def get_labels(self, state: str) -> dict:
        r"""#Returns the labels of selected state
            #Args
                state (String): Can be 'train', 'val', 'test'
        """
        return self.get_data(self.labels[state])

    def get_data(self, data: dict) -> list:
        r"""#Returns the data in a list rather than the stored folder strucure
            #Args
                data ({}): Dictionary ordered by {id, {slice, img_name}}
        """
        lst = data.values()
        lst_out = []
        for d in lst:
            lst_out.extend([*d.values()])
        return lst_out

    def split_files(self, files: dict, data_split: list) -> dict:
        r"""Split files into train, val, test according to definition
            while keeping patients slics together.
            #Args
                files ({int, {int, string}}): {id, {slice, img_name}}
                data_split ([float, float, float]): Sum of 1
        """
        dic = {'train': {}, 'val': {}, 'test': {}}
        for index, key in enumerate(files):
            if index / len(files) < data_split[0]:
                dic['train'][key] = files[key]
            elif index / len(files) < data_split[0] + data_split[1]:
                dic['val'][key] = files[key]
            else:
                dic['test'][key] = files[key]
        print("Datasplit-> train entries: {}, val entries: {}, test entries: {}".format(len(dic['train']),
                                                                                        len(dic['val']),
                                                                                        len(dic['test'])))
        return dic

    def getFilesInFolder(self, path: str, dataset: Dataset) -> list:
        r"""Get files in folder
            #Args
                path (String): Path to folder
                dataset (Dataset)
        """
        return dataset.getFilesInPath(path)


def merge_config(config_parent: dict, config_child: dict) -> None:
    r"""Merge config with current config
    """
    # try:
    #    config_child['name'] = config_parent['name'] + "_" + config_child['name']
    # except:
    #    print('MISSING NAME IN CONFIG')
    return {**config_parent, **config_child}
