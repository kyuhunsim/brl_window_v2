import os
import shutil

import torch

from typing import Dict, Any, Optional
from datetime import datetime
import pickle

from pneu_utils.utils import get_pkg_path, color

class Logger():
    def __init__(
        self,
        save_name: Optional[str] = None
    ):
        self.model_name = save_name
        self.folder_path = f'{get_pkg_path("pneu_rl")}/models/{save_name}'
        self.infos = {}
        if not os.path.isdir(self.folder_path):
            self.create_folder(self.folder_path)
        
    def create_folder(
        self,
        folder_path: str
    ) -> None:
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except OSError:
            print('Error!')
    
    def _remove_folder(
        self,
        folder_path: str
    ):
        shutil.rmtree(folder_path)
    
    def save_params(
        self,
        kwargs: Dict[str, Any]
    ):
        with open(self.param_path, 'wb') as f:
            pickle.dump(kwargs, f)
    
    def load_infos(
        self,
        save_name,
    ):
        self.folder_path = f'{get_pkg_path("pneu_rl")}/models/{save_name}'
        
        with open(self.info_path, 'rb') as f:
            self.infos = pickle.load(f)
        
        last_epi = list(self.infos.keys())[-1]
        last_steps = list(self.infos.values())[-1]['steps']

        return (
            last_epi,
            last_steps
        )
    
    def save_infos(
        self,
        epi: int,
        reward: float,
        step: Optional[int] = None,
        alpha: Optional[torch.Tensor] = None,
        temporal_weight: Optional[float] = None,
        critic_loss : Optional[float] = None,
        policy_loss : Optional[float] = None
    ):
        self.infos[epi] = dict()
        self.infos[epi]['reward'] = reward
        self.infos[epi]['steps'] = step
        if alpha is not None:
            self.infos[epi]['alpha'] = alpha
        self.infos[epi]['temporal_weight'] = temporal_weight

        if critic_loss is not None:
            self.infos[epi]['critic_loss'] = critic_loss
        if policy_loss is not None:
            self.infos[epi]['policy_loss'] = policy_loss
            
        with open(self.info_path,'wb') as f:
            pickle.dump(self.infos, f)
    
    def set_retrain_model(
        self,
        is_model_loaded: bool = False,
        retrain_model_name: Optional[str] = None
    ) -> str:
        assert is_model_loaded, color('[ERROR] Model is not loaded!','red')
        
        if (retrain_model_name is None) or (len(retrain_model_name) == 0):            
            parent_folder_path = f'{get_pkg_path("pneu_rl")}/models'

            model_version_name = self.model_name.split('_')[-1] # v000
            model_name = '_'.join(self.model_name.split('_')[:-1])

            retrain_model_name = f'{self.model_name}_retrain'
            if model_version_name[0] == 'v': # check model version name is v000
                model_version = int(model_version_name[1:])
                retrain_version = model_version + 1
                retrain_model_name = f'{model_name}_v{retrain_version:02}'

                while os.path.exists(f'{parent_folder_path}/{retrain_model_name}'):
                    retrain_version += 1
                    retrain_model_name = f'{model_name}_v{retrain_version:02}'

        assert self.model_name != retrain_model_name, color('[ERROR] Model name and retrain model name should be different!', 'red')
        retrain_model_path = f"{get_pkg_path('pneu_rl')}/models/{retrain_model_name}"

        shutil.copytree(
            f"{get_pkg_path('pneu_rl')}/models/{self.model_name}", 
            retrain_model_path
        )

        return retrain_model_name
    
    @property
    def param_path(self) -> str:
        return f'{self.folder_path}/params.pkl'

    @property
    def model_path(self) -> str:
        return f'{self.folder_path}/model.pth'
    
    @property
    def info_path(self) -> str:
        return f'{self.folder_path}/infos.pkl'
    
    @property
    def buffer_path(self) -> str:
        return f'{self.folder_path}/buffer.pkl'
    
    def check_value(
        self
    ):
        with open(f'{self.folder_path}/rewards.pkl', 'rb') as f:
            self.rewards = pickle.load(f)
            print(self.rewards)

if __name__ == '__main__':
    logger = Logger()
    
    # init_params, model_path = logger.load_init_params_and_model_path('test')
    print(logger.model_path)
    logger.folder_path = logger._set_folder_path('test')
    print(logger.model_path)
