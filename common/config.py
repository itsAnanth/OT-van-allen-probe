import json
import os
from dataclasses import dataclass, asdict
from common.utils import autodetect_device


def get_model_class(channel):
    """
    Returns the appropriate model class for a given channel.
    Each channel has its own LSTM model with channel-specific MLP head.
    """
    if channel == 3:
        from models.lstm_54 import FluxLSTM
        return FluxLSTM
    elif channel == 11:
        from models.lstm_235 import FluxLSTM_235keV
        return FluxLSTM_235keV
    elif channel == 14:
        from models.lstm_597 import FluxLSTM_597keV
        return FluxLSTM_597keV
    elif channel == 16:
        from models.lstm_909 import FluxLSTM_909keV
        return FluxLSTM_909keV
    else:
        raise ValueError(f"Invalid channel {channel}. Must be one of [3, 11, 14, 16]")


CHANNEL_MAP = { 
    3: {
        'name': '54kev',
        'positional_features': ["ED_R_OP77Q_intxt", "ED_MLAT_OP77Q_intxt", "ED_MLT_OP77Q_intxt_sin", "ED_MLT_OP77Q_intxt_cos"],
        'time_series_features': ["AE_INDEX", "flow_speed", "SYM_H", "Pressure"],
        # 'best_params': {
        #     'seq_length': 1500,
        #     'hidden_size': 512
        # }
        'best_params': {} 
    }, 
    11: {
        'name': '235kev',
        'positional_features': ["ED_R_OP77Q_intxt", "ED_MLAT_OP77Q_intxt", "ED_MLT_OP77Q_intxt_sin", "ED_MLT_OP77Q_intxt_cos"],
        'time_series_features': ["AL_INDEX", "flow_speed", "SYM_H", "Pressure"],
        'best_params': {}   
    }, 
    14: {
        'name': '597kev',
        'positional_features':["ED_R_OP77Q_intxt", "ED_MLAT_OP77Q_intxt", "ED_MLT_OP77Q_intxt_sin", "ED_MLT_OP77Q_intxt_cos"],
        'time_series_features': ["AL_INDEX","SYM_H","Pressure","flow_speed"],
        'best_params': {}   
    }, 
    16: {
        'name': '909kev',
        'positional_features': ["ED_R_OP77Q_intxt", "ED_MLAT_OP77Q_intxt", "ED_MLT_OP77Q_intxt_sin", "ED_MLT_OP77Q_intxt_cos"],
        'time_series_features':["AL_INDEX","SYM_H","flow_speed","Pressure"],
        'best_params': {}    
    } 
}

@dataclass
class Config:
    
    description: str = ""
    data_dir: str = 'dataset/preprocessed'
    
    save_checkpoint: bool = False
    checkpoint_dir: str = 'checkpoints'
    load_from_checkpoint: int = -1
    channel: int = 3
    channel_name: str = CHANNEL_MAP[channel]['name']
    channel_data = CHANNEL_MAP[channel]
    device=None
    
    
    # hyperparams
    lr: int = 1e-3
    max_epochs: int = 10
    seq_length: int = 1500
    batch_size: int = 64
    hidden_size: int = 512
    
    gpu: int = 0
    tune: bool = False
    tune_param: str = None
    data_limit: bool = False

    def __post_init__(self):
        self.device = autodetect_device(self)

        self.channel_name = CHANNEL_MAP[self.channel]['name']
        self.channel_data = CHANNEL_MAP[self.channel]
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        print("Setting attribute", name, "to", value)
        
        if name == "channel":
            super().__setattr__("channel_name", CHANNEL_MAP[value]['name'])
            super().__setattr__("channel_data", CHANNEL_MAP[value])


        
    


def save_config(config):

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Convert to dict then JSON and save
    with open(f"{config.checkpoint_dir}/config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

def load_config(config_dir):
    with open(os.path.join(config_dir, "config.json"), "r") as f:
        cfg_dict = json.load(f)

    # Recreate Config object
    cfg = Config(**cfg_dict)
    return cfg
