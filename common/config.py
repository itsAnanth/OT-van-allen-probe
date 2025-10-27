from dataclasses import dataclass


CHANNEL_MAP = { 
    3: {
        'name': '54kev',
        'positional_features': ["ED_R_OP77Q_intxt", "ED_MLAT_OP77Q_intxt", "ED_MLT_OP77Q_intxt_sin", "ED_MLT_OP77Q_intxt_cos"],
        'time_series_features': ["AE_INDEX", "flow_speed", "SYM_H", "Pressure"]
    }, 
    11: {
        'name': '235kev',
        'positional_features': None,
        'time_series_features': None    
    }, 
    14: {
        'name': '597kev',
        'positional_features': None,
        'time_series_features': None    
    }, 
    16: {
        'name': '909kev',
        'positional_features': None,
        'time_series_features': None    
    } 
}

@dataclass
class Config:
    
    data_dir: str = 'dataset/preprocessed'
    
    checkpoint: bool = False
    checkpoint_dir: str = 'checkpoints'
    load_from_checkpoint: bool = False
    channel: int = 3
    channel_name: str = CHANNEL_MAP[channel]['name']
    channel_data = CHANNEL_MAP[channel]
    
    
    # hyperparams
    lr: int = 1e-3
    max_epochs: int = 10
    seq_length: int = 1500
    batch_size: int = 64
    hidden_size: int = 512
    
    gpu: int = 0
    tune: bool = False
    tune_param: str = None
    data_limt: bool = False
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        
        if name == "channel":
            super().__setattr__("channel_name", CHANNEL_MAP[value]['name'])
            super().__setattr__("channel_data", CHANNEL_MAP[value])