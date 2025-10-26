from dataclasses import dataclass

@dataclass
class Config:
    
    checkpoint: bool = False
    checkpoint_dir: str = 'checkpoints'