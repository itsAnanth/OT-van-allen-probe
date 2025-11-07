import torch
import os
import argparse
import matplotlib.pyplot as plt
from common.config import load_config
from pathlib import Path



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    args, remaining_args = parser.parse_known_args()

    assert remaining_args == [], remaining_args
    assert args.checkpoint_dir is not None, "checkpoint directory cannot be None"

    folder = Path(args.checkpoint_dir)
    files = list(folder.glob("*.pth")) 
    print("debug", len(files) + 1)

    config = load_config("/".join(folder.parts))

    rscores = []
    train_loss = []
    val_loss = []

    for i, file in enumerate(files):
        print(f"Loading checkpoint no. {i + 1}")
        data = torch.load(file, map_location=torch.device('cpu'), weights_only=True)
        
        rscores.append(data['metrics']['rscore'])
        val_loss.append(data['metrics']['val_loss'])
        train_loss.append(data['metrics']['train_loss'])

    print(rscores)


    os.makedirs(f'plots/{"_".join(folder.parts[-2:])}', exist_ok=True)


    plt.plot(range(1, len(rscores) + 1), rscores)
    plt.xticks(range(1, len(rscores) + 1))
    plt.grid()
    plt.title('rscores vs epochs')
    plt.savefig(f'plots/{"_".join(folder.parts[-2:])}/rscore.jpg', dpi=300)
    plt.clf()
    
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='val')
    plt.xticks(range(1, len(train_loss) + 1))
    plt.grid()
    plt.title('losses')
    plt.legend()
    plt.savefig(f'plots/{"_".join(folder.parts[-2:])}/loss.jpg', dpi=300)






