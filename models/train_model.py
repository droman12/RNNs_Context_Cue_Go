import time
import torch
from models.model import *
from plotting_functions import*
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm     import ScalarMappable
from torch.utils.tensorboard import SummaryWriter
from models.hp import *
import random, numpy as np

def main():
    writer = SummaryWriter("tanh_less_noiss/CTRNN")
    hp = get_default_hp(2, 1, activation = 'tanh')
    params = get_default_params(batch_size=hp['batch_size'], test= False)
    # seed everything
    random.seed(hp['seed'])
    np.random.seed(hp['seed'])
    torch.manual_seed(hp['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if device.type=='cuda':
        torch.cuda.manual_seed_all(hp['seed'])

    run_model = Run_Model(hp, params, RNNLayer).to(device)
    optim = torch.optim.Adam(
        run_model.parameters(),
        lr=hp['learning_rate'],
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
        factor=hp['lr_factor'],
        patience=hp['lr_patience'],
        verbose=True
    )

    best_val_loss = float('inf')
    X_val, Y_val = run_model.generate_trials(hp['batch_size_val'])

    for epoch in range(1, hp['n_epochs']+1):
        t0 = time.time()
        run_model.train()
        optim.zero_grad()

        total_loss, data_loss, reg_loss, _, _ = run_model(batch_size=hp['batch_size'])
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(run_model.parameters(), hp['grad_clip'])
        optim.step()

        writer.add_scalar("Loss/train_total", total_loss.item(), epoch)

        # validation
        run_model.eval()
        with torch.no_grad():
            total_val, data_val, reg_val, pred_seq, _ = run_model(
                batch_size=hp['batch_size_val'],
                X=X_val, Y=Y_val
            )
        data_val = data_val.item()
        total_val = total_val.item()
        scheduler.step(data_val)
        
        writer.add_scalar("Loss/val_total", total_val, epoch)
        writer.add_scalar("LR",             optim.param_groups[0]['lr'], epoch)

        if epoch % 10 == 0:
            fig = plot_output_targets(pred_seq, Y_val)
            writer.add_figure("Output vs Target", fig, epoch)
            print(f"[{epoch:04d}] trn={data_loss:.4f}|{total_loss:.4f}  "
                  f"val={data_val:.4f}|{total_val:.4f}  "
                  f"lr={optim.param_groups[0]['lr']:.1e}  t={time.time()-t0:.2f}s")

        # checkpoint on validation data-loss
        if data_val < best_val_loss:
            best_val_loss = data_val
            torch.save(run_model.state_dict(), f"{hp['save_name']}_best.pt")

        if best_val_loss < hp['target_perf']:
            print(f"Reached target perf ({best_val_loss:.4f} < {hp['target_perf']:.4f}), stopping.")
            break

    writer.close()

if __name__ == "__main__":
    main()
