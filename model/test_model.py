from input_target import *
from model.hp import *
from model.model import *
from plotting.plot_trajectories import *

start_color = (1.0, 0.549, 0.0)
end_color   = (0.392, 0.584, 0.929)
saved_model_path = r"C:\Users\roman\Documents\task_mice - Copy\model_relu_best.pt"

params = get_default_params(batch_size=16, test = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hp_tanh = get_default_hp(2, 1, activation = 'relu', test = True)
model_tanh = Run_Model(hp_tanh, params, RNNLayer=RNNLayer).to(device)
model_tanh.load_state_dict(torch.load(saved_model_path, map_location = device, weights_only =False))

X_test, Y_test = model_tanh.generate_trials(batch_size=16)
inputs = X_test.detach().cpu().numpy()

model_tanh.eval()
with torch.no_grad():
    total_tanh, data_loss_tanh, reg_loss_tanh, pred_tanh, hid= model_tanh( X=X_test, Y=Y_test)
    fig_tanh = plot_output_targets(pred_tanh, Y_test)
    plt.show()


