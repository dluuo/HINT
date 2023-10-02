#---------------------------------------------------------
#Code is from https://github.com/AdityaLab/PROFHiT
#PROFHIT: Probabilistic Robust Forecasting for Hierarchical Time-series
#Kamarthi, Harshavardhan and Kong, Lingkai and Rodriguez, Alexander and Zhang, Chao and Prakash, B Aditya
#---------------------------------------------------------

#----------------------------IMPORTS-----------------------------
from importlib.metadata import metadata
import numpy as np
import pandas as pd
import torch as th
from torch.nn.functional import mse_loss
from models.fnpmodels import EmbedMetaAttenSeq, RegressionSepFNP, Corem
from utils import lag_dataset, get_hmatrix, calc_bootstrap, jsd_loss
from models.utils import float_tensor, long_tensor
from tqdm import tqdm
import properscoring as ps

#import wrangled datasets
import sys
sys.path = ['../'] + sys.path
from data.data import HierarchicalDataset
import argparse

# Create an argument parser to accept command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Name of the dataset')
parser.add_argument('pretrain_input', type=int, help='# Pretrain Epochs', default=10)
parser.add_argument('train_input', type=int, help='# Train Epochs', default=100)

# Parse the command-line arguments
args = parser.parse_args()
DATASET = args.dataset
PRETRAIN_INPUT = args.pretrain_input
TRAIN_INPUT = args.train_input

#----------------------------PARAMETERS-----------------------------


print("Using dataset: ", DATASET)

hdataset = HierarchicalDataset()
data = hdataset.load_process_data(dataset=DATASET)
#import Y_df and remove future df NaNs
Y_df = data['Y_df']
S_df = data['S_df']
n_series = len(S_df.index)
Y_np = Y_df.y.values.reshape(n_series,-1)

# S_df, properties = data['S'], data['properties']
S_np = S_df.values
if DATASET == 'TourismLarge':
    # split up agg matrix into two
    S_np_1 = np.zeros_like(S_np)
    S_np_2 = np.zeros_like(S_np)
    S_np_1[:111, :] = S_np[:111, :]
    S_np_2[0, :] = S_np[0, :]
    S_np_2[1:445, :] = S_np[111:, :]
    hmatrix = float_tensor(get_hmatrix(S_np_1))
    hmatrix2 = float_tensor(get_hmatrix(S_np_2))
else:
    hmatrix = float_tensor(get_hmatrix(S_np))

SEASONALITY, HORIZON = data['seasonality'], data['horizon']
HIER_IDXS = data['hier_idxs']
LEVEL = np.arange(0, 100, 2)
qs = [[50-lv/2, 50+lv/2] for lv in LEVEL]
QUANTILES = np.sort(np.concatenate(qs)/100)
TOTAL_TIMESTEPS = Y_np.shape[1]

#------

BACKUP_TIMES = {
    'Labour': 50, 
    'Traffic': 50,
    'TourismSmall': 30,
    'TourismLarge': 30,
    'Wiki2': 50
}

PRE_TRAIN_DICT = {
    'Labour': 10, 
    'Traffic': 10,
    'TourismSmall': 10,
    'TourismLarge': 10,
    'Wiki2': 10
}

TRAIN_DICT = {
    'Labour': 100, 
    'Traffic': 100,
    'TourismSmall': 20,
    'TourismLarge': 100,
    'Wiki2': 100
}


SEED = 42

AHEAD = SEASONALITY
TRAIN_UPTO = TOTAL_TIMESTEPS - SEASONALITY
BACKUP_TIME = BACKUP_TIMES[DATASET]
PRE_BATCH_SIZE = 10
PRE_TRAIN_LR = 0.001
# PRE_TRAIN_EPOCHS = PRE_TRAIN_DICT[DATASET]
PRE_TRAIN_EPOCHS = PRETRAIN_INPUT
FRAC_VAL = 0.1
C = 5.0
BATCH_SIZE = 10
TRAIN_LR = 0.001
LAMBDA = 0.0
# TRAIN_EPOCHS = TRAIN_DICT[DATASET]
TRAIN_EPOCHS = TRAIN_INPUT
EVAL_SAMPLES = 100

np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)


if th.cuda.is_available():
    device = th.device("cuda")
else:
    device = th.device("cpu")

print("Using device: ", device)

# Let's create dataset
full_data = Y_np
train_data_raw = full_data[:, :TRAIN_UPTO]
train_means = np.mean(train_data_raw, axis=1)
train_std = np.std(train_data_raw, axis=1)
train_data = (train_data_raw - train_means[:, None]) / train_std[:, None]

dataset_raw = lag_dataset(train_data, BACKUP_TIME)



#----------------------------MODEL-----------------------------

class SeqDataset(th.utils.data.Dataset):
    def __init__(self, dataset):
        self.X, self.Y = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dataset[idx]


dataset = SeqDataset(dataset_raw)


# Let's create FNP model
encoder = EmbedMetaAttenSeq(
    dim_seq_in=1,
    # num_metadata=len(data_obj.idx_dict),
    num_metadata=n_series,
    dim_metadata=1,
    dim_out=60,
    n_layers=2,
    bidirectional=True,
).to(device)
decoder = RegressionSepFNP(
    dim_x=60,
    dim_y=1,
    dim_h=60,
    n_layers=3,
    dim_u=60,
    dim_z=60,
    # nodes=len(data_obj.idx_dict),
    nodes=n_series,
).to(device)


pre_opt = th.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=PRE_TRAIN_LR
)

# Create training validation set
perm = np.random.permutation(np.arange(BACKUP_TIME, TRAIN_UPTO))
train_idx = perm[: int(len(perm) * (1 - FRAC_VAL))]
val_idx = perm[int(len(perm) * (1 - FRAC_VAL)) :]


def pretrain_epoch():
    encoder.train()
    decoder.train()
    losses = []
    means, stds = [], []
    pre_opt.zero_grad()
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(train_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample, logstd_sample, log_py, log_pqz, _ = decoder(ref_out_x, out_x, y)
        loss = -(log_py + log_pqz) / x.shape[0]
        loss.backward()
        losses.append(loss.detach().cpu().item())
        means.append(mean_sample.detach().cpu().numpy())
        stds.append(logstd_sample.detach().cpu().numpy())
        if (i + 1) % PRE_BATCH_SIZE == 0:
            pre_opt.step()
            pre_opt.zero_grad()
    if i % PRE_BATCH_SIZE != 0:
        pre_opt.step()
    return np.mean(losses), np.array(means), np.array(stds)


def pre_validate(sample=False):
    encoder.eval()
    decoder.eval()
    losses = []
    means, stds = [], []
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(val_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=sample)
        mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
        losses.append(mse_loss)
        means.append(mean_y.detach().cpu().numpy())
        stds.append(logstd_y.detach().cpu().numpy())
    return np.mean(losses), np.array(means), np.array(stds)


print("Pretraining...")
for ep in tqdm(range(PRE_TRAIN_EPOCHS)):
    loss, means, stds = pretrain_epoch()
    print(f"Epoch {ep} loss: {loss}")
    with th.no_grad():
        loss, means, stds = pre_validate()
        print(f"Epoch {ep} Val loss: {loss}")

# Let's real_train
corem = Corem(nodes=n_series, c=C,).to(device)

opt = th.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(corem.parameters()),
    lr=TRAIN_LR,
)


def train_epoch():
    encoder.train()
    decoder.train()
    corem.train()
    losses = []
    means, stds, gts = [], [], []
    opt.zero_grad()
    ref_x = float_tensor(train_data[:, :, None])
    th_means = float_tensor(train_means)
    th_std = float_tensor(train_std)
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(train_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample1, logstd_sample1, log_py1, log_pqz, py1 = decoder(
            ref_out_x, out_x, y
        )
        mean_sample, logstd_sample, log_py, py = corem(
            mean_sample1.squeeze(), logstd_sample1.squeeze(), y
        )
        loss1 = -(log_py + log_pqz) / x.shape[0]
        loss2 = (
            jsd_loss(
                mean_sample.squeeze(),
                logstd_sample.squeeze(),
                hmatrix,
                th_means,
                th_std,
            )
            / x.shape[0]
        )
        loss3 = 0
        if (DATASET == 'TourismLarge'):
            loss3 = (
                jsd_loss(
                    mean_sample.squeeze(),
                    logstd_sample.squeeze(),
                    hmatrix2,
                    th_means,
                    th_std,
                )
                / x.shape[0]
            )
        loss = loss1 + LAMBDA * (loss2+loss3)
        if th.isnan(loss):
            import pdb
            pdb.set_trace()
        loss.backward()
        losses.append(loss.detach().cpu().item())
#         print(f"Loss1: {loss1.detach().cpu().item()}")
#         print(f"Loss2: {loss2.detach().cpu().item()}")
        means.append(mean_sample.detach().cpu().numpy())
        stds.append(logstd_sample.detach().cpu().numpy())
        gts.append(y.detach().cpu().numpy())
        if (i + 1) % BATCH_SIZE == 0:
            opt.step()
            opt.zero_grad()
    if i % BATCH_SIZE != 0:
        opt.step()
    return np.mean(losses), np.array(means), np.array(stds)


def validate(sample=False):
    encoder.eval()
    decoder.eval()
    corem.eval()
    losses = []
    means, stds, gts = [], [], []
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(val_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
        y_pred, mean_y, logstd_y, _ = corem.predict(
            mean_y.squeeze(), logstd_y.squeeze(), sample=sample
        )
        mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
        losses.append(mse_loss)
        means.append(mean_y.detach().cpu().numpy())
        stds.append(logstd_y.detach().cpu().numpy())
        gts.append(y.cpu().numpy())
    return np.mean(losses), np.array(means), np.array(stds)


print("Training....")
for ep in tqdm(range(TRAIN_EPOCHS)):
    loss, means, stds = train_epoch()
    print(f"Epoch {ep} loss: {loss}")
    with th.no_grad():
        loss, means, stds = validate()
        print(f"Epoch {ep} Val loss: {loss}")

# Lets evaluate

# One sampple
def sample_data():
    curr_data = train_data.copy()
    encoder.eval()
    decoder.eval()
    corem.eval()
    for t in range(AHEAD):
        ref_x = float_tensor(train_data[:, :, None])
        meta_x = long_tensor(np.arange(ref_x.shape[0]))
        x = float_tensor(curr_data[:, :, None])
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
        y_pred, mean_y, logstd_y, _ = corem.predict(
            mean_y.squeeze(), logstd_y.squeeze(), sample=True
        )
        y_pred = y_pred.cpu().numpy()
        curr_data = np.concatenate([curr_data, y_pred], axis=1)
    return curr_data[:, -AHEAD:]


ground_truth = full_data[:, TRAIN_UPTO : TRAIN_UPTO + AHEAD]

crps_list = []
msse_list = []
for _ in range(5):
    with th.no_grad():
        preds = [sample_data() for _ in tqdm(range(EVAL_SAMPLES))]
    preds = np.array(preds)
    preds = preds * train_std[:, None] + train_means[:, None]
    mean_preds = np.mean(preds, axis=0)
    
    quantile_preds = np.quantile(preds, q=QUANTILES, axis=0)
    reshaped_preds = np.transpose(quantile_preds, (1, 2, 0)) #reshape into n_series, horizon, quantiles

    crps_all_levels = hdataset._get_hierarchical_scrps(Y=ground_truth, Yq_hat=reshaped_preds, q_to_pred=QUANTILES, hier_idxs=HIER_IDXS)
    msse_all_levels = hdataset._get_hierarchical_msse(Y=ground_truth, Y_hat=mean_preds, Y_train=train_data_raw, hier_idxs=HIER_IDXS)

    crps_list.append(crps_all_levels)
    msse_list.append(msse_all_levels)

crps_bootstrap = calc_bootstrap(crps_list)
msse_bootstrap = calc_bootstrap(msse_list)

print(f"CRPS BOOTSTRAP: {crps_bootstrap}")
print(f"MSSE BOOTSTRAP: {msse_bootstrap}")
