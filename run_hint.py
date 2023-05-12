import numpy as np
import pandas as pd

from src.data.data import HierarchicalDataset

def evaluate_forecast(nf, hdataset, Y_train_df, Y_test_df, hier_idxs, hier_levels, n_samples):
    # Extract hint.model from nf core
    model_name = type(nf.models[0]).__name__
    model = nf.models[0].model

    # Parse Yq_hat_sample and Y_test
    # Keep only quantile columns from Y_hat_df
    # Removing mean and median default outputs
    Y_hat_df_list = [nf.predict() for _ in range(n_samples)]
    model_columns = [model_name + n for n in model.loss.output_names]

    # Parse shapes
    n_series = len(Y_test_df.unique_id.unique())
    horizon = len(Y_test_df.ds.unique())   
    n_quantiles = len(model_columns)-1
    
    Y_hat_sample = []
    for sample_idx in range(n_samples):
        Y_hat = Y_hat_df_list[sample_idx][model_columns].values
        Y_hat_sample.append(Y_hat.reshape(1, n_series, horizon, -1))
    Y_hat_sample = np.concatenate(Y_hat_sample, axis=0)
    Ymean_hat_sample = Y_hat_sample[:,:,:,0] # Default mean
    Yq_hat_sample = Y_hat_sample[:,:,:,3:] # Drop mean/median default
    Y_test = Y_test_df['y'].values.reshape(n_series, horizon)
    Y_train = Y_train_df['y'].values.reshape(n_series, -1)

    # Compute bootstraped sCRPS, mean and 95% confidence intervals
    sample_scrps = []
    sample_relmse = []
    n_samples = Yq_hat_sample.shape[0]
    for sample_idx in range(n_samples):
        relmse = hdataset._get_hierarchical_msse(hier_idxs=hier_idxs,
                                          Y=Y_test,
                                          Y_hat=Ymean_hat_sample[sample_idx],
                                          Y_train=Y_train)
        sample_relmse.append(relmse)

        scrps = hdataset._get_hierarchical_scrps(hier_idxs=hier_idxs,
                                        Y=Y_test,
                                        Yq_hat=Yq_hat_sample[sample_idx],
                                        q_to_pred=quantiles)
        sample_scrps.append(scrps)

    mean_relmse = np.mean(np.array(sample_relmse), axis=0)
    std_relmse  = np.std(np.array(sample_relmse), axis=0)
    relmse_results = [f'{mean_relmse[level_idx]:.4f}±{(1.96 * std_relmse[level_idx]):.4f}' \
                     for level_idx in range(len(mean_relmse))]

    mean_scrps = np.mean(np.array(sample_scrps), axis=0)
    std_scrps  = np.std(np.array(sample_scrps), axis=0)
    scrps_results = [f'{mean_scrps[level_idx]:.4f}±{(1.96 * std_scrps[level_idx]):.4f}' \
                     for level_idx in range(len(mean_scrps))]
    
    evaluation_df = dict(Levels=hier_levels,
                         sCRPS=scrps_results,
                         relMSE=relmse_results)
    evaluation_df = pd.DataFrame(evaluation_df)
    return evaluation_df

if __name__ == '__main__':

    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        MLP, NBEATS, NBEATSx, NHITS, 
        RNN, GRU, LSTM, DilatedRNN, TCN,
        TFT, VanillaTransformer, Informer, Autoformer, PatchTST,
        HINT,
    )
    from neuralforecast.losses.pytorch import GMM, sCRPS

    dataset = 'TourismLarge'

    #------------------------ Read/Parse Data ------------------------#
    hdataset = HierarchicalDataset()
    data = hdataset.load_process_data(dataset=dataset)
    
    # Experiment parameters
    horizon = data['horizon']
    hier_idxs = data['hier_idxs']
    hier_levels = data['hier_levels']
    freq = data['freq']
    Y_df = data['Y_df']
    S_df = data['S_df']
    level = np.arange(0, 100, 2)
    qs = [[50-lv/2, 50+lv/2] for lv in level]
    quantiles = np.sort(np.concatenate(qs)/100)
    n_series = len(S_df)

    # Train/Test split
    Y_test_df  = Y_df.groupby('unique_id').tail(horizon)
    Y_train_df = Y_df.drop(Y_test_df.index)
    Y_test_df  = Y_test_df#.set_index('unique_id')
    Y_train_df = Y_train_df#.set_index('unique_id')

    #----------------------- Fit/Predict HINT ------------------------#
    # HINT := BaseNetwork + Distribution + Reconciliation
    model = NHITS(h=horizon,
                  input_size=24,
                  loss=GMM(n_components=10, quantiles=quantiles),
                  hist_exog_list=['month'],
                  max_steps=2000,
                  early_stop_patience_steps=10,
                  val_check_steps=50,
                  scaler_type='robust',
                  learning_rate=1e-3,
                  valid_loss=sCRPS(quantiles=quantiles))
    model_name = type(model).__name__
    print(f'\n\n model_name {model_name} \n\n')

    hint = HINT(h=horizon, S=S_df.values,
                model=model,  reconciliation='BottomUp')
    
    nf = NeuralForecast(models=[hint], freq=freq)
    nf.fit(df=Y_train_df, val_size=12)


    #------------------------- Evaluate HINT -------------------------#
    # Parse y_test and y_rec
    # Keep only quantile columns from fcsts_df
    evaluation_df = evaluate_forecast(nf=nf, hdataset=hdataset,
                                      Y_test_df=Y_test_df, Y_train_df=Y_train_df,
                                      hier_idxs=hier_idxs, hier_levels=hier_levels,
                                      n_samples=8)
    print(evaluation_df)