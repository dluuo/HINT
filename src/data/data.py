import numpy as np
import pandas as pd

from neuralforecast.utils import augment_calendar_df

from hierarchicalforecast.evaluation import scaled_crps, msse
from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo


class HierarchicalDataset(object):
    # Class with loading, processing and
    # prediction evaluation methods for hierarchical data

    available_datasets = ['Labour','Traffic',
                          'TourismSmall','TourismLarge','Wiki2']

    def _get_hierarchical_scrps(self, hier_idxs, Y, Yq_hat, q_to_pred):
        # We use the indexes obtained from the aggregation tags
        # to compute scaled CRPS across the hierarchy levels 
        scrps_list = []
        for idxs in hier_idxs:
            y      = Y[idxs, :]
            yq_hat = Yq_hat[idxs, :, :]
            scrps  = scaled_crps(y, yq_hat, q_to_pred)
            scrps_list.append(scrps)
        return scrps_list

    def _get_hierarchical_msse(self, hier_idxs, Y, Y_hat, Y_train):
        # We use the indexes obtained from the aggregation tags
        # to compute scaled CRPS across the hierarchy levels         
        msse_list = []
        for idxs in hier_idxs:
            y       = Y[idxs, :]
            y_hat   = Y_hat[idxs, :]
            y_train = Y_train[idxs, :]
            crps    = msse(y, y_hat, y_train)
            msse_list.append(crps)
        return msse_list    

    def _sort_hier_df(self, Y_df, S_df):
        # NeuralForecast core, sorts unique_id lexicographically
        # deviating from S_df, this class matches S_df and Y_hat_df order.
        Y_df.unique_id = Y_df.unique_id.astype('category')
        Y_df.unique_id = Y_df.unique_id.cat.set_categories(S_df.index)
        Y_df = Y_df.sort_values(by=['unique_id', 'ds'])
        return Y_df

    def _nonzero_indexes_by_row(self, M):
        return [np.nonzero(M[row,:])[0] for row in range(len(M))]    

    def load_process_data(self, dataset, directory='./data'):
        # Load data
        assert dataset in self.available_datasets
        data_info = HierarchicalInfo[dataset]
        Y_df, S_df, tags = HierarchicalData.load(directory=directory,
                                                 group=dataset)

        # Parse and augment data
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        Y_df = self._sort_hier_df(Y_df=Y_df, S_df=S_df)
        Y_df, calendar_cols = augment_calendar_df(df=Y_df, freq=data_info.freq)

        # Obtain indexes for plots and evaluation
        hier_levels = ['Overall'] + list(tags.keys())
        hier_idxs = [np.arange(len(S_df))] +\
            [S_df.index.get_indexer(tags[level]) for level in list(tags.keys())]
        hier_linked_idxs = self._nonzero_indexes_by_row(S_df.values.T)

        # Final output
        data = dict(Y_df=Y_df, S_df=S_df, tags=tags,
                    # Hierarchical idxs
                    hier_idxs=hier_idxs,
                    hier_levels=hier_levels,
                    hier_linked_idxs=hier_linked_idxs,
                    # Properties
                    horizon=data_info.seasonality,
                    freq=data_info.freq,
                    seasonality=data_info.seasonality,
                    futr_exog_list=calendar_cols)
        return data