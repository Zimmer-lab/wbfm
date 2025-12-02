import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Dict, Tuple, Union, Optional

import matplotlib
import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.exceptions
from backports.cached_property import cached_property
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, Ridge, ElasticNetCV
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import cross_validate, RepeatedKFold, cross_val_score, GridSearchCV, \
    cross_val_predict, KFold
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from tqdm.auto import tqdm

from wbfm.utils.external.custom_errors import NoBehaviorAnnotationsError
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_pandas import correlate_return_cross_terms, save_valid_ind_1d_or_2d, \
    fill_missing_indices_with_nan
from wbfm.utils.external.utils_sklearn import middle_40_cv_split
from wbfm.utils.external.utils_statsmodels import ols_groupby
from wbfm.utils.external.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.projects.finished_project_data import ProjectData
import statsmodels.api as sm
from wbfm.utils.external.utils_neuron_names import name2int_neuron_and_tracklet
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_project


@dataclass
class NeuronEncodingBase:
    """General class for behavioral encoding or correlations"""
    project_path: Union[str, ProjectData] = None

    dataframes_to_load: List[str] = field(default_factory=lambda: ['ratio'])  # 'red', 'green', 'ratio_filt'])

    is_valid: bool = True
    df_kwargs: dict = field(default_factory=lambda: dict(filter_mode='rolling_mean',
                                                         rename_neurons_using_manual_ids=True,))

    # Needed to interpret the coefficients
    z_score: bool = False

    use_residual_traces: bool = False
    _retained_neuron_names: list = None

    # Alternate method that doesn't use the project data
    dict_of_precalculated_dfs: Dict[str, pd.DataFrame] = None

    # If subsets of the data are desired
    rectification_indices: List[bool] = None

    # For visualization
    use_plotly: bool = False

    @property
    def retained_neuron_names(self):
        if not self._retained_neuron_names:
            _ = self.all_dfs
        return self._retained_neuron_names

    @cached_property
    def project_data(self) -> Optional[ProjectData]:
        if self.project_path is not None:
            return ProjectData.load_final_project_data_from_config(self.project_path)
        else:
            return None

    @property
    def shortened_name(self):
        if self.project_data is not None:
            return self.project_data.shortened_name
        else:
            return 'custom'

    @cached_property
    def all_dfs(self) -> Dict[str, pd.DataFrame]:
        if self.dict_of_precalculated_dfs is not None:
            return self.dict_of_precalculated_dfs

        all_dfs = dict()
        for key in self.dataframes_to_load:
            # Assumes keys are a basic data mode, perhaps with a _filt suffix
            new_opt = dict()
            channel_key = key
            if '_filt' in key:
                channel_key = key.replace('_filt', '')
            if self.use_residual_traces:
                new_opt['interpolate_nan'] = True
                new_opt['residual_mode'] = 'pca'
            new_opt['channel_mode'] = channel_key

            opt = self.df_kwargs.copy()
            opt.update(new_opt)
            df = self.project_data.calc_default_traces(**opt)
            df = fill_nan_in_dataframe(df)
            all_dfs[key] = df

        # Align columns to common subset
        # If I didn't have this block, then I could just use the project data cache directly
        all_column_names = [df.columns for df in all_dfs.values()]
        common_column_names = list(reduce(np.intersect1d, all_column_names))
        all_to_drop = [set(df.columns) - set(common_column_names) for df in all_dfs.values()]
        for key, to_drop in zip(all_dfs.keys(), all_to_drop):
            all_dfs[key].drop(columns=to_drop, inplace=True)

        self._retained_neuron_names = common_column_names

        # print("Finished calculating traces!")
        return all_dfs


@dataclass
class NeuronToUnivariateEncoding(NeuronEncodingBase):
    """Subclass for specifically encoding a 1-d behavioral variable. By default this is speed"""

    cv_factory: callable = KFold
    estimator: callable = Ridge

    # Optional pipeline_alternate method for the target variable, instead of calculating it from a project
    df_of_behaviors: pd.DataFrame = None

    # Results of the fits
    _best_single_neuron: str = None

    _best_multi_neuron_model: callable = None
    _multi_neuron_cv_results: dict = None
    _multi_neuron_model_args: dict = None

    _best_single_neuron_model: callable = None
    _best_leifer_model: callable = None

    def __post_init__(self):
        self.df_kwargs['interpolate_nan'] = True

    def calc_multi_neuron_encoding(self, df_name, y_train=None, only_model_single_state=None, correlation_not_r2=False,
                                   use_null_model=False,
                                   DEBUG=False, **kwargs):
        """
        Calculate the encoding of a single behavioral variable using all neurons

        Uses cross_val_predict and cross_val_score to get the predictions and scores for each cv split

        Parameters
        ----------
        df_name
        y_train: Speed by default
        only_model_single_state
        correlation_not_r2
        DEBUG
        kwargs

        Returns
        -------

        """
        X = self.all_dfs[df_name]
        X, y, y_binary, y_train_name = self.prepare_training_data(X, y_train, only_model_single_state)
        if use_null_model:
            X = pd.DataFrame(y_binary)
        inner_cv = self.cv_factory()
        model = self._setup_inner_cross_validation(inner_cv)

        with warnings.catch_warnings():
            # Outer cross validation: get score
            outer_cv = self.cv_factory()
            if not correlation_not_r2:
                # nested_scores = cross_val_score(model, X=X, y=y, cv=outer_cv)
                cv_results = cross_validate(model, X=X, y=y, cv=outer_cv,
                                            return_train_score=True, return_estimator=True)
                nested_scores = cv_results['test_score']
            else:
                # Loop over all folds and calculate the correlation
                # Note: this is only for Leifer comparison, and should not be used
                nested_scores = []
                cv_results = None
                for train_index, test_index in outer_cv.split(X, y=y_binary):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    nested_scores.append(pearsonr(y_test, y_pred)[0])

            # Also do a prediction step
            try:
                y_pred = cross_val_predict(model, X=X, y=y, cv=outer_cv)
            except ValueError:
                # Fails with TimeSeriesSplit, because the first block is never part of the test set
                y_pred = model.fit(X, y).predict(X)
            y_pred = pd.Series(y_pred, index=y.index)

        self._best_multi_neuron_model = model
        self._multi_neuron_cv_results = cv_results
        self._multi_neuron_model_args = dict(df_name=df_name, y_train=y_train_name,
                                             only_model_single_state=only_model_single_state,
                                             correlation_not_r2=correlation_not_r2, use_null_model=use_null_model)
        if DEBUG:
            # plt.plot(X_test, label='X')
            plt.plot(y, label='y')
            plt.plot(y_pred, label='y hat')
            plt.legend()
            plt.title("Test dataset")
            scores = cross_val_score(model, X, y)
            print(f"Cross validation scores, calculated manually: {scores}, "
                  f"{scores.mean():.2f} +- {scores.std():.2f}")
        return nested_scores, model, y, y_pred, y_train_name

    def _setup_inner_cross_validation(self, inner_cv):
        alphas = np.logspace(-10, 10, 21)  # alpha values to be chosen from by cross-validation
        p_grid = {"alpha": alphas}
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=sklearn.exceptions.ConvergenceWarning)
            estimator = self.estimator()

            # Inner cross validation: get parameter
            model = GridSearchCV(estimator=estimator, param_grid=p_grid, cv=inner_cv)
        return model

    def calc_single_neuron_encoding(self, df_name, y_train=None, only_model_single_state=None, correlation_not_r2=False,
                                    use_null_model=False,
                                    DEBUG=False, **kwargs):
        """
        Best single neuron encoding

        Note that this does nested cross validation to select:
            ridge alpha (inner) and best neuron (outer)

        Parameters
        ----------
        df_name
        y_train
        only_model_single_state
        correlation_not_r2
        use_null_model
        DEBUG
        kwargs

        Returns
        -------

        """
        X = self.all_dfs[df_name]
        X, y, y_binary, y_train_name = self.prepare_training_data(X, y_train, only_model_single_state)
        if use_null_model:
            raise NotImplementedError("Null model not implemented for single neuron encoding")
        inner_cv = self.cv_factory() #.split(X, y_binary)
        model = self._setup_inner_cross_validation(inner_cv)

        with warnings.catch_warnings():

            # Outer cross validation: get best neuron
            # Note that this takes a while because it has to redo the inner cross validation for each feature
            # It can be parallelized but has a pickle error on my machine
            sfs_cv = self.cv_factory()
            sfs = SequentialFeatureSelector(estimator=model,
                                            n_features_to_select=1, direction='forward', cv=sfs_cv)
            sfs.fit(X, y)

            feature_names = get_names_from_df(X)
            best_neuron = [feature_names[s] for s in sfs.get_support(indices=True)]
            X_best_single_neuron = X[best_neuron].values.reshape(-1, 1)

            # Calculate the error using this neuron (CV again)
            outer_cv = self.cv_factory()
            if not correlation_not_r2:
                nested_scores = cross_val_score(model, X=X_best_single_neuron, y=y, cv=outer_cv)
            else:
                # Loop over all folds and calculate the correlation
                nested_scores = []
                for train_index, test_index in outer_cv.split(X_best_single_neuron, y):
                    X_train, X_test = X_best_single_neuron[train_index], X_best_single_neuron[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    corr = pearsonr(y_pred, y_test)[0]
                    nested_scores.append(corr)
            # model = RidgeCV(cv=self.cv, alphas=alphas).fit(X_train, y_train)

            # Also do a prediction step
            try:
                y_pred = cross_val_predict(model, X=X_best_single_neuron, y=y, cv=outer_cv)
            except ValueError:
                # Fails with TimeSeriesSplit, because the first block is never part of the test set
                y_pred = model.fit(X_best_single_neuron, y).predict(X_best_single_neuron)
            y_pred = pd.Series(y_pred, index=y.index)

        self._best_single_neuron_model = model
        self._best_single_neuron = best_neuron
        if DEBUG:
            plt.plot(X_best_single_neuron, label='X')
            plt.plot(y, label='y')
            plt.plot(y_pred, label='y hat')
            plt.legend()
            plt.title("Test dataset")
            # scores = cross_val_score(estimator, X_train_best_single_neuron, y_train)
            print(f"Cross validation scores, calculated outside SequentialFeatureSelector: {nested_scores}, "
                  f"{nested_scores.mean():.2f} +- {nested_scores.std():.2f}")
        return nested_scores, model, y, y_pred, y_train_name, best_neuron

    def best_single_neuron(self):
        """Returns the name of the best single neuron, if it was calculated"""
        if self._best_single_neuron is None:
            logging.warning("Best single neuron was not calculated, using default settings")
            self.calc_single_neuron_encoding(df_name='ratio')
        return self._best_single_neuron

    def prepare_training_data(self, X, y_train_name, only_model_single_state=None,
                              binary_state = BehaviorCodes.REV) -> \
            Tuple[pd.DataFrame, pd.Series, pd.Series, str]:
        """
        Converts a string describing a behavioral time series into the appropriate series, and aligns with the neural
        data (which may have fewer frames)

        Parameters
        ----------
        X
        y_train_name
        only_model_single_state: See BehaviorCodes

        Returns
        -------

        """
        trace_len = X.shape[0]
        y, y_train_name = self.unpack_behavioral_time_series_from_name(y_train_name, trace_len)

        # If used, subset the data using a rectification variable
        # Note that this is similar to only_model_single_state, but can be used if there is no project saved
        #  (or if this class refers to multiple projects)
        if self.rectification_indices is not None:
            assert only_model_single_state is None, "Can't use both rectification and only_model_single_state"
            X = X.loc[self.rectification_indices, :]
            y = y.loc[self.rectification_indices]

        # Remove nan points, if any (the regression can't handle them)
        valid_ind = np.where(~np.isnan(y))[0]
        X = save_valid_ind_1d_or_2d(X.copy(), valid_ind)
        y = save_valid_ind_1d_or_2d(y.copy(), valid_ind)

        # Also build a binary class variable; possibly used for cross validation or as a null model
        if self.project_data is not None:
            worm = self.project_data.worm_posture_class
            try:
                y_binary = (worm.beh_annotation(fluorescence_fps=True) == binary_state).copy()
                y_binary.index = y.index
            except NoBehaviorAnnotationsError:
                y_binary = y.copy()
        else:
            y_binary = y.copy()

        # Optionally subset the data to be only a specific state
        if only_model_single_state is not None:
            try:
                BehaviorCodes.assert_is_valid(only_model_single_state)
                beh = worm.beh_annotation(fluorescence_fps=True).reset_index(drop=True)
                ind = beh == only_model_single_state
                X = X.loc[ind, :]
                y = y.loc[ind]
                y_binary = y_binary.loc[ind]
            except NoBehaviorAnnotationsError:
                logging.warning("No behavior annotations found, can't subset data")
                pass

        # z-score the data
        if self.z_score:
            X = (X - X.mean()) / X.std()
            y = (y - y.mean()) / y.std()

        return X, y, y_binary, y_train_name

    def unpack_behavioral_time_series_from_name(self, y_train_name, trace_len):
        """
        See calc_behavior_from_alias for valid aliases

        Parameters
        ----------
        y_train_name
        trace_len

        Returns
        -------

        """
        if self.df_of_behaviors is not None and y_train_name in self.df_of_behaviors:
            y = self.df_of_behaviors[y_train_name].copy()
        else:
            if y_train_name is None:
                y_train_name = 'signed_stage_speed'
            # Get 1d series from behavior
            y = self.project_data.worm_posture_class.calc_behavior_from_alias(y_train_name)
        y = y.iloc[:trace_len]
        y = fill_nan_in_dataframe(y)
        y.reset_index(drop=True, inplace=True)
        return y, y_train_name

    def plot_model_prediction(self, df_name, y_train=None, use_multineuron=True, use_leifer_method=False,
                              only_model_single_state=None, correlation_not_r2=False, use_null_model=False,
                              DEBUG=False, **plot_kwargs):
        """Plots model prediction over raw data"""
        opt = dict(y_train=y_train, only_model_single_state=only_model_single_state,
                   correlation_not_r2=correlation_not_r2, use_null_model=use_null_model,
                   DEBUG=DEBUG)
        if use_leifer_method:
            score_list, model, y_total, y_pred, y_train_name = \
                self.calc_leifer_encoding(df_name, **opt)
            y_name = f"leifer_{y_train_name}"
            best_neuron = ""
        elif use_multineuron:
            score_list, model, y_total, y_pred, y_train_name = \
                self.calc_multi_neuron_encoding(df_name, **opt)
            y_name = f"multineuron_{y_train_name}"
            best_neuron = ""
        else:
            score_list, model, y_total, y_pred, y_train_name, best_neuron = \
                self.calc_single_neuron_encoding(df_name, **opt)
            y_name = f"single_best_neuron_{y_train_name}"

        fig = self._plot_predictions(df_name, y_pred, y_total, y_name=y_name, score_list=score_list, best_neuron=best_neuron,
                                     **plot_kwargs)

        return fig, model, best_neuron

    def plot_x(self, df_name, y_train=None):
        X = self.all_dfs[df_name]
        X, y, y_binary, y_train_name = self.prepare_training_data(X, y_train)

        if self.use_plotly:
            fig = px.line(X, title=f"Training data for {df_name}")
            fig.show()
        else:
            fig = plt.figure()
            plt.plot(X)
            plt.title(f"Training data for {df_name}")
            plt.show()
        return fig

    def plot_y(self, df_name, y_train=None):
        X = self.all_dfs[df_name]
        X, y, y_binary, y_train_name = self.prepare_training_data(X, y_train)

        if self.use_plotly:
            fig = px.line(y, title=f"Target data for {y_train_name}")
            fig.show()
        else:
            fig = plt.figure()
            plt.plot(y)
            plt.title(f"Target for {y_train_name}")
            plt.show()
        return fig

    def calc_leifer_encoding(self, df_name, y_train=None, use_multineuron=True, only_model_single_state=None,
                             correlation_not_r2=False, use_null_model=False,
                             DEBUG=False, **kwargs):
        """
        Fits model using the Leifer settings, which does not use full cross validation

        Rather, it uses a single train/test split, and then calculates the score on the test set
        The test set is the middle 40% of the data

        Note that this produces significantly more optimistic scores than the "best practice" nested cross validation
        method, used in the other methods in this class

        Note also that the correlation_not_r2 score is what was actually calculated in the paper, and again produces
        more optimistic scores than the r2 score or the other cross validation methods

        Parameters
        ----------
        df_name
        y_train

        Returns
        -------

        """
        X = self.all_dfs[df_name]
        X, y, y_binary, y_train_name = self.prepare_training_data(X, y_train,
                                                                  only_model_single_state=only_model_single_state)
        if use_null_model:
            X = pd.DataFrame(y_binary)

        # Get train-test split
        trace_len = X.shape[0]
        ind_test, ind_train = middle_40_cv_split(trace_len)
        X = X - X.mean()
        X = X / X.std()
        X_train = X.iloc[ind_train, :]
        X_test = X.iloc[ind_test, :]
        y_train = y.iloc[ind_train]
        y_test = y.iloc[ind_test]

        # Fit; note that even though we have CV, the result is sensitive to the exact value space
        # alphas = np.logspace(-6, 6, 21)  # alpha values to be chosen from by cross-validation
        # l1_ratio = np.logspace(-7, 0, 13)
        alphas = np.logspace(-6, 2, 11)  # alpha values to be chosen from by cross-validation
        l1_ratio = np.logspace(-6, 0, 7)
        if self._best_leifer_model is None:
            model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio)
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=sklearn.exceptions.ConvergenceWarning)
                model.fit(X=X_train, y=y_train)
        else:
            model = self._best_leifer_model
        if not correlation_not_r2:
            score = model.score(X_test, y_test)
        else:
            y_pred = model.predict(X_test)
            score = pearsonr(y_test, y_pred)[0]
        y_pred = model.predict(X)
        y_pred = pd.Series(y_pred, index=y.index)

        self._best_leifer_model = model

        return [score], model, y, y_pred, y_train_name

    def plot_sorted_correlations(self, df_name, y_train=None, to_save=False, saving_folder=None):
        """
        Does not fit a model, just raw correlation
        """
        X = self.all_dfs[df_name]
        # Note: just use this function to resolve the name; do not actually use the train-test split
        _, y_total, y_binary, y_train_name = self.prepare_training_data(X, y_train)

        corr = X.corrwith(y_total)
        idx = np.argsort(corr)
        names = get_names_from_df(X)

        fig, ax = plt.subplots(dpi=200)
        x = range(len(idx))
        plt.bar(x, corr.iloc[idx.values])

        labels = np.array(names)[idx.values]
        labels = [name2int_neuron_and_tracklet(n) for n in labels]
        # plt.xticks(x, labels="")
        # ymin = np.min(corr) - 0.1
        # for i, name in enumerate(labels):
        #     plt.annotate(text=name, xy=(i, ymin), xytext=(i, ymin-0.1*(-i % 8)-0.1), xycoords='data', arrowprops={'width':1, 'headwidth':0}, annotation_clip=False)
        # ax.xaxis.set_major_locator(MultipleLocator(10))
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        plt.xticks(ticks=x, labels=labels, fontsize=6)
        # ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
        plt.grid(which='major', axis='x')
        ax.set_axisbelow(True)
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            tick.set_pad(8 * (i % 4))
        plt.title(f"Sorted correlation: {df_name} traces with {y_train_name}")

        if to_save:
            fname = f"sorted_correlation_{df_name}_{y_train_name}.png"
            self._savefig(fname, saving_folder)

    def calc_dataset_summary_df(self, df_name: str, **kwargs) -> pd.DataFrame:
        """
        Calculates a summary number for the full dataset:
            The linear model error for a) the best single neuron and b) the multivariate encoding
            Also calculates the leifer error

        Parameters
        ----------
        name

        Returns
        -------

        """

        kwargs['df_name'] = df_name
        multi_list = self.calc_multi_neuron_encoding(**kwargs)[0]
        try:
            single_list = self.calc_single_neuron_encoding(**kwargs)[0]
        except NotImplementedError:
            single_list = None
        leifer_score = self.calc_leifer_encoding(**kwargs)[0]

        df_dict = {'multi_neuron': np.nanmean(multi_list),
                   'leifer_score': leifer_score,
                   'dataset_name': self.shortened_name}
        if single_list is not None:
            df_dict['single_neuron'] = np.nanmean(single_list)
        df = pd.DataFrame(df_dict, index=[0])
        return df

    def calc_prediction_or_raw_df(self, df_name, y_train=None, use_multineuron=True, only_model_single_state=None,
                                  prediction_not_raw=True, use_leifer_method=False, **kwargs) -> pd.DataFrame:
        """
        Similar to plot_model_prediction, but returns one dataframe (prediction or raw)

        Parameters
        ----------
        df_name
        y_train
        use_multineuron
        only_model_single_state

        Returns
        -------

        """
        opt = dict(y_train=y_train, only_model_single_state=only_model_single_state)
        opt.update(kwargs)
        if prediction_not_raw:
            if use_leifer_method:
                score_list, model, y_total, y_pred, y_train_name = \
                    self.calc_leifer_encoding(df_name, **opt)
            elif use_multineuron:
                score_list, model, y_total, y_pred, y_train_name = \
                    self.calc_multi_neuron_encoding(df_name, **opt)
            else:
                score_list, model, y_total, y_pred, y_train_name, best_neuron = \
                    self.calc_single_neuron_encoding(df_name, **opt)
            y = y_pred

        else:
            X = self.all_dfs[df_name]
            X, y, y_binary, y_train_name = self.prepare_training_data(X, y_train, only_model_single_state)

        y = y.sort_index()
        y, _ = fill_missing_indices_with_nan(y, expected_max_t=self.project_data.num_frames)
        y = y.sort_index()
        df_summary = pd.DataFrame({self.shortened_name: y})

        return df_summary

    def calc_dataset_per_neuron_summary_df(self, df_name, x_name):
        """
        Like calc_dataset_summary_df, but summarizes activity per neuron (index), not the entire dataset in one number

        Currently just computes the correlations, both overall and rectified. Thus the output is n x 7, with columns:
            ['correlation', 'rev_correlation', 'fwd_correlation',
                     'correlation_std', 'rev_correlation_std', 'fwd_correlation_std',
                     'neuron_name']
        Parameters
        ----------
        df_name - e.g. 'ratio'
        x_name - e.g. 'signed_speed'

        Returns
        -------

        """

        col_names = ['coefficient', 'rev_coefficient', 'fwd_coefficient',
                     'coefficient_std', 'rev_coefficient_std', 'fwd_coefficient_std',
                     'neuron_name']
        df_dict = {n: list() for n in col_names}
        for neuron_name in tqdm(self.retained_neuron_names, leave=False):
            df, x_train_name = self.build_df_for_correlation(df_name, neuron_name, x_name)

            df_dict['neuron_name'].append(neuron_name)
            # Full coefficient, no subsetting
            res = ols_groupby(df, x=x_train_name, y=neuron_name)[0]
            df_dict['coefficient'].append(res.params[x_name])
            df_dict['coefficient_std'].append(res.bse[x_name])
            # Rectified coefficient
            res_fwd, res_rev = ols_groupby(df, x=x_train_name, y=neuron_name, hue='reversal')
            df_dict['fwd_coefficient'].append(res_fwd.params[x_name])
            df_dict['fwd_coefficient_std'].append(res_fwd.bse[x_name])
            df_dict['rev_coefficient'].append(res_rev.params[x_name])
            df_dict['rev_coefficient_std'].append(res_rev.bse[x_name])

        df = pd.DataFrame(df_dict)
        return df

    def plot_multineuron_weights(self, saving_folder=None, to_show=True):
        """
        Uses saved model, and plots the weights

        Parameters
        ----------
        df_name
        y_name
        saving_folder
        kwargs

        Returns
        -------

        """
        if not self.z_score:
            logging.warning("Not z-scored, so weights are not interpretable")
        model_args = self._multi_neuron_model_args
        if model_args is None:
            raise ValueError("No model has been trained yet")
        df_name = model_args['df_name']
        y_train = model_args['y_train']
        feature_names = list(self.all_dfs[df_name].columns)

        # Uses precalculated results of cross validation
        coefs, fig = boxplot_from_cross_validation_dict(self._multi_neuron_cv_results,
                                                        feature_names=feature_names,
                                                        name=self.shortened_name)

        if saving_folder is not None:
            fname = f"regression_weights_{df_name}_{y_train}.png"
            self._savefig(fig, fname, saving_folder)

        if to_show:
            plt.show()

    def plot_permutation_feature_importance(self, df_name, y_train, saving_folder=None, to_show=True, **kwargs):
        """
        Does pfi on the saved _multi_neuron_cv_results, which assumes that the model has just been trained by the same
        arguments as passed to this function

        Parameters
        ----------
        df_name
        y_train

        Returns
        -------

        """
        if not self.z_score:
            logging.warning("Not z-scored, so pfi is not interpretable")
        X = self.all_dfs[df_name]
        X, y, y_binary, y_train_name = self.prepare_training_data(X, y_train)

        all_pfi = []
        for idx, estimator in tqdm(enumerate(self._multi_neuron_cv_results['estimator']), leave=False):
            # Use train and test data again
            pfi = permutation_importance(estimator.best_estimator_, X, y)
            all_pfi.append(pfi['importances'])

        all_pfi = np.hstack(all_pfi)
        df_pfi = pd.DataFrame(all_pfi.T, columns=X.columns)
        # Sort by median value
        df_pfi = df_pfi.reindex(df_pfi.median().sort_values(ascending=False).index, axis=1)
        fig = px.box(df_pfi, title=f"Feature importance for predicting: {y_train}")
        if to_show:
            fig.show()

        if saving_folder is not None:
            fname = f"pfi_{df_name}_{y_train}.png"
            self._savefig(fig, fname, saving_folder)

    def _plot_predictions(self, df_name, y_pred, y_train, y_name="", score_list: list = None, best_neuron="",
                          saving_folder=None, to_show=True):
        """
        Plots predictions and training data

        Assumes both y_pred and y_train are the length of the entire dataset (not a train-test split)

        Parameters
        ----------
        df_name
        y_pred
        y_train
        y_name
        score_list
        saving_folder

        Returns
        -------

        """
        if score_list is None:
            score_mean = median_absolute_error(y_train, y_pred)
            score_std = score_mean
        else:
            score_mean = np.mean(score_list)
            score_std = np.std(score_list)
        # Metadata for plots
        title_str = f"R2={score_mean:.2f}+-{score_std:.2f} ({df_name}; {self.shortened_name})"
        if best_neuron != "":
            title_str = f"{best_neuron}: {title_str}"
        # Actually plot
        if not self.use_plotly:
            fig, ax = plt.subplots(dpi=200)
            opt = dict()
            if df_name == 'green' or df_name == 'red':
                opt['color'] = df_name
            ax.plot(y_pred, label='prediction', **opt)
            ax.set_title(title_str)
            plt.ylabel(f"{y_name}")
            plt.xlabel("Time (volumes)")
            ax.plot(y_train, color='black', label='Target', alpha=0.8)
            plt.legend()
            if self.project_data is not None:
                self.project_data.shade_axis_using_behavior()
        else:
            # Make a dataframe for plotly
            df = pd.DataFrame({'Prediction': y_pred, 'Target': y_train})
            fig = px.line(df, title=title_str, labels={'index': 'Time (volumes)', 'value': f"{y_name}"})
            if to_show:
                fig.show()

        if saving_folder is not None:
            fname = f"regression_fit_{df_name}_{y_name}.png"
            self._savefig(fig, fname, saving_folder)

        return fig

    def plot_single_neuron_scatter(self, df_name, neuron_name, x_name,
                                   do_rectified=True):
        """
        Plots a scatter plot of behavior and neural activity

        Parameters
        ----------
        df_name
        neuron_name
        x_name - Name of behavior to be on the x axis. See unpack_behavioral_time_series_from_name
        do_rectified

        Returns
        -------

        """
        df, x_train_name = self.build_df_for_correlation(df_name, neuron_name, x_name)

        plot_opt = dict(data=df, x=x_train_name, y=neuron_name)
        reg_opt = dict(scatter=False, robust=True, n_boot=10)
        if do_rectified:
            plot_opt['hue'] = 'reversal'
            sns.lmplot(**plot_opt, **reg_opt)
        else:
            sns.regplot(**plot_opt, **reg_opt)

        sns.kdeplot(**plot_opt)
        plt.ylim(0, 1)

    def build_df_for_correlation(self, df_name, neuron_name, x_name):
        df_traces = self.all_dfs[df_name]
        y = df_traces[neuron_name]
        y, x, binary_state, x_train_name = self.prepare_training_data(y, x_name)
        df = pd.DataFrame({x_name: x, neuron_name: y, 'reversal': binary_state})
        return df, x_train_name

    def _savefig(self, fig, fname, saving_folder, also_save_svg=True):
        fname = os.path.join(saving_folder, f"{self.shortened_name}-{fname}")
        print(f"Saving figure to {fname}")

        if type(fig) == matplotlib.figure.Figure:
            if saving_folder is None:
                vis_cfg = self.project_data.project_config.get_visualization_config(make_subfolder=True)
                fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            fig.savefig(fname)
            if also_save_svg:
                fig.savefig(fname.replace(".png", ".svg"))
        else:
            # Assume plotly
            fig.write_image(fname)
            if also_save_svg:
                fig.write_image(fname.replace(".png", ".svg"))

    def _plot_linear_regression_coefficients(self, X, y, df_name, model=None,
                                             only_plot_nonzero=True, also_plot_traces=True, y_name="speed"):
        # From https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py
        if model is None:
            alphas = np.logspace(-10, 10, 21)  # alpha values to be chosen from by cross-validation
            model = LassoCV(alphas=alphas, max_iter=1000)

        feature_names = get_names_from_df(self.all_dfs[df_name])
        initial_val = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
        cv_model = cross_validate(
            model,
            X,
            y,
            cv=cv,
            return_estimator=True,
            n_jobs=2,
        )
        name = self.shortened_name
        coefs = boxplot_from_cross_validation_dict(cv_model, feature_names, only_plot_nonzero, name)

        if self.project_data is not None:
            fname = f"lasso_coefficients_{df_name}_{y_name}.png"
            self.project_data.save_fig_in_project(fname)

        # gridplot of traces
        if also_plot_traces and self.project_data is not None:
            direct_shading_dict = coefs.mean().to_dict()
            make_grid_plot_from_project(self.project_data, 'ratio', 'integration',
                                        neuron_names_to_plot=get_names_from_df(coefs),
                                        direct_shading_dict=direct_shading_dict,
                                        sort_using_shade_value=True, savename_suffix=f"{y_name}_encoding")

        os.environ["PYTHONWARNINGS"] = initial_val  # Also affect subprocesses

        return cv_model, coefs


def boxplot_from_cross_validation_dict(cv_model, feature_names=None, only_plot_nonzero=False, name=None):
    def get_coef(estimator_like):
        try:
            return estimator_like.coef_
        except AttributeError:
            # For GridSearchCV objects
            return estimator_like.best_estimator_.coef_

    if feature_names is not None:
        coefs = pd.DataFrame(
            [get_coef(est) for est in cv_model["estimator"]], columns=feature_names
        )
    else:
        coefs = pd.DataFrame([get_coef(est) for est in cv_model["estimator"]])

    # Only keep neurons with nonzero values
    tol = 1e-3
    if only_plot_nonzero:
        coefs = coefs.loc[:, coefs.mean().abs() > tol]
    # Boxplot of variability
    fig = plt.figure(dpi=100)
    sns.stripplot(data=coefs, orient="h", color="k", alpha=0.5, linewidth=1)
    sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=100)
    plt.axvline(x=0, color=".5")
    title_str = f"Coefficient variability for {name}"
    plt.title(title_str)
    plt.subplots_adjust(left=0.3)
    plt.grid(axis='y', which='both')
    return coefs, fig


@dataclass
class NeuronToMultivariateEncoding(NeuronEncodingBase):
    """
    Designed for single-neuron correlations to all kymograph body segments

    Can also use other kymograph-like dataframes, such as hilbert amplitude or instantaneous frequency
    """

    posture_attribute: str = "curvature"  # Must be a function of WormFullVideoPosture
    posture_index_start: int = 2
    posture_index_end: int = 30

    allow_negative_correlations: bool = True

    def __post_init__(self):
        if self.project_data.worm_posture_class.has_full_kymograph and self.project_data.check_traces():
            self.is_valid = True
        else:
            logging.warning("Kymograph not found, this class will not work")
            self.is_valid = False

    def get_kymo_like_df(self):
        opt = dict(fluorescence_fps=True)
        kymo = getattr(self.project_data.worm_posture_class, self.posture_attribute)(**opt).reset_index(drop=True)
        df_kymo = kymo.loc[:, self.posture_index_start:self.posture_index_end].copy()
        return df_kymo

    @cached_property
    def all_dfs_corr(self) -> Dict[str, pd.DataFrame]:
        df_kymo = self.get_kymo_like_df()

        all_dfs = self.all_dfs
        all_dfs_corr = {key: correlate_return_cross_terms(df, df_kymo) for key, df in all_dfs.items()}
        return all_dfs_corr

    @cached_property
    def all_dfs_corr_fwd(self) -> Dict[str, pd.DataFrame]:
        assert self.project_data.worm_posture_class.has_beh_annotation, "Behavior annotations required"
        kymo = self.get_kymo_like_df()

        # New: only do certain indices
        _beh_ind = self.project_data.worm_posture_class.beh_annotation(fluorescence_fps=True, reset_index=True)
        ind = BehaviorCodes.vector_equality(_beh_ind, BehaviorCodes.FWD)
        all_dfs = self.all_dfs
        df_kymo = kymo.loc[ind, :].copy()
        all_dfs_corr = {key: correlate_return_cross_terms(df.loc[ind, :], df_kymo) for key, df in all_dfs.items()}
        return all_dfs_corr

    @cached_property
    def all_dfs_corr_rev(self) -> Dict[str, pd.DataFrame]:
        assert self.project_data.worm_posture_class.has_beh_annotation, "Behavior annotations required"
        kymo = self.get_kymo_like_df()

        # New: only do certain indices
        _beh_ind = self.project_data.worm_posture_class.beh_annotation(fluorescence_fps=True, reset_index=True)
        ind = BehaviorCodes.vector_equality(_beh_ind, BehaviorCodes.REV)
        all_dfs = self.all_dfs

        df_kymo = kymo.loc[ind, :].copy()
        all_dfs_corr = {key: correlate_return_cross_terms(df.loc[ind, :], df_kymo) for key, df in all_dfs.items()}
        return all_dfs_corr

    @property
    def all_labels(self):
        return list(self.all_dfs.keys())

    @property
    def all_colors(self):
        cols = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
        return cols[:len(self.all_labels)]

    def calc_per_neuron_df(self, name: str, rectification_variable=None) -> pd.DataFrame:
        """
        Calculates a summary dataframe of information per neuron.
            Rows: neuron names
            Columns: ['median_brightness', 'var_brightness', 'body_segment_argmax', 'corr_max', 'dataset_name']

            Note that dataset_name is used when this is concatenated with other dataframes

        Parameters
        ----------
        name - str, one of self.all_labels
        rectification_variable - optional str, one of None, 'rev', 'fwd'

        Returns
        -------

        """
        if rectification_variable is None:
            df_corr = self.all_dfs_corr[name].copy()
        elif rectification_variable == 'rev':
            df_corr = self.all_dfs_corr_rev[name].copy()
        elif rectification_variable == 'fwd':
            df_corr = self.all_dfs_corr_fwd[name].copy()
        else:
            raise ValueError(f"rectification_variable must be one of None, 'rev', 'fwd', not {rectification_variable}")
        if self.allow_negative_correlations:
            df_corr = df_corr.abs()
        df_traces = self.all_dfs[name]

        body_segment_argmax = df_corr.columns[df_corr.apply(pd.Series.argmax, axis=1)]
        body_segment_argmax = pd.Series(body_segment_argmax, index=df_corr.index)

        corr_max = df_corr.max(axis=1)
        median = df_traces.median(axis=0)
        var = df_traces.var(axis=0)

        df_all = pd.concat([median, var, body_segment_argmax, corr_max], axis=1)
        df_all.columns = ['median_brightness', 'var_brightness', 'body_segment_argmax', 'corr_max']

        # Add column with name of dataset
        df_all['dataset_name'] = self.shortened_name
        df_all.dataset_name = df_all.dataset_name.astype('category')

        return df_all

    def calc_wide_pairwise_summary_df(self, start_name, final_name, to_add_columns=True):
        """
        Calculates basic parameters for single data types, as well as phase shifts

        Returns a widened dataframe, with new columns for each variable

        Example usage (with seaborn):
            df = plotter_gcamp.calc_wide_pairwise_summary_df('red', 'green')
            sns.pairplot(df)

        Parameters
        ----------
        start_name
        final_name

        Returns
        -------

        """
        # Get data for both individually
        df_start = self.calc_per_neuron_df(start_name)
        df_final = self.calc_per_neuron_df(final_name)
        df = df_start.join(df_final, lsuffix=f"_{start_name}", rsuffix=f"_{final_name}")

        # Build additional numeric columns
        if to_add_columns:
            to_subtract = 'body_segment_argmax'
            df['phase_difference'] = df[f"{to_subtract}_{final_name}"] - df[f"{to_subtract}_{start_name}"]

        return df

    def calc_long_pairwise_summary_df(self, start_name, final_name):
        """
        Calculates basic parameters for single data types

        Returns a long dataframe, with new columns for the original datatype ('source_data')
        Is also reindexed, with a new column referring to neuron names (these are duplicated)

        Example usage:
            df = plotter_gcamp.calc_long_pairwise_summary_df('red', 'green')
            sns.pairplot(df, hue='source_data', palette={'red': 'pink', 'green': 'green'})

        Parameters
        ----------
        start_name
        final_name

        Returns
        -------

        """
        # Get data for both individually
        df_start = self.calc_per_neuron_df(start_name)
        df_final = self.calc_per_neuron_df(final_name)

        # Build columns and join
        df_start['source_data'] = start_name
        df_final['source_data'] = final_name

        df = pd.concat([df_start, df_final], axis=0)
        df.source_data = df.source_data.astype('category')
        df = df.reset_index().rename(columns={'index': 'neuron_name'})

        return df

    def plot_correlation_of_examples(self, to_save=True, only_within_state=None, **kwargs):
        # Calculate correlation dataframes
        if only_within_state is None:
            all_dfs = list(self.all_dfs_corr.values())
        elif only_within_state.lower() == 'fwd':
            all_dfs = list(self.all_dfs_corr_fwd.values())
        elif only_within_state.lower() == 'rev':
            all_dfs = list(self.all_dfs_corr_rev.values())
        else:
            raise NotImplementedError

        self._multi_plot(list(self.all_dfs.values()), all_dfs,
                         self.all_labels, self.all_colors,
                         project_data=self.project_data, to_save=to_save, **kwargs)

    def plot_correlation_of_prefentially_one_state(self, to_save=True, only_within_state=None, **kwargs):
        # Calculate correlation dataframes
        all_dfs_fwd = list(self.all_dfs_corr_fwd.values())
        all_dfs_rev = list(self.all_dfs_corr_rev.values())
        if only_within_state.lower() == 'fwd':
            all_dfs = [f - r for f, r in zip(all_dfs_fwd, all_dfs_rev)]
        elif only_within_state.lower() == 'rev':
            all_dfs = [r - f for f, r in zip(all_dfs_fwd, all_dfs_rev)]
        else:
            raise NotImplementedError

        all_figs = self._multi_plot(list(self.all_dfs.values()), all_dfs,
                                    self.all_labels, self.all_colors,
                                    project_data=self.project_data, to_save=to_save, **kwargs)
        # for fig in all_figs:
        #     fig.axes[0][0].set_ylabel("Differential correlation")

    def plot_correlation_histograms(self, to_save=False):
        plt.figure(dpi=100)
        all_max_corrs = [df_corr.abs().max(axis=1) for df_corr in self.all_dfs_corr.values()]

        plt.hist(all_max_corrs,
                 color=self.all_colors,
                 label=self.all_labels)
        plt.xlim(-0.2, 1)
        plt.title(self.project_data.shortened_name)
        plt.xlabel("Maximum correlation")
        plt.legend()

        if to_save:
            fname = f'maximum_correlation_kymograph_histogram.png'
            self.project_data.save_fig_in_project(fname)

    def plot_histogram_difference_after_ratio(self, df_start_names=None, df_final_name='ratio', to_save=True):
        plt.figure(dpi=100)

        if df_start_names is None:
            df_start_names = ['red', 'green']
        # Get data
        all_df_starts = [self.all_dfs_corr[name] for name in df_start_names]
        df_final = self.all_dfs_corr[df_final_name]

        # Get differences
        df_final_maxes = df_final.max(axis=1)
        all_diffs = [df_final_maxes - df.max(axis=1) for df in all_df_starts]

        # Plot
        plt.hist(all_diffs)

        plt.xlabel("Maximum correlation difference")
        title_str = f"Correlation difference between {df_start_names} to {df_final_name}"
        plt.title(title_str)

        if to_save:
            fname = f'{title_str}.png'
            self.project_data.save_fig_in_project(suffix=fname)

    def plot_paired_boxplot_difference_after_ratio(self, df_start_name='red', df_final_name='ratio', to_save=True):
        plt.figure(dpi=100)
        # Get data
        both_maxes = self.get_data_for_paired_boxplot(df_final_name, df_start_name)

        # Plot
        paired_boxplot_from_dataframes(both_maxes, [df_start_name, df_final_name])

        plt.ylim(0, 0.8)
        plt.ylabel("Absolute correlation")
        title_str = f"Change in correlation from {df_start_name} to {df_final_name}"
        plt.title(title_str)

        if to_save:
            fname = f'{title_str}.png'
            self.project_data.save_fig_in_project(suffix=fname)

    def get_data_for_paired_boxplot(self, df_final_name, df_start_name):
        df_start = self.all_dfs_corr[df_start_name]
        df_final = self.all_dfs_corr[df_final_name]
        start_maxes = df_start.max(axis=1)
        final_max = df_final.max(axis=1)
        both_maxes = pd.concat([start_maxes, final_max], axis=1).T
        return both_maxes

    def plot_phase_difference(self, df_start_name='red', df_final_name='green', corr_thresh=0.2, remove_zeros=True,
                              to_save=True):
        """
        Green minus red

        Returns
        -------

        """
        plt.figure(dpi=100)
        df_start = self.all_dfs_corr[df_start_name].copy()
        df_final = self.all_dfs_corr[df_final_name].copy()

        ind_to_keep = df_start.abs().max(axis=1) > corr_thresh
        df_start = df_start.loc[ind_to_keep, :]
        df_final = df_final.loc[ind_to_keep, :]

        start_body_segment_argmax = df_start.columns[df_start.abs().apply(pd.Series.argmax, axis=1)]
        final_body_segment_argmax = df_final.columns[df_final.abs().apply(pd.Series.argmax, axis=1)]

        diff = final_body_segment_argmax - start_body_segment_argmax
        title_str = f"{df_final_name} - {df_start_name} with starting corr > {corr_thresh}"
        if remove_zeros:
            diff = diff[diff != 0]
            title_str = f"{title_str} (zeros removed)"
        plt.hist(diff, bins=np.arange(diff.min(), diff.max()))
        plt.title(title_str)
        plt.xlabel("Phase shift (body segments)")

        if to_save:
            fname = f'{title_str.replace(">", "ge")}.png'
            self.project_data.save_fig_in_project(suffix=fname)

    @staticmethod
    def _multi_plot(all_dfs_list, all_dfs_corr_list, all_labels, all_colors, ax_locations=None,
                    project_data: ProjectData=None,
                    corr_thresh=0.3, which_df_to_apply_corr_thresh=-1, max_num_plots=None,
                    xlim=None, to_save=False, all_names=None):
        all_figs = []
        if xlim is None:
            xlim = [100, 450]
        if ax_locations is None:
            ax_locations = [1, 1, 3, 3, 3]

        if all_names is None:
            all_names = list(all_dfs_corr_list[0].index)
        else:
            # Plot all that are sent
            corr_thresh = None
        num_open_plots = 0

        for i in range(all_names):
            abs_corr = all_dfs_corr_list[which_df_to_apply_corr_thresh].iloc[i, :]
            if corr_thresh is not None and abs_corr.max() < corr_thresh:
                continue
            else:
                num_open_plots += 1

            fig, axes = plt.subplots(ncols=2, nrows=2, dpi=100, figsize=(15, 5))
            all_figs.append(fig)
            axes = np.ravel(axes)
            neuron_name = all_names[i]

            for df, df_corr, lab, col, ax_loc in zip(all_dfs_list, all_dfs_corr_list, all_labels, all_colors, ax_locations):

                plt_opt = dict(label=lab, color=col)
                # Always put the correlation on ax 0
                abs_corr = df_corr.iloc[i, :]
                axes[0].plot(abs_corr, **plt_opt)

                # Put the trace on the passed axis
                trace = df[neuron_name]
                axes[ax_loc].plot(trace / trace.mean(), **plt_opt)

            axes[0].set_xlabel("Body segment")
            axes[0].set_ylabel("Correlation")
            axes[0].set_ylim(-0.75, 0.75)
            axes[0].set_title(neuron_name)
            axes[0].legend()

            for ax in [axes[1], axes[3]]:
                ax.set_xlim(xlim[0], xlim[1])
                ax.legend()
                ax.set_xlabel("Time (frames)")
                ax.set_ylabel("Normalized amplitude")
                if project_data is not None:
                    project_data.shade_axis_using_behavior(ax)

            axes[2].plot(all_dfs_list[0][neuron_name], all_dfs_list[1][neuron_name], '.')
            axes[2].set_xlabel("Red")
            axes[2].set_ylabel("Green")

            fig.tight_layout()

            # if to_save:
            #     fname = f'traces_kymo_correlation_{neuron_name}.png'
            #     self.project_data.save_fig_in_project(suffix=fname)

            if max_num_plots is not None and num_open_plots >= max_num_plots:
                break
        return all_figs


@dataclass
class MarkovRegressionModel:
    project_path: str

    behavior_to_predict: str = 'speed'

    project_data: ProjectData = None
    df: pd.DataFrame = None

    aic_list: list = None
    resid_list: list = None
    neuron_list: list = None
    results_list: list = None

    def __post_init__(self):

        project_data = ProjectData.load_final_project_data_from_config(self.project_path)
        self.project_data = project_data

        kwargs = dict(channel_mode='dr_over_r_20', min_nonnan=0.9)
        self.df = project_data.calc_default_traces(interpolate_nan=True, **kwargs)

    def get_valid_ind_and_trace(self) -> Tuple[np.ndarray, pd.Series]:
        if self.behavior_to_predict == 'speed':
            trace = self.project_data.worm_posture_class.worm_speed(fluorescence_fps=True, signed=True)
            trace = pd.Series(trace)
        elif self.behavior_to_predict == 'summed_curvature':
            worm = self.project_data.worm_posture_class
            trace = worm.summed_curvature_from_kymograph(fluorescence_fps=True)[worm.subsample_indices].copy().reset_index(drop=True)
        else:
            raise NotImplementedError(self.behavior_to_predict)

        valid_ind = np.where(~np.isnan(trace))[0]
        valid_ind = valid_ind[valid_ind < self.trace_len]

        return valid_ind, trace[valid_ind]

    @property
    def trace_len(self):
        return self.df.shape[0]

    def plot_no_neuron_markov_model(self, to_save=True):
        valid_ind, trace = self.get_valid_ind_and_trace()
        mod = sm.tsa.MarkovRegression(trace, k_regimes=2)
        res = mod.fit()
        pred = res.predict()

        plt.figure(dpi=100)
        plt.plot(trace, label=self.behavior_to_predict)
        plt.plot(pred, label=f'predicted {self.behavior_to_predict}')
        plt.legend()
        self.project_data.shade_axis_using_behavior()

        plt.ylabel(f"{self.behavior_to_predict}")
        plt.xlabel("Time (Frames)")
        r = trace.corr(pred)
        plt.title(f"Correlation: {r:.2f}")

        if to_save:
            fname = f'{self.behavior_to_predict}_no_neurons.png'
            self.project_data.save_fig_in_project(suffix=fname)

        plt.show()

    def calc_aic_feature_selected_neurons(self, num_iters=4):
        valid_ind, trace = self.get_valid_ind_and_trace()
        # Get features
        aic_list = []
        resid_list = []
        neuron_list = []

        remaining_neurons = get_names_from_df(self.df)
        previous_traces = []

        for i in tqdm(range(num_iters)):
            best_aic = 0
            best_resid = np.inf
            best_neuron = None

            for n in tqdm(remaining_neurons, leave=False):
                exog = pd.concat(previous_traces + [self.df[n][valid_ind]], axis=1)

                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=ConvergenceWarning)
                    warnings.simplefilter(action='ignore', category=ValueWarning)
                    warnings.simplefilter(action='ignore', category=RuntimeWarning)
                    mod = sm.tsa.MarkovRegression(trace, k_regimes=2, exog=exog)
                    res = mod.fit()

                if np.sum(res.resid ** 2) < best_resid:
                    best_resid = np.sum(res.resid ** 2)
                    best_aic = res.aic
                    best_neuron = n

            print(f"{best_neuron} selected for iteration {i}")
            aic_list.append(best_aic)
            resid_list.append(best_resid)
            neuron_list.append(best_neuron)
            previous_traces.append(self.df[best_neuron][valid_ind])
            remaining_neurons.remove(best_neuron)

        # Fit models
        results_list = []
        previous_traces = []
        for n in tqdm(neuron_list):
            exog = pd.concat(previous_traces + [self.df[n][valid_ind]], axis=1)
            mod = sm.tsa.MarkovRegression(trace, k_regimes=2, exog=exog)
            res = mod.fit()
            previous_traces.append(self.df[n][valid_ind])
            results_list.append(res)

        self.aic_list = aic_list
        self.resid_list = resid_list
        self.neuron_list = neuron_list
        self.results_list = results_list

    def plot_aic_feature_selected_neurons(self, to_save=True):
        valid_ind, trace = self.get_valid_ind_and_trace()
        if self.aic_list is None:
            self.calc_aic_feature_selected_neurons()
        aic_list = self.aic_list
        resid_list = self.resid_list
        neuron_list = self.neuron_list
        results_list = self.results_list

        # Plot 1
        fig, ax = plt.subplots(dpi=100)
        ax.plot(resid_list, label="Residual")

        ax2 = ax.twinx()
        ax2.plot(aic_list, label="AIC", c='tab:orange')

        ax.set_xticks(ticks=range(len(neuron_list)), labels=neuron_list, rotation=45)
        plt.xlabel("Neuron selected each iteration")

        if to_save:
            fname = f'{self.behavior_to_predict}_error_across_neurons.png'
            self.project_data.save_fig_in_project(fname)

        # Plot 2
        all_pred = [r.predict() for r in results_list]
        plt.figure(dpi=100)
        plt.plot(trace, label=self.behavior_to_predict, lw=2)

        for p, lab in zip(all_pred, neuron_list[0:8]):
            line = plt.plot(p, label=lab)
        plt.legend()
        plt.title("Predictions with cumulatively included neurons")
        plt.ylabel(f"{self.behavior_to_predict}")
        plt.xlabel("Time (Frames)")
        self.project_data.shade_axis_using_behavior()

        r = trace.corr(all_pred[-1])
        plt.title(f"Best correlation: {r:.2f}")

        if to_save:
            fname = f'{self.behavior_to_predict}_with_all_neurons_aic_feature_selected.png'
            self.project_data.save_fig_in_project(fname)

        # Plot 3
        all_traces = [self.df[n][valid_ind] for n in neuron_list]

        plt.figure(dpi=100)
        for i, (t, lab) in enumerate(zip(all_traces, neuron_list[:5])):
            line = plt.plot(t - i, label=lab)
        plt.legend()
        self.project_data.shade_axis_using_behavior()
        plt.title("Neurons selected as predictive (top is best)")

        if to_save:
            fname = f'{self.behavior_to_predict}_aic_predictive_traces.png'
            self.project_data.save_fig_in_project(fname)

        plt.show()


def calculate_post_reversal_peaks_from_project(all_projects_gcamp, neuron_names=None, behavior_names=None):
    if neuron_names is None:
        neuron_names = ['SMDVL', 'SMDVR', 'RIVL', 'RIVR', 'RID']

    all_peaks_named_neurons = defaultdict(dict)
    all_collision_flags = {}

    if behavior_names is None:
        behavior_names = ['ventral_only_head_curvature', 'summed_signed_curvature', 'ventral_quantile_curvature',
                          'head_signed_curvature', 'ventral_only_body_curvature']
    all_curvature_peaks_dict = {a: {} for a in behavior_names}
    for name, p in tqdm(all_projects_gcamp.items()):
        try:
            df_traces = p.calc_paper_traces()
            y_neurons = {n: df_traces[n] for n in neuron_names if n in df_traces}

            worm = p.worm_posture_class

            beh_annotation = worm.beh_annotation(fluorescence_fps=True, reset_index=True)
            y_collision = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.SELF_COLLISION).astype(int)

            all_collision_flags[name] = worm.get_peaks_post_reversal(y_collision, allow_reversal_before_peak=True)[0]


            for alias in all_curvature_peaks_dict.keys():
                y_curvature = worm.calc_behavior_from_alias(alias)
                curv_peaks = worm.get_peaks_post_reversal(y_curvature, allow_reversal_before_peak=True,
                                                          use_idx_of_absolute_max=True)[0]
                all_curvature_peaks_dict[alias][name] = curv_peaks

            # Even if the neurons aren't found, the lists must be populated to be the same length as all others
            for _name in neuron_names:
                y = y_neurons.get(_name, None)
                if y is None:
                    all_peaks_named_neurons[_name][name] = [np.nan] * len(curv_peaks)
                else:
                    all_peaks_named_neurons[_name][name] = \
                        worm.get_peaks_post_reversal(y, allow_reversal_before_peak=True)[0]

        except (ValueError, KeyError) as e:
            print(f"Error on dataset: {name}")
            print(e)

    # Reshape to final output
    my_reshape = lambda d: pd.DataFrame.from_dict(d, orient='index').T.stack(dropna=True).reset_index(drop=False)
    all_columns_neurons = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for
                           k, d in all_peaks_named_neurons.items()]

    df_neurons = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'),
                        all_columns_neurons)
    # Make sure the neurons are actually aligned with the other events
    df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().reset_index(drop=False).rename(
        columns={'level_0': 'dataset_idx', 'level_1': 'dataset_name'})
    all_columns_curvature = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'})
                             for k, d in all_curvature_peaks_dict.items()]
    df_curvature = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'),
                          all_columns_curvature)

    all_dfs = [df_neurons, df_collision, df_curvature]
    df_multi_neurons = reduce(
        lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_dfs)

    # df_multi_neurons = pd.concat([df_neurons, df_collision.astype(bool), df_curvature], join='inner', axis=1)
    df_multi_neurons.columns = ('dataset_idx', 'dataset_name') + tuple(neuron_names) + \
                               ('collision_flag',) + tuple(behavior_names)
    df_multi_neurons['collision_flag'] = df_multi_neurons['collision_flag'].astype(bool)

    df_multi_neurons['ventral_body_curvature'] = df_multi_neurons['summed_signed_curvature'] > 0

    return df_multi_neurons
