import numpy as np
import pandas as pd
import warnings
import itertools
import scipy
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    recall_score,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from prediction_utils.util import df_dict_concat
from collections import ChainMap
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

"""
Evaluators
    - StandardEvaluator
        - Computes standard model performance metrics
    - FairOVAEvaluator
        - Computes fairness metrics in one-vs-all (OVA) fashion
    - CalibrationEvaluator
        - Computes calibration metrics (absolute and relative calibration error) 
            using an auxiliary estimator
"""


class StandardEvaluator:
    def __init__(self, metrics=None, threshold_metrics=None, thresholds=None, *args, **kwargs):
        # default behavior: use all metrics, do not use any threshold metrics
        if metrics is None:
            self.metrics = self.get_default_threshold_free_metrics()
        else:
            self.metrics = metrics
        self.thresholds = thresholds

        if self.thresholds is not None:
            assert isinstance(self.thresholds, list)
            self.thresholds = [float(x) for x in self.thresholds]

        self.threshold_metrics = threshold_metrics
        self.args = {**kwargs}

    def evaluate(
        self,
        df,
        strata_vars=None,
        result_name="performance",
        weight_var=None,
        label_var="labels",
        pred_prob_var="pred_probs",
    ):
        """
        Evaluates predictions against a set of labels with a set of metric functions
        Arguments:
            df: a dataframe with one row per prediction
            result_name: a string that will be used to label the metric values in the result
            weight_var: a string identifier for sample weights in df
            label_var: a string identifier for the outcome labels in df
            pred_prob_var: a string identifier for the predicted probabilities in df
        """
        metric_fns = self.get_metric_fns(
            metrics=self.metrics,
            threshold_metrics=self.threshold_metrics,
            thresholds=self.thresholds,
            weighted=weight_var is not None,
        )

        if df[pred_prob_var].dtype == "float32":
            df[pred_prob_var] = df[pred_prob_var].astype(np.float64)

        if strata_vars is not None:
            strata_vars = [var for var in strata_vars if var in df.columns]
        if (strata_vars is None) or (len(strata_vars) == 0):
            result_df = (
                pd.DataFrame(
                    {
                        metric: metric_fn(
                            df[label_var].values, 
                            df[pred_prob_var].values,
                            **self.args if metric=='auprc_c' else {},
                        )
                        if weight_var is None
                        else metric_fn(
                            df[label_var].values,
                            df[pred_prob_var].values,
                            sample_weight=df[weight_var].values,
                            **self.args if metric=='auprc_c' else {},
                        )
                        for metric, metric_fn in metric_fns.items()
                    },
                    index=[result_name],
                )
                .transpose()
                .rename_axis("metric")
                .reset_index()
            )
        else:
            result_df = df_dict_concat(
                {
                    metric: df.groupby(strata_vars)
                    .apply(
                        lambda x: metric_func(
                            x[label_var].values, 
                            x[pred_prob_var].values,
                            **self.args if metric=='auprc_c' else {},
                        )
                        if weight_var is None
                        else metric_func(
                            x[label_var].values,
                            x[pred_prob_var].values,
                            sample_weight=x[weight_var].values,
                            **self.args if metric=='auprc_c' else {},
                        )
                    )
                    .rename(index=result_name)
                    .rename_axis(strata_vars)
                    .reset_index()
                    for metric, metric_func in metric_fns.items()
                },
                "metric",
            )
        return result_df

    def get_result_df(
        self,
        df,
        strata_vars=None,
        weight_var=None,
        label_var="labels",
        pred_prob_var="pred_probs",
        group_var_name="group",
        result_name="performance",
        compute_group_min_max=False,
        group_overall_name="overall",
    ):
        """
        A convenience function that calls evaluate with and without stratifying on group_var_name
        """
        if strata_vars is not None:
            strata_vars = [var for var in strata_vars if var in df.columns]
        else:
            strata_vars = []

        if group_overall_name in (df[group_var_name].unique()):
            raise ValueError("group_overall_name must not be a defined group")

        if group_var_name in strata_vars:
            strata_vars = strata_vars.copy()
            strata_vars.remove(group_var_name)

        result_df_by_group = self.evaluate(
            df,
            strata_vars=strata_vars + [group_var_name],
            result_name=result_name,
            weight_var=weight_var,
            label_var=label_var,
            pred_prob_var=pred_prob_var,
        )

        if compute_group_min_max:
            result_df_min_max = self.compute_group_min_max_fn(
                result_df_by_group,
                group_var_name=group_var_name,
                strata_vars=strata_vars,
            )
            result_df_min_max[group_var_name] = group_overall_name
            result_df_by_group = pd.concat([result_df_by_group, result_df_min_max])

        result_df_overall = self.evaluate(
            df,
            strata_vars=strata_vars,
            result_name=result_name,
            weight_var=weight_var,
            label_var=label_var,
            pred_prob_var=pred_prob_var,
        )

        result_df_overall[group_var_name] = group_overall_name

        result_df = pd.concat([result_df_by_group, result_df_overall])

        return result_df

    def compute_group_min_max_fn(
        self, df, group_var_name, result_name="performance", strata_vars=None
    ):
        """
        Computes the min and max of metrics across groups
        """
        strata_vars = self.union_lists(["metric"], strata_vars)
        result = (
            df.query("~{}.isnull()".format(group_var_name), engine="python")
            .groupby(strata_vars)[[result_name]]
            .agg(["min", "max"])
            .reset_index()
            .melt(id_vars=strata_vars)
            .assign(metric=lambda x: x["metric"].str.cat(x["variable_1"], sep="_"))
            .rename(columns={"value": result_name})
            .drop(columns=["variable_0", "variable_1"])
        )
        return result

    def evaluate_by_group(self, *args, **kwargs):
        """
        Deprecated, but keeping around for legacy purposes
        """
        warnings.warn("evaluate_by_group is deprecated, use evaluate")
        return self.evaluate(*args, **kwargs)

    def get_metric_fns(
        self, metrics=None, threshold_metrics=None, thresholds=None, weighted=False
    ):
        """
        Returns a dictionary of metric functions
        Arguments
            metrics: a list of string identifiers for metrics defined in get_threshold_free_metrics
            threshold_metrics: a list of string identifiers for metrics defined in get_threshold_metrics
            thresholds: a list of thresholds to evaluate the threshold based metrics at
            weighted: whether the threshold metric functions returned should take a sample_weight argument
        """
        threshold_free_metrics = self.get_threshold_free_metrics(metrics=metrics,)
        threshold_metrics = self.get_threshold_metrics(
            threshold_metrics=threshold_metrics,
            thresholds=thresholds,
            weighted=weighted,
        )
        return {**threshold_free_metrics, **threshold_metrics}

    def get_default_threshold_free_metrics(self):
        """
        Defines the string identifiers for the default threshold free metrics
        """
        return [
            "auc",
            "auprc",
            "loss_bce",
            "ace_rmse_logistic_log",
            "ace_abs_logistic_log",
        ]

    def get_threshold_free_metrics(self, metrics=None):
        """
        Defines the set of allowable threshold free metric functions
        """
        base_metric_dict = {
            "auc": try_roc_auc_score,
            "auprc": average_precision_score,
            "auprc_c": average_precision_score_calibrated,
            "brier": brier_score_loss,
            "loss_bce": try_log_loss,
            "ece_q_abs": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="abs", quantile_bins=True, **kwargs
            ),
            "ece_q_rmse": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="rmse", quantile_bins=True, **kwargs
            ),
            "ece_abs": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="abs", quantile_bins=False, **kwargs
            ),
            "ece_rmse": lambda *args, **kwargs: expected_calibration_error(
                *args, metric_variant="rmse", quantile_bins=False, **kwargs
            ),
            "ace_abs_logistic_log": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="abs",
                model_type="logistic",
                transform="log",
                **kwargs,
            ),
            "ace_abs_bin_log": lambda *args, **kwargs: try_absolute_calibration_error(
                *args, metric_variant="abs", model_type="bin", transform="log", **kwargs
            ),
            "ace_rmse_logistic_log": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="logistic",
                transform="log",
                **kwargs,
            ),
            "ace_rmse_bin_log": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="bin",
                transform="log",
                **kwargs,
            ),
            "ace_signed_logistic_log": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="logistic",
                transform="log",
                **kwargs,
            ),
            "ace_signed_bin_log": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="bin",
                transform="log",
                **kwargs,
            ),
            "ace_abs_logistic_logit": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="abs",
                model_type="logistic",
                transform="logit",
                **kwargs,
            ),
            "ace_abs_bin_logit": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="abs",
                model_type="bin",
                transform="logit",
                **kwargs,
            ),
            "ace_rmse_logistic_logit": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="logistic",
                transform="logit",
                **kwargs,
            ),
            "ace_rmse_bin_logit": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="bin",
                transform="logit",
                **kwargs,
            ),
            "ace_signed_logistic_logit": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="logistic",
                transform="logit",
                **kwargs,
            ),
            "ace_signed_bin_logit": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="bin",
                transform="logit",
                **kwargs,
            ),
            "ace_abs_logistic_none": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="abs",
                model_type="logistic",
                transform=None,
                **kwargs,
            ),
            "ace_abs_bin_none": lambda *args, **kwargs: try_absolute_calibration_error(
                *args, metric_variant="abs", model_type="bin", transform=None, **kwargs
            ),
            "ace_rmse_logistic_none": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="logistic",
                transform=None,
                **kwargs,
            ),
            "ace_rmse_bin_none": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="rmse",
                model_type="bin",
                transform=None,
                **kwargs,
            ),
            "ace_signed_logistic_none": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="logistic",
                transform=None,
                **kwargs,
            ),
            "ace_signed_bin_none": lambda *args, **kwargs: try_absolute_calibration_error(
                *args,
                metric_variant="signed",
                model_type="bin",
                transform=None,
                **kwargs,
            ),
            "mean_prediction": lambda *args, **kwargs: mean_prediction(
                *args, the_label=None, **kwargs
            ),
            "mean_prediction_0": lambda *args, **kwargs: mean_prediction(
                *args, the_label=0, **kwargs
            ),
            "mean_prediction_1": lambda *args, **kwargs: mean_prediction(
                *args, the_label=1, **kwargs
            ),
            "outcome_rate": lambda *args, **kwargs: outcome_rate(*args, **kwargs),
        }
        if metrics is None:
            return base_metric_dict
        else:
            return {
                key: base_metric_dict[key]
                for key in metrics
                if key in base_metric_dict.keys()
            }

    def get_threshold_metrics(
        self,
        threshold_metrics=None,
        thresholds=[0.01, 0.05, 0.1, 0.2, 0.5],
        weighted=False,
    ):
        """
        Returns a set of metric functions that are defined with respect to a set of thresholds
        """
        if thresholds is None:
            return {}

        if threshold_metrics is None:
            threshold_metrics = [
                "recall",
                "precision",
                "specificity",
            ]  # acts as default value

        result = {}

        if "recall" in threshold_metrics:
            result["recall"] = {
                "recall_{}".format(threshold): generate_recall_at_threshold(
                    threshold, weighted=weighted
                )
                for threshold in thresholds
            }
        if "precision" in threshold_metrics:
            result["precision"] = {
                "precision_{}".format(threshold): generate_precision_at_threshold(
                    threshold, weighted=weighted
                )
                for threshold in thresholds
            }
        if "specificity" in threshold_metrics:
            result["specificity"] = {
                "specificity_{}".format(threshold): generate_specificity_at_threshold(
                    threshold, weighted=weighted
                )
                for threshold in thresholds
            }

        if "tce_abs_logistic_logit" in threshold_metrics:
            result["tce_abs_logistic_logit"] = {
                "tce_abs_logistic_logit_{}".format(
                    threshold
                ): generate_threshold_calibration_error(
                    threshold=threshold,
                    metric_variant="abs",
                    model_type="logistic",
                    transform="logit",
                )
                for threshold in thresholds
            }
        if "tce_abs_logistic_log" in threshold_metrics:
            result["tce_abs_logistic_log"] = {
                "tce_abs_logistic_log_{}".format(
                    threshold
                ): generate_threshold_calibration_error(
                    threshold=threshold,
                    metric_variant="abs",
                    model_type="logistic",
                    transform="log",
                )
                for threshold in thresholds
            }
        if "tce_abs_logistic_none" in threshold_metrics:
            result["tce_abs_logistic_none"] = {
                "tce_abs_logistic_none_{}".format(
                    threshold
                ): generate_threshold_calibration_error(
                    threshold=threshold,
                    metric_variant="abs",
                    model_type="logistic",
                    transform=None,
                )
                for threshold in thresholds
            }

        if "tce_signed_logistic_logit" in threshold_metrics:
            result["tce_signed_logistic_logit"] = {
                "tce_signed_logistic_logit_{}".format(
                    threshold
                ): generate_threshold_calibration_error(
                    threshold=threshold,
                    metric_variant="signed",
                    model_type="logistic",
                    transform="logit",
                )
                for threshold in thresholds
            }
        if "tce_signed_logistic_log" in threshold_metrics:
            result["tce_signed_logistic_log"] = {
                "tce_signed_logistic_log_{}".format(
                    threshold
                ): generate_threshold_calibration_error(
                    threshold=threshold,
                    metric_variant="signed",
                    model_type="logistic",
                    transform="log",
                )
                for threshold in thresholds
            }
        if "tce_signed_logistic_none" in threshold_metrics:
            result["tce_signed_logistic_none"] = {
                "tce_signed_logistic_none_{}".format(
                    threshold
                ): generate_threshold_calibration_error(
                    threshold=threshold,
                    metric_variant="signed",
                    model_type="logistic",
                    transform=None,
                )
                for threshold in thresholds
            }

        if len(result) > 0:
            return dict(ChainMap(*result.values()))
        else:
            return result

    def clean_result_df(self, df):

        return (
            df.query("(not performance.isnull())", engine="python")
            .query('(not (metric == "auc" & (performance < 0.0)))')
            .query('(not (metric == "loss_bce" & (performance == 1e18)))')
        )

    @staticmethod
    def union_lists(x=None, y=None):

        if x is not None:
            assert isinstance(x, list)
            if y is None:
                return x
        if y is not None:
            assert isinstance(y, list)
            if x is None:
                return y

        if (x is not None) and (y is not None):
            return list(set(x) | set(y))

    def bootstrap_evaluate(
        self,
        df,
        n_boot=1000,
        strata_vars_eval=None,
        strata_vars_boot=None,
        strata_var_replicate=None,
        replicate_aggregation_mode=None,
        strata_var_experiment=None,
        baseline_experiment_name=None,
        strata_var_group=None,
        compute_overall=False,
        group_overall_name="overall",
        compute_group_min_max=False,
        result_name="performance",
        weight_var=None,
        label_var="labels",
        pred_prob_var="pred_probs",
        patient_id_var="person_id",
        n_jobs=None,
        verbose=False,
        return_result_df=False,
    ):
        """
        Arguments
            df: A dataframe to evaluate
            n_boot: The number of bootstrap iterations
            stata_vars_eval: The variables for perform stratified evaluation on
            strata_vars_boot: The variables to stratify the bootstrap sampling on
            strata_vars_replicate: A variable designating replicates
            replicate_aggregation_mode: None or 'mean'
            strata_var_experiment: The variable designating experimental condition column
            baseline_experiment_name: An element of strata_var_experiment column designating a baseline experiment
            strata_var_group: The variable designating a group
            compute_overall: If true, computes overall metrics without stratifying by group
            compute_group_min_max: If true, computes min and max metrics without stratifying by group
            result_name: The name of the returned metrics in the result dataframe
            weight_var: The variable designating sample weights
            label_var: The variable designating the outcome variable
            pred_probs_var: The variable designating the predicted score
            n_jobs: If None, runs bootstrap iterations serially. Otherwise, specifies the number of jobs for joblib parallelization. -1 uses all cores
        """

        def compute_bootstrap(i=None, verbose=False):
            if verbose:
                print(f"Bootstrap iteration: {i}")
            cohort_boot = (
                df[[patient_id_var] + strata_vars_boot]
                .drop_duplicates()
                .groupby(strata_vars_boot)
                .sample(frac=1.0, replace=True)
            )

            df_boot = df.merge(cohort_boot)
            if compute_overall or compute_group_min_max:
                return self.get_result_df(
                    df=df_boot,
                    strata_vars=strata_vars_eval,
                    group_var_name=strata_var_group,
                    weight_var=weight_var,
                    compute_group_min_max=compute_group_min_max,
                    group_overall_name=group_overall_name,
                    label_var=label_var,
                )
            else:
                return self.evaluate(
                    df=df_boot,
                    strata_vars=strata_vars_eval,
                    weight_var=weight_var,
                    result_name=result_name,
                    label_var=label_var,
                )

        if n_jobs is not None:
            result = Parallel(n_jobs=n_jobs)(
                delayed(compute_bootstrap)(i, verbose=verbose) for i in range(n_boot)
            )
            result_df = (
                pd.concat(result, keys=np.arange(len(result)))
                .reset_index(level=-1, drop=True)
                .rename_axis("boot_id")
                .reset_index()
            )
        else:
            result_df_dict = {}
            for i in range(n_boot):
                result_df_dict[i] = compute_bootstrap(i, verbose=verbose)
            result_df = (
                pd.concat(result_df_dict)
                .reset_index(level=-1, drop=True)
                .rename_axis("boot_id")
                .reset_index()
            )

        strata_vars_ci = strata_vars_eval + ["metric"]

        if strata_var_replicate is not None:
            strata_vars_ci.remove(strata_var_replicate)
            if replicate_aggregation_mode is None:
                pass
            elif replicate_aggregation_mode == "mean":
                result_df = (
                    result_df.groupby(strata_vars_ci + ["boot_id"])
                    .agg(performance=(result_name, "mean"))
                    .reset_index()
                )
            else:
                raise ValueError("Invalid aggregation mode")

        ## Aggregates results ##
        if (strata_var_experiment is not None) and (
            baseline_experiment_name is not None
        ):

            result_df_baseline = result_df.query(
                f"{strata_var_experiment} == @baseline_experiment_name"
            )
            result_df_baseline = result_df_baseline.drop(columns=strata_var_experiment)
            result_df_baseline = result_df_baseline.rename(
                columns={f"{result_name}": f"{result_name}_baseline"}
            ).reset_index(drop=True)
            result_df_merged = result_df.merge(result_df_baseline)
            assert result_df.shape[0] == result_df_merged.shape[0]

            result_df = result_df_merged
            result_df[f"{result_name}_delta"] = (
                result_df[f"{result_name}"] - result_df[f"{result_name}_baseline"]
            )

            result_df_ci = (
                result_df.groupby(strata_vars_ci)
                .apply(
                    lambda x: pd.DataFrame(
                        {
                            "comparator": np.quantile(
                                x[f"{result_name}"], [0.025, 0.5, 0.975]
                            ),
                            "baseline": np.quantile(
                                x[f"{result_name}_baseline"], [0.025, 0.5, 0.975]
                            ),
                            "delta": np.quantile(
                                x[f"{result_name}_delta"], [0.025, 0.5, 0.975]
                            ),
                        }
                    )
                    .rename_axis("CI_quantile_95")
                    .rename({i: el for i, el in enumerate(["lower", "mid", "upper"])})
                )
                .reset_index()
            )
        else:
            # If there are no baselines
            result_df_ci = (
                result_df.groupby(strata_vars_ci)
                .apply(lambda x: np.quantile(x[result_name], [0.025, 0.5, 0.975]))
                .rename(result_name)
                .reset_index()
                .assign(
                    CI_lower=lambda x: x[result_name].str[0],
                    CI_med=lambda x: x[result_name].str[1],
                    CI_upper=lambda x: x[result_name].str[2],
                )
                .drop(columns=[result_name])
            )
        
        if return_result_df:
            return result_df_ci, result_df
        else:
            return result_df_ci


class FairOVAEvaluator(StandardEvaluator):
    """
    Computes fairness metrics in a "one-vs-all" (OVA) manner
    Supported functionality
        * Computing a metric for each group and comparing to the value for the whole population
        * Computing cross-group ranking metrics (xAUC)
        * Comparing the distribution of predictions for each group with the marginal
    """

    def __init__(self, metrics=None, threshold_metrics=None, thresholds=None):
        super().__init__(
            metrics=metrics, threshold_metrics=threshold_metrics, thresholds=thresholds
        )

    def get_fair_ova_fns(self, weighted=False):
        """
        Returns a dictionary of fair OVA metric functions
        """
        metric_fn_dict = self.get_metric_fns(
            metrics=self.metrics,
            threshold_metrics=self.threshold_metrics,
            thresholds=self.thresholds,
            weighted=weighted,
        )
        # Lazy evaluation, see https://sopython.com/wiki/Common_Gotchas_In_Python
        metric_fairness_ova_dict = {
            "{}_ova".format(
                key
            ): lambda *args, metric_fn=value, **kwargs: metric_fairness_ova(
                *args, metric_fn=metric_fn, **kwargs
            )
            for key, value in metric_fn_dict.items()
        }
        xauc_dict = self.get_xauc_fns(weighted=weighted)
        emd_ova_dict = self.get_emd_fns(weighted=weighted)
        fair_fn_dict = {**metric_fairness_ova_dict, **emd_ova_dict, **xauc_dict}
        return fair_fn_dict

    def get_xauc_fns(self, weighted=False):
        """
        Returns a dictionary of xAUC metric functions
        """
        return {
            "xauc_1_ova": lambda *args, **kwargs: xauc(
                *args, **kwargs, the_label=1, exclude_the_group_from_marginal=False
            ),
            "xauc_0_ova": lambda *args, **kwargs: xauc(
                *args, **kwargs, the_label=0, exclude_the_group_from_marginal=False
            ),
            "xauc_1": lambda *args, **kwargs: xauc(
                *args, **kwargs, the_label=1, exclude_the_group_from_marginal=True
            ),
            "xauc_0": lambda *args, **kwargs: xauc(
                *args, **kwargs, the_label=0, exclude_the_group_from_marginal=True
            ),
        }

    def get_emd_fns(self, weighted=False):
        """
        Returns functions that compares the earth movers distance across distributions of predictions across groups
        """
        return {
            "emd_ova": lambda *args, **kwargs: emd_ova(*args, the_label=None, **kwargs),
            "emd_0_ova": lambda *args, **kwargs: emd_ova(*args, the_label=0, **kwargs),
            "emd_1_ova": lambda *args, **kwargs: emd_ova(*args, the_label=1, **kwargs),
        }

    def get_result_df(
        self,
        df,
        strata_vars=["attribute", "task", "phase"],
        group_var_name="group",
        weight_var=None,
        label_var="labels",
        pred_prob_var="pred_probs",
        result_name="performance",
    ):
        """
        Evaluates predictions against a set of labels with a set of fair OVA metric function
        Arguments:
            df: a dataframe with one row per prediction
            strata_vars: a list of string identifiers of columns to stratify the evaluation on
            group_var_name: a string identifier indicating which column corresponds to the group variable
            weight_var: a string identifier for sample weights in df
            label_var: a string identifier for the outcome labels in df
            pred_prob_var: a string identifier for the predicted probabilities in df
            result_name: a string that will be used to label the metric values in the result
        """
        assert group_var_name is not None
        metric_fns = self.get_fair_ova_fns(weighted=weight_var is not None)

        if df[pred_prob_var].dtype == "float32":
            # df[pred_prob_var] = df[pred_prob_var].astype(np.float64)
            df.loc[:, pred_prob_var] = df[pred_prob_var].astype(np.float64)

        if strata_vars is None:
            strata_vars = ["task", "attribute", "phase"]

        strata_vars = [var for var in strata_vars if var in df.columns]
        if len(strata_vars) == 0:
            # No strata vars are provided
            result = {}
            for metric_fn_key, group_id in itertools.product(
                metric_fns.keys(), df[group_var_name].unique()
            ):
                result[(metric_fn_key, group_id)] = metric_fns[metric_fn_key](
                    df[label_var].values,
                    df[pred_prob_var].values,
                    df[group_var_name].values,
                    the_group=group_id,
                    sample_weight=df[weight_var].values
                    if weight_var is not None
                    else None,
                )
            result_df = pd.DataFrame(result, index=[0]).transpose()
            result_df = (
                result_df.rename_axis(["metric", group_var_name])
                .rename(columns={0: result_name})
                .reset_index()
            )

        else:
            stratified = df.groupby(strata_vars)
            result = {}
            for (meta, df_strata) in stratified:
                if isinstance(meta, str) or isinstance(meta, int):
                    # If only one strata var
                    meta = tuple([meta])
                for metric_fn_key, group_id in itertools.product(
                    metric_fns.keys(), df_strata[group_var_name].unique()
                ):
                    result_key = tuple([metric_fn_key]) + meta + tuple([group_id])
                    result[result_key] = metric_fns[metric_fn_key](
                        df_strata[label_var].values,
                        df_strata[pred_prob_var].values,
                        df_strata[group_var_name].values,
                        the_group=group_id,
                        sample_weight=df_strata[weight_var].values
                        if weight_var is not None
                        else None,
                    )
            result_df = pd.DataFrame(result, index=[0]).transpose()
            result_df = (
                result_df.rename_axis(["metric"] + strata_vars + [group_var_name])
                .rename(columns={0: result_name})
                .reset_index()
            )

        return result_df

    def evaluate(self, *args, **kwargs):
        return self.get_result_df(*args, **kwargs)


class CalibrationEvaluator:
    """
    Evaluator that computes absolute and relative calibration errors
    TODOs:
        Single call to generate 
            a set of absolute metrics (ACE and TCE)
            using a group variable to generate RCE and R-TCE
            with bootstrapping functionality
            and the ability to log calibration curves
    """

    @staticmethod
    def clean_for_log_transform(x, eps=1e-15):
        return np.maximum(np.minimum(x, 1 - eps), eps)

    def get_calibration_curve_df(
        self, df, label_var, pred_prob_var, weight_var=None, strata_vars=None, **kwargs
    ):

        if strata_vars is None:
            result = self.get_calibration_curve(
                labels=df[label_var],
                pred_probs=df[pred_prob_var],
                sample_weight=df[weight_var] if weight_var is not None else None,
                **kwargs,
            )
        else:
            result = (
                df.groupby(strata_vars)
                .apply(
                    lambda x: self.get_calibration_curve(
                        labels=x[label_var],
                        pred_probs=x[pred_prob_var],
                        sample_weight=x[weight_var] if weight_var is not None else None,
                        **kwargs,
                    )
                )
                .reset_index(level=-1, drop=True)
                .reset_index()
            )
        return result

    def get_calibration_curve(
        self,
        labels,
        pred_probs,
        sample_weight=None,
        model_type="logistic",
        transform=None,
        return_model=False,
        score_values=None,
        **kwargs,
    ):
        model = self.init_model(model_type=model_type, **kwargs)

        df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels})
        if sample_weight is not None:
            df = df.assign(sample_weight=sample_weight)

        if score_values is None:
            score_values = np.linspace(1e-4, 1 - 1e-4, 1000)
        valid_transforms = ["log", "c_log_log", "logit"]
        if transform is None:
            df = df.assign(model_input=lambda x: x.pred_probs)
            model_predict_input = score_values.copy()
        elif transform in valid_transforms:
            df = df.query("(pred_probs > 1e-15) & (pred_probs < (1 - 1e-15))")
            if transform == "log":
                df = df.assign(model_input=lambda x: np.log(x.pred_probs))
                model_predict_input = np.log(score_values.copy())
            elif transform == "c_log_log":
                df = df.assign(model_input=lambda x: self.c_log_log(x.pred_probs))
                model_predict_input = self.c_log_log(score_values.copy())
            elif transform == "logit":
                df = df.assign(model_input=lambda x: self.logit(x.pred_probs))
                model_predict_input = self.logit(score_values.copy())
        else:
            raise ValueError("Invalid transform provided")
        model.fit(
            df.model_input.values.reshape(-1, 1),
            df.labels.values,
            sample_weight=df.sample_weight.values
            if "sample_weight" in df.columns
            else None,
        )
        pred_probs = model.predict_proba(model_predict_input.reshape(-1, 1))
        if len(pred_probs.shape) > 1:
            pred_probs = pred_probs[:, -1]
        result = pd.DataFrame(
            {
                "score": score_values,
                "calibration_density": pred_probs
            }
        )

        if return_model:
            return result, model
        else:
            return result

    def bootstrap_calibration_curve(
        self,
        df,
        n_boot=10,
        strata_vars_eval=None,
        strata_vars_boot=None,  # required
        strata_var_replicate=None,
        replicate_aggregation_mode=None,
        strata_var_experiment=None,
        baseline_experiment_name=None,
        result_name="performance",
        weight_var=None,
        label_var="labels",
        pred_prob_var="pred_probs",
        patient_id_var="person_id",
        n_jobs=None,
        verbose=False,
        **kwargs,
    ):
        """
        Arguments
            df: A dataframe to evaluate
            n_boot: The number of bootstrap iterations
            stata_vars_eval: The variables for perform stratified evaluation on
            strata_vars_boot: The variables to stratify the bootstrap sampling on
            strata_vars_replicate: A variable designating replicates
            replicate_aggregation_mode: None or 'mean'
            strata_var_experiment: The variable designating experimental condition column
            baseline_experiment_name: An element of strata_var_experiment column designating a baseline experiment
            result_name: The name of the returned metrics in the result dataframe
            weight_var: The variable designating sample weights
            label_var: The variable designating the outcome variable
            pred_probs_var: The variable designating the predicted score
            n_jobs: If None, runs bootstrap iterations serially. Otherwise, specifies the number of jobs for joblib parallelization. -1 uses all cores
        """

        def compute_bootstrap(i=None, verbose=False):
            if verbose:
                print(f"Bootstrap iteration: {i}")
            cohort_boot = (
                df[
                    [patient_id_var] + strata_vars_boot
                    if strata_vars_boot is not None
                    else [patient_id_var]
                ]
                .drop_duplicates()
                .groupby(strata_vars_boot)
                .sample(frac=1.0, replace=True)
            )

            df_boot = df.merge(cohort_boot)
            return self.get_calibration_curve_df(
                df=df_boot,
                strata_vars=strata_vars_eval,
                weight_var=weight_var,
                label_var=label_var,
                pred_prob_var=pred_prob_var,
                **kwargs,
            )

        if n_jobs is not None:
            result = Parallel(n_jobs=n_jobs)(
                delayed(compute_bootstrap)(i, verbose=verbose) for i in range(n_boot)
            )
            result_df = (
                pd.concat(result, keys=np.arange(len(result)))
                .reset_index(level=-1, drop=True)
                .rename_axis("boot_id")
                .reset_index()
            )
        else:
            result_df_dict = {}
            for i in range(n_boot):
                result_df_dict[i] = compute_bootstrap(i, verbose=verbose)
            result_df = (
                pd.concat(result_df_dict)
                .reset_index(level=-1, drop=True)
                .rename_axis("boot_id")
                .reset_index()
            )

        strata_vars_ci = strata_vars_eval + ["score"]

        if strata_var_replicate is not None:
            strata_vars_ci.remove(strata_var_replicate)
            if replicate_aggregation_mode is None:
                pass
            elif replicate_aggregation_mode == "mean":
                result_df = (
                    result_df.groupby(strata_vars_ci + ["boot_id"])
                    .agg(performance=("calibration_density", "mean"))
                    .reset_index()
                )
            else:
                raise ValueError("Invalid aggregation mode")

        result_name = "calibration_density"
        result_df_ci = (
            result_df.groupby(strata_vars_ci)
            .apply(lambda x: np.quantile(x[result_name], [0.025, 0.5, 0.975]))
            .rename(result_name)
            .reset_index()
            .assign(
                CI_lower=lambda x: x[result_name].str[0],
                CI_med=lambda x: x[result_name].str[1],
                CI_upper=lambda x: x[result_name].str[2],
            )
            .drop(columns=[result_name])
        )
        return result_df_ci

    def get_calibration_density_df(
        self,
        labels,
        pred_probs,
        sample_weight=None,
        model_type="logistic",
        transform=None,
        **kwargs,
    ):

        model = self.init_model(model_type=model_type, **kwargs)

        df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels})
        if sample_weight is not None:
            df = df.assign(sample_weight=sample_weight)

        valid_transforms = ["log", "c_log_log", "logit"]
        if transform is None:
            df = df.assign(model_input=lambda x: x.pred_probs)
            model_input = df.model_input.values.reshape(-1, 1)
        elif transform in valid_transforms:
            df = df.query("(pred_probs > 1e-15) & (pred_probs < (1 - 1e-15))")
            if transform == "log":
                df = df.assign(model_input=lambda x: np.log(x.pred_probs))
            elif transform == "c_log_log":
                df = df.assign(model_input=lambda x: self.c_log_log(x.pred_probs))
            elif transform == "logit":
                df = df.assign(model_input=lambda x: self.logit(x.pred_probs))
        else:
            raise ValueError("Invalid transform provided")
        model_input = df.model_input.values.reshape(-1, 1)
        model.fit(
            model_input,
            df.labels.values,
            sample_weight=df.sample_weight.values
            if "sample_weight" in df.columns
            else None,
        )
        calibration_density = model.predict_proba(model_input)
        if len(calibration_density.shape) > 1:
            calibration_density = calibration_density[:, -1]

        df = df.assign(calibration_density=calibration_density)
        return df, model

    def absolute_calibration_error(
        self,
        labels,
        pred_probs,
        sample_weight=None,
        metric_variant="abs",
        model_type="logistic",
        transform=None,
    ):

        df, model = self.get_calibration_density_df(
            labels,
            pred_probs,
            sample_weight=sample_weight,
            model_type=model_type,
            transform=transform,
        )
        if "sample_weight" in df.columns:
            sample_weight = df.sample_weight
        else:
            sample_weight = None

        if metric_variant == "squared":
            return self.weighted_mean(
                (df.calibration_density - df.pred_probs) ** 2,
                sample_weight=sample_weight,
            )
        elif metric_variant == "rmse":
            return np.sqrt(
                self.weighted_mean(
                    (df.calibration_density - df.pred_probs) ** 2,
                    sample_weight=sample_weight,
                )
            )
        elif metric_variant == "abs":
            return self.weighted_mean(
                np.abs(df.calibration_density - df.pred_probs),
                sample_weight=sample_weight,
            )
        elif metric_variant == "signed":
            return self.weighted_mean(
                df.calibration_density - df.pred_probs, sample_weight=sample_weight
            )
        else:
            raise ValueError("Invalid option specified for metric")

    def threshold_calibration_error(
        self,
        labels,
        pred_probs,
        threshold,
        sample_weight=None,
        metric_variant="signed",
        model_type="logistic",
        transform=None,
    ):
        df, model = self.get_calibration_density_df(
            labels,
            pred_probs,
            sample_weight=sample_weight,
            model_type=model_type,
            transform=transform,
        )

        threshold = np.array(threshold)
        valid_transforms = ["log", "c_log_log", "logit"]
        model_input = threshold.reshape(-1, 1)
        if transform is None:
            pass
        elif transform in valid_transforms:
            model_input = self.clean_for_log_transform(model_input)
            if transform == "log":
                model_input = np.log(model_input)
            elif transform == "c_log_log":
                model_input = self.c_log_log(model_input)
            elif transform == "logit":
                model_input = self.logit(model_input)

        else:
            raise ValueError("Invalid transform provided")

        model_output = model.predict_proba(model_input)[:, -1]
        signed_difference = (model_output - threshold)[0]
        if metric_variant == "squared":
            return (signed_difference) ** 2
        elif metric_variant == "rmse":
            return np.sqrt((signed_difference) ** 2)
        elif metric_variant == "abs":
            return np.abs(signed_difference)
        elif metric_variant == "signed":
            return signed_difference

    def relative_calibration_error(
        self,
        labels,
        pred_probs,
        group,
        sample_weight=None,
        metric_variant="abs",
        model_type="logistic",
        transform=None,
        compute_ace=False,
        return_models=False,
        return_calibration_density=False,
    ):

        calibration_density_df_overall, model_overall = self.get_calibration_density_df(
            labels,
            pred_probs,
            sample_weight=sample_weight,
            model_type=model_type,
            transform=transform,
        )

        df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels, "group": group})
        if sample_weight is not None:
            df = df.assign(sample_weight=sample_weight)

        ace_dict = {}
        rce_dict = {}
        model_dict = {}
        calibration_density_dict = {}
        for group_id, group_df in df.groupby("group"):

            (
                calibration_density_dict[group_id],
                model_dict[group_id],
            ) = self.get_calibration_density_df(
                group_df.labels,
                group_df.pred_probs,
                sample_weight=group_df.sample_weight
                if "sample_weight" in group_df.columns
                else None,
                model_type=model_type,
                transform=transform,
            )

            calib_diff = (
                model_dict[group_id].predict_proba(
                    calibration_density_dict[group_id].model_input.values.reshape(
                        -1, 1
                    ),
                )[:, -1]
                - model_overall.predict_proba(
                    calibration_density_dict[group_id].model_input.values.reshape(
                        -1, 1
                    ),
                )[:, -1]
            )

            group_sample_weight = (
                calibration_density_dict[group_id].sample_weight
                if "sample_weight" in calibration_density_dict[group_id].columns
                else None
            )
            if metric_variant == "squared":
                rce_dict[group_id] = self.weighted_mean(
                    calib_diff ** 2, sample_weight=group_sample_weight
                )
            elif metric_variant == "rmse":
                rce_dict[group_id] = np.sqrt(
                    self.weighted_mean(
                        calib_diff ** 2, sample_weight=group_sample_weight
                    )
                )
            elif metric_variant == "abs":
                rce_dict[group_id] = self.weighted_mean(
                    np.abs(calib_diff), sample_weight=group_sample_weight
                )
            elif metric_variant == "signed":
                rce_dict[group_id] = self.weighted_mean(
                    calib_diff, sample_weight=group_sample_weight
                )
            else:
                raise ValueError("Invalid option specified for metric")

            if compute_ace:
                if metric_variant == "squared":
                    ace_dict[group_id] = self.weighted_mean(
                        (
                            calibration_density_dict[group_id].calibration_density
                            - calibration_density_dict[group_id].pred_probs
                        )
                        ** 2,
                        sample_weight=group_sample_weight,
                    )
                elif metric_variant == "rmse":
                    ace_dict[group_id] = np.sqrt(
                        self.weighted_mean(
                            (
                                calibration_density_dict[group_id].calibration_density
                                - calibration_density_dict[group_id].pred_probs
                            )
                            ** 2,
                            sample_weight=group_sample_weight,
                        )
                    )
                elif metric_variant == "abs":
                    ace_dict[group_id] = self.weighted_mean(
                        np.abs(
                            calibration_density_dict[group_id].calibration_density
                            - calibration_density_dict[group_id].pred_probs
                        ),
                        sample_weight=group_sample_weight,
                    )
                elif metric_variant == "signed":
                    ace_dict[group_id] = self.weighted_mean(
                        calibration_density_dict[group_id].calibration_density
                        - calibration_density_dict[group_id].pred_probs,
                        sample_weight=group_sample_weight,
                    )
                else:
                    raise ValueError("Invalid option specified for metric")
        result_dict = {}
        result_dict["result"] = (
            pd.DataFrame(rce_dict, index=["relative_calibration_error"])
            .transpose()
            .rename_axis("group")
            .reset_index()
        )
        if compute_ace:
            ace_df = (
                pd.DataFrame(ace_dict, index=["absolute_calibration_error"])
                .transpose()
                .rename_axis("group")
                .reset_index()
            )
            result_dict["result"] = result_dict["result"].merge(ace_df)
        if return_models:
            result_dict["model_dict_group"] = model_dict
            result_dict["model_overall"] = model_overall
        if return_calibration_density:
            result_dict["calibration_density_group"] = (
                pd.concat(calibration_density_dict)
                .reset_index(level=-1, drop=True)
                .rename_axis("group")
                .reset_index()
            )
            result_dict["calibration_density_overall"] = calibration_density_df_overall

        return result_dict

    @staticmethod
    def c_log_log(x):
        return np.log(-np.log(1 - x))

    @staticmethod
    def logit(x):
        return np.log(x / (1 - x))

    @staticmethod
    def weighted_mean(x, sample_weight=None):
        if sample_weight is None:
            return x.mean()
        else:
            return np.average(x, weights=sample_weight)

    def init_model(self, model_type, **kwargs):
        if model_type == "logistic":
            model = LogisticRegression(
                solver="lbfgs", penalty="none", max_iter=10000, **kwargs
            )
        elif model_type == "spline":

            n_knots = kwargs.get("n_knots", 3)
            degree = kwargs.get("degree", 3)
            knots = kwargs.get("knots", "quantile")
            extrapolation = kwargs.get("extrapolation", "linear")
            model = SplineLogisticRegression(
                n_knots=n_knots, degree=degree, knots=knots, extrapolation=extrapolation
            )
        elif model_type == "rf":
            model = RandomForestClassifier(**kwargs)
        elif model_type == "bin":
            model = BinningEstimator(**kwargs)
        else:
            raise ValueError("Invalid model_type not provided")
        return model


class BinningEstimator:
    def __init__(self, num_bins=10, quantile_bins=True):
        self.num_bins = num_bins
        self.discretizer = KBinsDiscretizer(
            n_bins=num_bins,
            encode="ordinal",
            strategy="quantile" if quantile_bins else "uniform",
        )
        self.prob_y_lookup = -1e18 * np.ones(num_bins)

    def fit(self, x, y, sample_weight=None):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        binned_x = self.discretizer.fit_transform(x)
        binned_x = binned_x.squeeze()
        for bin_id in range(self.num_bins):
            mask = binned_x == bin_id
            if (mask).sum() == 0:
                print("No data in bin {}".format(bin_id))
            if sample_weight is None:
                self.prob_y_lookup[bin_id] = y[mask].mean()
            else:
                self.prob_y_lookup[bin_id] = np.average(
                    y[mask], weights=sample_weight[mask]
                )

    def predict_proba(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        binned_x = self.discretizer.transform(x).squeeze().astype(np.int64)
        return self.prob_y_lookup[binned_x]


class SplineLogisticRegression:
    def __init__(self, **kwargs):
        from sklearn.preprocessing import SplineTransformer  # requires sklearn >= 1.0

        self.spline_transformer = SplineTransformer(**kwargs)
        self.logistic_regression = LogisticRegression(
            solver="lbfgs", penalty="none", max_iter=10000
        )
        self.pipeline = Pipeline(
            [
                ("spline_transformer", self.spline_transformer),
                ("logistic_regression", self.logistic_regression),
            ]
        )

    def fit(self, x, y, sample_weight=None):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        self.pipeline.fit(
            x,
            y,
            spline_transformer__sample_weight=sample_weight,
            logistic_regression__sample_weight=sample_weight,
        )

    def predict_proba(self, x):
        return self.pipeline.predict_proba(x)


"""
Metrics
A metric function takes the following arguments
    - labels: A vector of labels
    - pred_probs: A vector of predicted probabilities
    - sample_weight: A per-sample weight. Default to None
    - Optional keyword arguments
"""

"""
Threshold performance metrics.
These metrics define recall, precision, or specificity at a given threshold using the metric function interface
"""


def recall_at_threshold(labels, pred_probs, sample_weight=None, threshold=0.5):
    """
    Computes recall at a threshold
    """
    return threshold_metric_fn(
        labels=labels,
        pred_probs=pred_probs,
        sample_weight=sample_weight,
        threshold=threshold,
        metric_generator_fn=generate_recall_at_threshold,
    )


def generate_recall_at_threshold(threshold, weighted=False):
    """
    Returns a lambda function that computes the recall at a provided threshold.
    If weights = True, the lambda function takes a third argument for the sample weights
    """
    if not weighted:
        return lambda labels, pred_probs: recall_score(
            labels, 1.0 * (pred_probs >= threshold)
        )
    else:
        return lambda labels, pred_probs, sample_weight: recall_score(
            labels, 1.0 * (pred_probs >= threshold), sample_weight=sample_weight
        )


def precision_at_threshold(labels, pred_probs, sample_weight=None, threshold=0.5):
    """
    Computes precision at a threshold
    """
    return threshold_metric_fn(
        labels=labels,
        pred_probs=pred_probs,
        sample_weight=sample_weight,
        threshold=threshold,
        metric_generator_fn=generate_precision_at_threshold,
    )


def generate_precision_at_threshold(threshold, weighted=False):
    """
    Returns a lambda function that computes the precision at a provided threshold.
    If weights = True, the lambda function takes a third argument for the sample weights
    """
    if not weighted:
        return lambda labels, pred_probs: precision_score(
            labels, 1.0 * (pred_probs >= threshold), zero_division=0
        )
    else:
        return lambda labels, pred_probs, sample_weight: precision_score(
            labels,
            1.0 * (pred_probs >= threshold),
            zero_division=0,
            sample_weight=sample_weight,
        )


def specificity_at_threshold(labels, pred_probs, sample_weight=None, threshold=0.5):
    """
    Computes specificity at a threshold
    """
    return threshold_metric_fn(
        labels=labels,
        pred_probs=pred_probs,
        sample_weight=sample_weight,
        threshold=threshold,
        metric_generator_fn=generate_specificity_at_threshold,
    )


def generate_specificity_at_threshold(threshold, weighted=False):
    """
    Returns a lambda function that computes the specificity at a provided threshold.
    If weights = True, the lambda function takes a third argument for the sample weights
    """
    if not weighted:
        return (
            lambda labels, pred_probs: (
                (labels == 0) & (labels == (pred_probs >= threshold))
            ).sum()
            / (labels == 0).sum()
            if (labels == 0).sum() > 0
            else 0.0
        )
    else:
        return (
            lambda labels, pred_probs, sample_weight: (
                ((labels == 0) & (labels == (pred_probs >= threshold))) * sample_weight
            ).sum()
            / ((labels == 0) * sample_weight).sum()
            if (labels == 0).sum() > 0
            else 0.0
        )


def threshold_metric_fn(
    labels, pred_probs, sample_weight=None, threshold=0.5, metric_generator_fn=None
):
    """
    Function that generates threshold metric functions.
    Calls a metric_generator_fn for customization
    """
    if metric_generator_fn is None:
        raise ValueError("metric_generator_fn must not be None")

    metric_fn = metric_generator_fn(
        threshold=threshold, weighted=sample_weight is not None
    )
    if sample_weight is None:
        return metric_fn(labels, pred_probs)
    else:
        return metric_fn(labels, pred_probs, sample_weight=sample_weight)


def try_metric_fn(*args, metric_fn=None, default_value=-1, **kwargs):
    """
    Tries to call a metric function, returns default_value if fails
    """
    if metric_fn is None:
        raise ValueError("Must provide metric_fn")
    try:
        return metric_fn(*args, **kwargs)
    except ValueError:
        warnings.warn("Error in metric_fn, filling with default_value")
        return default_value


def expected_calibration_error(
    labels, pred_probs, num_bins=10, metric_variant="abs", quantile_bins=False
):
    """
        Computes the calibration error with a binning estimator over equal sized bins
        See http://arxiv.org/abs/1706.04599 and https://arxiv.org/abs/1904.01685.
        Does not currently support sample weights
    """
    if metric_variant == "abs":
        transform_func = np.abs
    elif (metric_variant == "squared") or (metric_variant == "rmse"):
        transform_func = np.square
    elif metric_variant == "signed":
        transform_func = identity
    else:
        raise ValueError("provided metric_variant not supported")

    if quantile_bins:
        cut_fn = pd.qcut
    else:
        cut_fn = pd.cut

    bin_ids = cut_fn(pred_probs, num_bins, labels=False, retbins=False)
    df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels, "bin_id": bin_ids})
    ece_df = (
        df.groupby("bin_id")
        .agg(
            pred_probs_mean=("pred_probs", "mean"),
            labels_mean=("labels", "mean"),
            bin_size=("pred_probs", "size"),
        )
        .assign(
            bin_weight=lambda x: x.bin_size / df.shape[0],
            err=lambda x: transform_func(x.pred_probs_mean - x.labels_mean),
        )
    )
    result = np.average(ece_df.err.values, weights=ece_df.bin_weight)
    if metric_variant == "rmse":
        result = np.sqrt(result)
    return result


def pointwise_expected_calibration_error(
    labels,
    pred_probs,
    sample_weight=None,
    num_bins=10,
    norm_order=1,
    quantile_bins=False,
):
    """
        Computes the calibration error with a binning estimator over equal sized bins
        Compares individual predicted probabilities with bin estimates
        This function implements a version that takes sample weights
        For simplicity, bin boundaries are derived from the unweighted sample
    """
    if norm_order == 1:
        transform_func = np.abs
    elif norm_order == 2:
        transform_func = np.square
    elif norm_order is None:
        transform_func = identity
    else:
        raise ValueError("only norm_order == 1, 2, or None supported")

    if quantile_bins:
        cut_fn = pd.qcut
    else:
        cut_fn = pd.cut

    bin_ids = cut_fn(pred_probs, num_bins, labels=False, retbins=False)
    data_dict = {"pred_probs": pred_probs, "labels": labels, "bin_id": bin_ids}
    if sample_weight is not None:
        data_dict["sample_weight"] = sample_weight
    df = pd.DataFrame(data_dict)
    if sample_weight is None:
        ece_df = df.groupby("bin_id").agg(labels_mean=("labels", "mean"),).reset_index()
        result_df = df.merge(ece_df)
        result_df = result_df.assign(
            err=lambda x: transform_func(x.pred_probs - x.labels_mean)
        )
        result = result_df.err.mean()
        if norm_order == 2:
            result = np.sqrt(result)
    else:
        ece_df = (
            df.groupby("bin_id")
            .apply(lambda x: np.average(x.labels, weights=x.sample_weight))
            .rename(index="labels_mean")
            .reset_index()
        )
        result_df = df.merge(ece_df)
        result_df = result_df.assign(
            err=lambda x: transform_func(x.pred_probs - x.labels_mean)
        )
        result = np.average(
            result_df.err.values, weights=result_df.sample_weight.values
        )
        if norm_order == 2:
            result = np.sqrt(result)
    return result


def identity(x):
    """
    Returns its argument
    """
    return x


def mean_prediction(labels, pred_probs, sample_weight=None, the_label=None):
    """
    Computes the mean prediction, optionally conditioning on the_label
    """
    if the_label is not None:
        mask = labels == the_label
        labels = labels[mask]
        pred_probs = pred_probs[mask]
        sample_weight = sample_weight[mask] if sample_weight is not None else None

    if sample_weight is not None:
        return np.average(pred_probs, weights=sample_weight)
    else:
        return pred_probs.mean()


def outcome_rate(labels, pred_probs=None, sample_weight=None):
    if sample_weight is not None:
        return np.average(labels, weights=sample_weight)
    else:
        return np.average(labels)


def _binary_clf_curve(y_true, y_score, pos_label=1, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    
    y_true = (y_true == pos_label)
    
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = np.cumsum(weight)[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]
    
    
def precision_recall_curve(y_true, y_pred, pos_label=None,
                           sample_weight=None,pi0=None, *args, **kwargs):
    """Compute precision-recall (with optional calibration) pairs for different probability thresholds
    This implementation is a modification of scikit-learn "precision_recall_curve" function that adds calibration
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    calib_precision : array, shape = [n_thresholds + 1]
        Calibrated Precision values such that element i is the calibrated precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    
   
    
    
    if pi0 is not None:
        pi = np.sum(y_true)/float(np.array(y_true).shape[0])
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tps / (tps + ratio*fps)
    else:
        precision = tps / (tps + fps)
    
    precision[np.isnan(precision)] = 0
        
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score_calibrated(y_true, y_pred, pos_label=1, sample_weight=None,pi0=None, *args,**kwargs):
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, pi0=pi0, *args, **kwargs)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def metric_fairness_ova(
    labels,
    pred_probs,
    group,
    the_group,
    sample_weight=None,
    metric_fn=None,
    transform_func=identity,
):
    """
    Computes a fairness metric by a comparison of the metric computed over the_group with the metric computed over the marginal distribution
    """
    if metric_fn is None:
        raise ValueError("metric_fn must be provided")

    if sample_weight is None:
        result_group = metric_fn(
            labels[group == the_group], pred_probs[group == the_group]
        )
        result_marginal = metric_fn(labels, pred_probs)
        result = transform_func(result_group - result_marginal)
    else:
        result_group = metric_fn(
            labels[group == the_group],
            pred_probs[group == the_group],
            sample_weight=sample_weight[group == the_group],
        )

        result_marginal = metric_fn(labels, pred_probs, sample_weight=sample_weight)
        result = transform_func(result_group - result_marginal)
    return result


def roc_auc_ova(*args, **kwargs):
    return metric_fairness_ova(*args, metric_fn=roc_auc_score, **kwargs)


def average_precision_ova(*args, **kwargs):
    return metric_fairness_ova(*args, metric_fn=average_precision_score, **kwargs)


def log_loss_ova(*args, **kwargs):
    return metric_fairness_ova(*args, metric_fn=log_loss, **kwargs)


def brier_ova(*args, **kwargs):
    return metric_fairness_ova(*args, metric_fn=brier_score_loss, **kwargs)


def mean_prediction_ova(*args, the_label=None, **kwargs):
    return metric_fairness_ova(
        *args,
        metric_fn=lambda *args1, **kwargs1: mean_prediction(
            *args1, the_label=the_label, **kwargs1
        ),
        **kwargs,
    )


def try_wasserstein_distance(*args, **kwargs):
    return try_metric_fn(
        *args, metric_fn=scipy.stats.wasserstein_distance, default_value=-1, **kwargs
    )


def try_roc_auc_score(*args, **kwargs):
    return try_metric_fn(*args, metric_fn=roc_auc_score, default_value=-1, **kwargs)


def try_log_loss(*args, **kwargs):
    return try_metric_fn(*args, metric_fn=log_loss, default_value=1e18, **kwargs)


def try_absolute_calibration_error(*args, **kwargs):
    return try_metric_fn(
        *args, **kwargs, default_value=1e18, metric_fn=absolute_calibration_error
    )


def emd_ova(labels, pred_probs, group, the_group, sample_weight=None, the_label=None):
    """
    Computes the earth movers distance between the pred_probs of the_group vs those of the marginal population
    Specifying the_label performs the computation stratified on the label
    """
    if the_label is not None:
        mask = labels == the_label
        labels = labels[mask]
        pred_probs = pred_probs[mask]
        group = group[mask]
        sample_weight = sample_weight[mask] if sample_weight is not None else None

    if sample_weight is None:
        return try_wasserstein_distance(
            u_values=pred_probs[group == the_group], v_values=pred_probs
        )
    else:
        return try_wasserstein_distance(
            u_values=pred_probs[group == the_group],
            v_values=pred_probs,
            u_weights=sample_weight[group == the_group],
            v_weights=sample_weight,
        )


def xauc(
    labels,
    pred_probs,
    group,
    the_group,
    sample_weight=None,
    the_label=1,
    exclude_the_group_from_marginal=False,
):
    """
    Computes the xAUC (http://arxiv.org/abs/1902.05826)
        - Computes the AUROC on a dataset composed of 
            - Data from the intersection of the_group & the_label
            - Data from (not the_label), excluding the_group based on exclude_the_group_from_marginal
    """

    other_label = 1 - the_label
    mask_group = (group == the_group) & (labels == the_label)
    mask_marginal = (
        (group != the_group) & (labels == other_label)
        if exclude_the_group_from_marginal
        else (labels == other_label)
    )

    if sample_weight is None:
        return try_roc_auc_score(
            np.concatenate((labels[mask_group], labels[mask_marginal])),
            np.concatenate((pred_probs[mask_group], pred_probs[mask_marginal])),
        )
    else:
        return try_roc_auc_score(
            np.concatenate((labels[mask_group], labels[mask_marginal])),
            np.concatenate((pred_probs[mask_group], pred_probs[mask_marginal])),
            sample_weight=np.concatenate(
                (sample_weight[mask_group], sample_weight[mask_marginal])
            ),
        )


# Functions that alias CalibrationEvaluator methods
def absolute_calibration_error(*args, **kwargs):
    evaluator = CalibrationEvaluator()
    return evaluator.absolute_calibration_error(*args, **kwargs)


def relative_calibration_error(*args, **kwargs):
    evaluator = CalibrationEvaluator()
    return evaluator.relative_calibration_error(*args, **kwargs)


def threshold_calibration_error(*args, **kwargs):
    evaluator = CalibrationEvaluator()
    return evaluator.threshold_calibration_error(*args, **kwargs)


def generate_threshold_calibration_error(
    threshold, metric_variant="abs", model_type="logistic", transform="logit"
):
    return lambda *args, **kwargs: threshold_calibration_error(
        *args,
        threshold=threshold,
        metric_variant=metric_variant,
        model_type=model_type,
        transform=transform,
        **kwargs,
    )
