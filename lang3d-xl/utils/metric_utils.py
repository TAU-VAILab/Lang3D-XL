import numpy as np
from functools import partial

from sklearn.utils.multiclass import type_of_target
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics._base import _check_pos_label_consistency

import torch
from torch.nn import functional as F


def average_precision_score(
    y_true, y_score, dilated_y_true, *, average="macro", pos_label=1, sample_weight=None
):
    """Compute average precision (AP) from prediction scores.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    average : {'micro', 'samples', 'weighted', 'macro'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    pos_label : int or str, default=1
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    average_precision : float

    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.

    Notes
    -----
    .. versionchanged:: 0.19
      Instead of linearly interpolating between operating points, precisions
      are weighted by the change in recall since the last operating point.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)
    0.83...
    """

    def _binary_uninterpolated_average_precision(
        y_true, y_score, dilated_y_true, pos_label=1, sample_weight=None
    ):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, dilated_y_true, pos_label=pos_label, sample_weight=sample_weight
        )
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError(
            "Parameter pos_label is fixed to 1 for "
            "multilabel-indicator y_true. Do not set "
            "pos_label or set pos_label to 1."
        )
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(
        _binary_uninterpolated_average_precision, pos_label=pos_label
    )
    return _average_binary_score(
        average_precision, y_true, y_score, dilated_y_true, average, sample_weight=sample_weight
    )

def precision_recall_curve(y_true, probas_pred, dilated_y_true, *, pos_label=None, sample_weight=None):
    """Compute precision-recall pairs for different probability thresholds.

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold. This ensures that the graph starts on the
    y axis.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    probas_pred : ndarray of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    precision : ndarray of shape (n_thresholds + 1,)
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : ndarray of shape (n_thresholds + 1,)
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : ndarray of shape (n_thresholds,)
        Increasing thresholds on the decision function used to compute
        precision and recall. n_thresholds <= len(np.unique(probas_pred)).

    See Also
    --------
    PrecisionRecallDisplay.from_estimator : Plot Precision Recall Curve given
        a binary classifier.
    PrecisionRecallDisplay.from_predictions : Plot Precision Recall Curve
        using predictions from a binary classifier.
    average_precision_score : Compute average precision from prediction scores.
    det_curve: Compute error rates for different probability thresholds.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision
    array([0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])

    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, dilated_y_true, pos_label=pos_label, sample_weight=sample_weight
    )

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

def _binary_clf_curve(y_true, y_score, dilated_y_true, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.

    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    pos_label : int or str, default=None
        The label of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    if dilated_y_true is not None:
        dilated_y_true = column_or_1d(dilated_y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    if dilated_y_true is not None:
        assert_all_finite(dilated_y_true)
    assert_all_finite(y_score)

    # Filter out zero-weighted samples, as they should not impact the result
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = _check_sample_weight(sample_weight, y_true)
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        if dilated_y_true is not None:
            dilated_y_true = dilated_y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]

    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    y_true = prep_y_true(y_true, pos_label, desc_score_indices)
    if dilated_y_true is not None:
        dilated_y_true = prep_y_true(dilated_y_true, pos_label, desc_score_indices)

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        if dilated_y_true is None:
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = stable_cumsum((1 - dilated_y_true) * weight)[threshold_idxs]
    else:
        if dilated_y_true is None:
            fps = 1 + threshold_idxs - tps
        else:
            dilated_tps = stable_cumsum(dilated_y_true * weight)[threshold_idxs]
            fps = 1 + threshold_idxs - dilated_tps
    return fps, tps, y_score[threshold_idxs]

def prep_y_true(y_true, pos_label, desc_score_indices):
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = y_true == pos_label

    y_true = y_true[desc_score_indices]
    return y_true

def _average_binary_score(binary_metric, y_true, y_score, dilated_y_true,
                          average, sample_weight=None):
    """Average a binary metric for multilabel classification.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels in binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    average : {None, 'micro', 'macro', 'samples', 'weighted'}, default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    binary_metric : callable, returns shape [n_classes]
        The binary metric function to use.

    Returns
    -------
    score : float or array of shape [n_classes]
        If not ``None``, average the score, else return the score for each
        classes.

    """
    average_options = (None, "micro", "macro", "weighted", "samples")
    if average not in average_options:
        raise ValueError("average has to be one of {0}".format(average_options))

    y_type = type_of_target(y_true)
    if y_type not in ("binary", "multilabel-indicator"):
        raise ValueError("{0} format is not supported".format(y_type))

    if y_type == "binary":
        return binary_metric(y_true, y_score, dilated_y_true,
                             sample_weight=sample_weight)

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = check_array(y_true)
    dilated_y_true = check_array(dilated_y_true)
    y_score = check_array(y_score)

    not_average_axis = 1
    score_weight = sample_weight
    average_weight = None

    if average == "micro":
        if score_weight is not None:
            score_weight = np.repeat(score_weight, y_true.shape[1])
        y_true = y_true.ravel()
        dilated_y_true = dilated_y_true.ravel()
        y_score = y_score.ravel()

    elif average == "weighted":
        if score_weight is not None:
            average_weight = np.sum(
                np.multiply(y_true, np.reshape(score_weight, (-1, 1))), axis=0
            )
        else:
            average_weight = np.sum(y_true, axis=0)
        if np.isclose(average_weight.sum(), 0.0):
            return 0

    elif average == "samples":
        # swap average_weight <-> score_weight
        average_weight = score_weight
        score_weight = None
        not_average_axis = 0

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))

    n_classes = y_score.shape[not_average_axis]
    score = np.zeros((n_classes,))
    for c in range(n_classes):
        y_true_c = y_true.take([c], axis=not_average_axis).ravel()
        y_score_c = y_score.take([c], axis=not_average_axis).ravel()
        score[c] = binary_metric(y_true_c, y_score_c, sample_weight=score_weight)

    # Average the results
    if average is not None:
        if average_weight is not None:
            # Scores with 0 weights are forced to be 0, preventing the average
            # score from being affected by 0-weighted NaN elements.
            average_weight = np.asarray(average_weight)
            score[average_weight == 0] = 0
        return np.average(score, weights=average_weight)
    else:
        return score

def compute_psnr(img1, img2, max_val=1.0, eps=1e-9):
    """
    Compute PSNR between two torch tensors of shape (C, H, W)

    Args:
        img1: torch.Tensor, shape (C, H, W)
        img2: torch.Tensor, shape (C, H, W)
        max_val: float, maximum possible pixel value (e.g., 1.0 if normalized, 255 if uint8 range)

    Returns:
        psnr: float
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / (torch.sqrt(mse) + eps))
    return psnr.item()

