"""Role 4's shared evaluation toolkit. Import from here: `from common.evalkit import expected_cost, ...`."""

from common.evalkit import ledger  # noqa: F401
from common.evalkit.bootstrap import block_bootstrap_ci, paired_bootstrap_diff  # noqa: F401
from common.evalkit.calibration import brier, ece, reliability_table  # noqa: F401
from common.evalkit.costs import (  # noqa: F401
    CLASSES, POSITIVE, bayes_decision, choose_threshold, cost_vs_threshold, decide_with_threshold,
    expected_cost, load_cost_matrix, load_cost_policy, scaled_cost_matrix,
)
from common.evalkit.forecast import (  # noqa: F401
    block_skill_ci, coverage_and_width, diebold_mariano, exceedance_probability, interval_score, mae, pinball_loss,
    rmse, skill_score, smape,
)
from common.evalkit.leadtime import (  # noqa: F401
    episode_hours, episodes_from_series, lead_time, lead_time_by_node, wilson_interval,
)
from common.evalkit.metrics import (  # noqa: F401
    average_precision, balanced_accuracy, confusion_counts, log_loss, macro_f1, multiclass_brier,
    per_class_prf, roc_auc,
)
