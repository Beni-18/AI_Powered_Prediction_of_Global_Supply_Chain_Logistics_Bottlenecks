# Visual prediction benchmark (test)

Baseline ML detection (`xgb_lag`; `stage1_risk` = Stage 1 champion's shipment P(critical) averaged per node, at its frozen threshold) vs the DL forecasters, same episodes, same alert scoring.

| model | rule | n_episodes | n_caught | episode_recall | median_lead_h | false_alarms_per_node_week | warning_cost |
|---|---|---|---|---|---|---|---|
| graph_full | B_prob_ge_0.40 | 13 | 12 | 0.9231 | 27.5 | 0.3664 | 63.0 |
| lstm | B_prob_ge_0.55 | 13 | 9 | 0.6923 | 23.0 | 0.0968 | 54.0 |
| persistence | B_prob_ge_0.90 | 13 | 5 | 0.3846 | 56.0 | 0.083 | 92.0 |
| transformer | B_prob_ge_0.70 | 13 | 9 | 0.6923 | 32.0 | 0.0622 | 49.0 |
| xgb_lag | B_prob_ge_0.75 | 13 | 4 | 0.3077 | 23.5 | 0.0553 | 98.0 |
| stage1_risk | stage1_threshold | 13 | 8 | 0.6154 | 108.5 | 2.4267 | 401.0 |

Episodes in the figure: [{"node_id": "PORT_US_NYC", "onset": "2025-12-22T19:00:00+00:00", "peak_vs_threshold": 2.123, "first_alert_lead_h": {"graph_full": 1.0, "stage1_risk": 73.0}}, {"node_id": "CHOKE_HORMUZ", "onset": "2025-12-05T06:00:00+00:00", "peak_vs_threshold": 1.428, "first_alert_lead_h": {"persistence": 66.0, "xgb_lag": 36.0, "lstm": 36.0, "graph_full": 72.0, "transformer": 36.0, "stage1_risk": 144.0}}, {"node_id": "PORT_NL_RTM", "onset": "2025-12-07T23:00:00+00:00", "peak_vs_threshold": 1.312, "first_alert_lead_h": {"xgb_lag": 11.0, "lstm": 23.0, "graph_full": 23.0, "transformer": 11.0}}]
