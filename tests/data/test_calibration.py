"""Tests for common/dataio/calibration.py: the real drivers line up with the simulator's hours and nodes."""

import numpy as np
import pandas as pd
import pytest

from common.dataio import simulator as sim
from common.dataio.calibration import CAL_DIR, real_drivers

pytestmark = pytest.mark.skipif(not (CAL_DIR / "weather_hourly.csv.gz").exists(), reason="calibration files not present")


@pytest.fixture(scope="module")
def drivers():
    nodes, _ = sim.load_network()
    hours = sim.SIM_START + pd.to_timedelta(np.arange(sim.SIM_HOURS + sim.EXTRA_HOURS), unit="h")
    return real_drivers(hours, list(nodes["node_id"])), hours, nodes


def test_real_weather_covers_every_simulated_hour_and_node(drivers):
    real, hours, nodes = drivers
    assert real["weather"].shape == (len(hours), len(nodes))
    assert not np.isnan(real["weather"]).any()
    assert real["weather"].min() >= 0 and real["weather"].max() <= 10


def test_nodes_without_portwatch_data_get_neutral_factors(drivers):
    real, _, nodes = drivers
    no_profile = ~real["has_profile"]
    # Rail hub and distribution hubs are not in PortWatch: no seasonal push, average size.
    assert set(nodes.loc[no_profile, "node_type"]) <= {"RAIL_HUB", "DC"}
    assert (real["month_factor"][:, no_profile] == 1.0).all() and (real["relative_size"][no_profile] == 1.0).all()


def test_gscpi_is_the_real_monthly_value(drivers):
    real, hours, _ = drivers
    g = pd.read_csv(CAL_DIR / "gscpi_monthly.csv").set_index("month")["gscpi"]
    first_hour_of_dec = int(np.flatnonzero(hours.strftime("%Y-%m") == "2025-12")[0])
    assert real["gscpi"][first_hour_of_dec] == pytest.approx(g["2025-12"])
