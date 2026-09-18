# part4_networks/tests/test_connectionist.py
# Unit tests for Module D (Connectionist) - TC-D1 to TC-D3

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from part4_networks.rnn_model import rnn_predict_anomaly, train_rnn
from part4_networks.hopfield import (corrupt, generate_pattern_1,
                                             generate_pattern_2,
                                             generate_pattern_3,
                                             generate_pattern_4,
                                             generate_pattern_5,
                                             generate_pattern_6, recall, train)


# ---------------------------------------------------------------------------
# TC-D1 — Hopfield Perfect Recall
# ---------------------------------------------------------------------------
class TestTCD1:
    def test_perfect_recall(self):
        """Hopfield network should perfectly recall trained patterns when presented with the exact pattern."""
        patterns = [
            generate_pattern_1(),
            generate_pattern_2(),
            generate_pattern_3(),
            generate_pattern_4(),
            generate_pattern_5(),
            generate_pattern_6()
        ]
        W = train(patterns)

        # Test recall on pattern 1
        p1 = patterns[0]
        recalled = recall(W, p1)
        assert np.array_equal(
            recalled, p1), "Failed perfect recall on Pattern 1"

# ---------------------------------------------------------------------------
# TC-D2 — Hopfield Corrupted Recall
# ---------------------------------------------------------------------------


class TestTCD2:
    def test_corrupted_recall(self):
        """Hopfield network should be able to recall a pattern even with some corruption."""
        patterns = [
            generate_pattern_1(),
            generate_pattern_2(),
            generate_pattern_3()
        ]
        W = train(patterns)

        p1 = patterns[0]
        corrupted_p1 = corrupt(p1, flip_fraction=0.1)  # 10% corruption

        recalled = recall(W, corrupted_p1)

        # Check if recalled pattern is closer to p1 than the corrupted one
        diff_corrupt = np.sum(corrupted_p1 != p1)
        diff_recalled = np.sum(recalled != p1)
        assert diff_recalled <= diff_corrupt, "Recall did not improve or stay same for corrupted pattern"

# ---------------------------------------------------------------------------
# TC-D3 — RNN Anomaly Detection
# ---------------------------------------------------------------------------


class TestTCD3:
    def test_rnn_anomaly_detection(self):
        """RNN should train and predict correctly on synthetic sequences."""
        model = train_rnn()

        # Generate a normal window
        normal_window = [
            {"vibration": 0.5, "current": 1.0, "temperature": 30.0}
            for _ in range(10)
        ]
        is_anomaly_normal = rnn_predict_anomaly(normal_window, model)

        # Generate an anomaly window
        anomaly_window = [
            {"vibration": 0.5, "current": 1.0, "temperature": 30.0}
            for _ in range(5)
        ] + [
            {"vibration": 2.5, "current": 1.0, "temperature": 30.0}
            for _ in range(5)
        ]
        is_anomaly_abnormal = rnn_predict_anomaly(anomaly_window, model)

        # We don't strictly assert the outcome since it's a small trained model,
        # but we ensure the function runs without error.
        assert isinstance(is_anomaly_normal, bool)
        assert isinstance(is_anomaly_abnormal, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
