"""
Tests for the label policy and for scoring across label spaces and feature versions.

The label policy changes the question the model answers, and the feature
version changes what it is given. Both failure modes are silent - a
mis-projected column or a model fed the wrong feature vector still produces
plausible numbers - so the refusal paths are tested as carefully as the
happy ones.
"""
import json
import os
import pickle

import numpy as np
import pytest

from mlops.labels import (
    DROPPED_TAGS,
    MERGED_TAGS,
    THRESHOLD,
    apply_label_policy,
    canonical,
    predicted,
    project_probabilities,
    suppress_redundant,
)


class TestThreshold:
    def test_the_decided_threshold(self):
        assert THRESHOLD == 0.26

    @pytest.mark.parametrize('p,shown_as', [(0.2551, '0.26'), (0.26, '0.26'), (0.2649, '0.26'), (0.9, '0.90')])
    def test_a_tag_shown_at_the_threshold_is_predicted(self, p, shown_as):
        """What the player reads as 0.26 must count at a 0.26 threshold."""
        assert '%.2f' % p == shown_as
        assert predicted(p)

    @pytest.mark.parametrize('p', [0.2549, 0.25, 0.1])
    def test_a_tag_shown_below_the_threshold_is_not(self, p):
        assert '%.2f' % p < '0.26'
        assert not predicted(p)

    def test_works_on_arrays(self):
        np.testing.assert_array_equal(predicted(np.array([[0.2549, 0.2551], [0.3, 0.0]])),
                                      [[False, True], [True, False]])


class TestRedundancy:
    def test_jumps_hidden_next_to_a_specific_jump_tag(self):
        assert suppress_redundant(['aim', 'jumps', 'large jumps']) == ['aim', 'large jumps']
        assert suppress_redundant(['jumps', 'short jumps']) == ['short jumps']

    def test_jumps_kept_on_its_own(self):
        assert suppress_redundant(['aim', 'jumps']) == ['aim', 'jumps']

    def test_high_spacing_hidden_next_to_large_or_cross_screen_jumps(self):
        assert suppress_redundant(['high spacing', 'large jumps']) == ['large jumps']
        assert suppress_redundant(['cross screen jumps', 'high spacing']) == ['cross screen jumps']
        assert suppress_redundant(['high spacing', 'spaced streams']) == ['high spacing', 'spaced streams']

    def test_order_is_preserved(self):
        assert suppress_redundant(['large jumps', 'aim', 'jumps', 'streams']) == ['large jumps', 'aim', 'streams']


class TestPolicy:
    def test_merges_and_deduplicates(self):
        assert apply_label_policy(['alt', 'alternating', 'flow', 'jumps']) == \
            ['alternating', 'flow aim', 'jumps']

    def test_drops_non_skill_tags(self):
        assert apply_label_policy(['fast', 'dt speed', 'practise', 'streams']) == ['streams']

    def test_a_map_can_end_up_with_no_labels(self):
        """Its row is kept upstream; the policy just reports nothing left."""
        assert apply_label_policy(['comfortable']) == []

    def test_scrape_time_merges_live_here_too(self):
        assert canonical('linear patterns') == 'linear aim'
        assert canonical('star jumps') == canonical('triangle jumps') == 'geometric'

    def test_the_decided_policy(self):
        """Pinned: changing it moves every score in the registry (see promote.py)."""
        assert DROPPED_TAGS == {'progressive difficulty', 'practise', 'comfortable',
                                'dt speed', 'fast'}
        assert {k: MERGED_TAGS[k] for k in ('alt', 'snap', 'flow')} == \
            {'alt': 'alternating', 'snap': 'snap aim', 'flow': 'flow aim'}


class TestProjection:
    OLD = ['alt', 'alternating', 'fast', 'flow', 'flow aim', 'jumps']
    NEW = ['alternating', 'flow aim', 'jumps']

    def test_identity_when_label_spaces_match(self):
        proj = project_probabilities(self.NEW, self.NEW)
        assert proj.identity
        probs = np.array([[0.1, 0.2, 0.3]])
        np.testing.assert_array_equal(proj.apply(probs), probs)

    def test_drops_and_merges_with_max(self):
        proj = project_probabilities(self.OLD, self.NEW)
        probs = np.array([[0.30, 0.10, 0.90, 0.05, 0.40, 0.70]])
        np.testing.assert_allclose(proj.apply(probs), [[0.30, 0.40, 0.70]])
        assert proj.dropped == ['fast']
        assert set(proj.merged) == {'alternating', 'flow aim'}

    def test_refuses_a_label_the_policy_does_not_know(self):
        with pytest.raises(ValueError, match='does not know'):
            project_probabilities(self.OLD + ['mystery tag'], self.NEW)

    def test_refuses_when_the_model_cannot_produce_a_target_label(self):
        with pytest.raises(ValueError, match='cannot produce'):
            project_probabilities(['alternating', 'jumps'], self.NEW)


@pytest.fixture(scope='module')
def tiny_ensemble_factory(tmp_path_factory):
    """
    Build a real (tiny) ensemble directory: one Keras model, scaler, binarizer.
    Real artifacts rather than mocks, because the point is to exercise the same
    load path the gate uses.
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

    def make(classes, n_features, feature_version=None):
        d = tmp_path_factory.mktemp('ens')
        model = tf.keras.Sequential([tf.keras.Input(shape=(n_features,)),
                                     tf.keras.layers.Dense(len(classes), activation='sigmoid')])
        model.save(str(d / 'ensemble_model_1.keras'))
        scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(20, n_features)))
        binarizer = MultiLabelBinarizer().fit([classes])
        with open(d / 'ensemble_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(d / 'ensemble_binarizer.pkl', 'wb') as f:
            pickle.dump(binarizer, f)
        if feature_version is not None:
            from mlops.split import FEATURE_META_NAME
            with open(d / FEATURE_META_NAME, 'w') as f:
                json.dump({'feature_version': feature_version}, f)
        return str(d)
    return make


def prepared_for(classes, n_features, feature_version=1, n=10):
    from mlops.split import Prepared, Split
    rng = np.random.default_rng(1)
    return (
        Prepared(X=rng.normal(size=(n, n_features)),
                 y=rng.integers(0, 2, size=(n, len(classes))),
                 classes=list(classes), ids=[str(i) for i in range(n)], tag_lists=[],
                 dataset_sha='x', source_path='x', feature_version=feature_version),
        Split(train_idx=np.arange(5), test_idx=np.arange(5, n), holdout_ids=[], split_hash='h'))


class TestScalerGuard:
    def test_an_overflowing_feature_stops_training(self):
        """
        A 5e301 BPM once overflowed the scaler; every model then learned a
        constant output and it looked like a merely bad model. It must raise.
        """
        from mlops.split import scale_all
        prepared, _ = prepared_for(['jumps', 'streams'], 72, feature_version=2)
        prepared.X[0, 5] = 5.45e301
        with pytest.raises(ValueError, match='bpm_range_ratio'):
            scale_all(prepared)

    def test_normal_features_scale(self):
        from mlops.split import scale_all
        prepared, _ = prepared_for(['jumps', 'streams'], 72, feature_version=2)
        X_scaled, _ = scale_all(prepared)
        assert np.isfinite(X_scaled).all()


class TestScoringAcrossVersions:
    def test_an_old_label_space_is_projected_onto_the_current_one(self, tiny_ensemble_factory):
        from mlops.scoring import score_on_holdout
        model_dir = tiny_ensemble_factory(TestProjection.OLD, 4)
        prepared, split = prepared_for(TestProjection.NEW, 4)
        summary, per_tag, probs = score_on_holdout(model_dir, prepared, split)
        assert probs.shape == (5, len(TestProjection.NEW))
        assert 'merged' in summary['label_projection']

    def test_an_unexplained_label_mismatch_is_still_refused(self, tiny_ensemble_factory):
        from mlops.scoring import score_on_holdout
        model_dir = tiny_ensemble_factory(['jumps', 'mystery tag'], 4)
        prepared, split = prepared_for(['jumps'], 4)
        with pytest.raises(ValueError, match='Label space mismatch'):
            score_on_holdout(model_dir, prepared, split)

    def test_a_directory_without_feature_meta_is_v1(self, tiny_ensemble_factory):
        from mlops.split import model_feature_version
        assert model_feature_version(tiny_ensemble_factory(['jumps'], 4)) == 1

    def test_a_model_is_never_scored_on_another_versions_features(self, tiny_ensemble_factory):
        """Same width, different extractor: only the recorded version can catch it."""
        from mlops.scoring import score_on_holdout
        model_dir = tiny_ensemble_factory(['jumps'], 4, feature_version=2)
        prepared, split = prepared_for(['jumps'], 4, feature_version=1)
        with pytest.raises(ValueError, match='Feature version mismatch'):
            score_on_holdout(model_dir, prepared, split)

    def test_matching_versions_score(self, tiny_ensemble_factory):
        from mlops.scoring import score_on_holdout
        model_dir = tiny_ensemble_factory(['jumps', 'streams'], 4, feature_version=2)
        prepared, split = prepared_for(['jumps', 'streams'], 4, feature_version=2)
        summary, _, _ = score_on_holdout(model_dir, prepared, split)
        assert summary['label_projection'] == 'identity'
