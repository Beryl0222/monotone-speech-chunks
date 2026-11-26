import json
import unittest
from pathlib import Path

import numpy as np

from monotone_chunks.boundary import boundary_distribution
from monotone_chunks.constraints import build_constraint_matrix, geometric_feasibility
from monotone_chunks.reader import StreamingReader
from monotone_chunks.replay import replay_events
from monotone_chunks.sentence import SentenceInputs
from monotone_chunks.training import train_forward_backward
from monotone_chunks.batch import train_tensor_batch

ROOT = Path(__file__).resolve().parents[1]


class CoreProbabilityTests(unittest.TestCase):
    def test_dp_matches_exhaustive_non_decreasing_paths(self):
        rng = np.random.default_rng(7)
        w, t = 3, 5
        e = rng.normal(size=(w, t))
        c = rng.uniform(0.4, 1.0, (w, t))
        c[0, 0] = 0.0
        c[1, 4] = 0.0
        alpha = boundary_distribution(e, c).alpha

        brute = np.zeros_like(alpha)
        for i in range(w):
            import itertools

            for bounds in itertools.combinations_with_replacement(range(t), i + 1):
                mass = 1.0
                previous = 0
                p = 1.0 / (1.0 + np.exp(-e))
                for wi, j in enumerate(bounds):
                    for frame in range(previous, j):
                        mass *= 1.0 - p[wi, frame]
                    mass *= p[wi, j] * c[wi, j]
                    previous = j
                brute[i, bounds[-1]] += mass
        np.testing.assert_allclose(alpha, brute, atol=1e-12)

    def test_block_end_wait_is_not_given_to_last_frame(self):
        # All p are small enough that the final frame cannot absorb the mass.
        e = np.full((1, 4), -20.0)
        partial = boundary_distribution(e, stop=4)
        self.assertGreater(partial.wait, 0.99)
        self.assertLess(partial.alpha[0, -1], 1e-8)

        forced = boundary_distribution(e, force_end=True)
        np.testing.assert_allclose(forced.alpha[0, :3], 0.0, atol=1e-7)
        np.testing.assert_allclose(forced.alpha[0, 3], 1.0, atol=1e-12)
        self.assertEqual(float(np.asarray(forced.wait).sum()), 0.0)

    def test_invalid_padding_does_not_become_boundary(self):
        e = np.full((1, 4), 2.0)
        valid = np.array([True, True, True, False])
        out = boundary_distribution(e, valid=valid, force_end=True)
        self.assertEqual(out.n_frames, 3)
        self.assertLess(out.alpha[0, 2], 1.0)  # first crossing is earlier, not forced mass only

    def test_overlapping_constraint_factors_multiply(self):
        records = [
            {"word_index": 0, "lo": 1, "hi": 3, "outside_factor": 0.2},
            {"word_index": 0, "lo": 2, "hi": 4, "outside_factor": 0.5},
        ]
        c = build_constraint_matrix(1, 6, records)
        # frame 0 outside both; frame 1 in first only; frame 2/3 in both;
        # frame 4 in second only; frame 5 outside both.
        np.testing.assert_allclose(c[0], [0.1, 0.5, 1.0, 1.0, 0.2, 0.1])

    def test_separate_word_windows_do_not_imply_sentence_feasibility(self):
        records = [
            {"word_index": 0, "lo": 5, "hi": 7, "outside_factor": 0.0},
            {"word_index": 1, "lo": 2, "hi": 4, "outside_factor": 0.0},
        ]
        c = build_constraint_matrix(2, 10, records)
        geo = geometric_feasibility(c, 10)
        self.assertTrue(geo.hard_allowed[0].any())
        self.assertTrue(geo.hard_allowed[1].any())
        self.assertFalse(geo.feasible)
        self.assertEqual(geo.reason, "joint_monotone_constraints_infeasible")


class GradientTests(unittest.TestCase):
    def test_analytic_gradients_match_central_differences(self):
        rng = np.random.default_rng(11)
        w, t, d, dv, cw = 2, 6, 3, 2, 3
        q = rng.normal(scale=0.4, size=(w, d))
        ks = rng.normal(scale=0.4, size=(t, d))
        kc = rng.normal(scale=0.4, size=(t, d))
        v = rng.normal(scale=0.4, size=(t, dv))
        b = rng.normal(scale=0.1, size=w)
        e = q @ ks.T + b[:, None]
        u = q @ kc.T / np.sqrt(d)
        upstream = rng.normal(size=(w, dv))
        c = np.ones((w, t))
        c[0, :2] = 0.3

        def loss(ee, uu):
            return train_forward_backward(
                e=ee, u=uu, V=v, chunk_width=cw, constraints=c, context_upstream=upstream
            ).loss

        result = train_forward_backward(
            e=e, u=u, V=v, chunk_width=cw, constraints=c, context_upstream=upstream
        )
        h = 1e-6
        for name, arr, grad in [("e", e, result.grad_e), ("u", u, result.grad_u), ("V", v, result.grad_V)]:
            for idx in np.ndindex(arr.shape):
                old = arr[idx]
                arr[idx] = old + h
                plus = loss(e, u)
                arr[idx] = old - h
                minus = loss(e, u)
                arr[idx] = old
                numeric = (plus - minus) / (2 * h)
                self.assertLess(abs(numeric - grad[idx]), 3e-4 * max(1.0, abs(numeric)), (name, idx))


class BatchIsolationTests(unittest.TestCase):
    def test_window_contexts_do_not_cross_batch(self):
        e = np.zeros((2, 1, 3))
        u = np.zeros((2, 1, 3))
        v = np.zeros((2, 3, 2))
        v[1] = 50.0
        result = train_tensor_batch(e, u, v, chunk_width=2)
        ctx0 = result.reports[0].context[0]
        ctx1 = result.reports[1].context[0]
        np.testing.assert_allclose(ctx0, 0.0, atol=1e-12)
        self.assertGreater(np.linalg.norm(ctx1), 1.0)
        self.assertEqual(result.eligible_weight_denominator, 2.0)


class StreamingEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.materials = json.loads((ROOT / "data/materials.json").read_text())
        cls.events = json.loads((ROOT / "data/events.json").read_text())

    def test_supplied_event_story(self):
        report = replay_events(self.materials, self.events)
        by_op = []
        for step in report.steps:
            if step.event["op"] in {"online_propose", "commit", "train_query"}:
                by_op.append(step.output)

        first_proposal = report.steps[2].output
        self.assertEqual(first_proposal.status, "provisional")
        self.assertEqual(first_proposal.boundary, 6)
        self.assertTrue(first_proposal.temporary)

        withdrawal = report.steps[3].output
        self.assertTrue(withdrawal.accepted)
        self.assertEqual(withdrawal.last_committed_boundary, 0)

        bad_train = report.steps[5].output
        self.assertFalse(bad_train.eligible)
        self.assertEqual(bad_train.status, "constraints_infeasible")

        final_proposal = report.steps[9].output
        self.assertEqual(final_proposal.status, "committable")
        self.assertEqual(final_proposal.boundary, 6)

        commit = report.steps[10].output
        self.assertTrue(commit.accepted)
        self.assertEqual(commit.boundary, 6)

        second = report.steps[11].output
        self.assertEqual(second.boundary, 6)
        self.assertTrue(second.committable)

        good_train = report.steps[12].output
        self.assertTrue(good_train.eligible)
        self.assertTrue(np.isfinite(good_train.loss))
        self.assertEqual(report.final.last_committed_boundary, 6)

    def test_withdrawn_preview_does_not_move_cursor(self):
        reader = StreamingReader.from_materials(self.materials, constraints=[])
        reader.audio_arrive(0, 8, 4)
        proposal = reader.online_propose("w0")
        self.assertEqual(proposal.boundary, 6)
        reader.withdraw_uncommitted("w0")
        self.assertEqual(reader.last_committed_boundary, 0)
        self.assertNotIn(0, reader.provisional)
        # A fresh reader decision starts at the formal frame-zero cursor.
        again = reader.online_propose("w0")
        self.assertEqual(again.start_boundary, 0)

    def test_committed_word_cannot_be_withdrawn_or_overwritten(self):
        s = SentenceInputs.from_materials(self.materials, load_constraints=False)
        reader = StreamingReader(s, constraints=[])
        reader.audio_arrive(0, 18, 17)
        reader.seal_utterance(16)
        proposal = reader.online_propose("w0")
        self.assertTrue(proposal.committable)
        receipt = reader.commit("w0")
        self.assertTrue(receipt.accepted)
        blocked = reader.withdraw_uncommitted("w0")
        self.assertFalse(blocked.accepted)
        self.assertEqual(reader.subtitles[0].boundary, 6)


class LatencyAndFeedingTests(unittest.TestCase):
    def test_tau_changes_first_threshold_boundary(self):
        p = np.array([[0.1, 0.8, 0.2, 0.95, 0.99]])
        u = np.zeros((1, 5))
        v = np.ones((5, 2))
        reader = StreamingReader(
            word_ids=["w"], chunk_width=2, p=p, u=u, V=v, tau=0.5, max_lookahead=5
        )
        reader.push_chunk({"start": 0, "stop": 5, "confirmed_stop": 5})
        self.assertEqual(reader.online_propose(tau=0.5).boundary, 1)
        self.assertEqual(reader.online_propose(tau=0.9).boundary, 3)

    def test_max_lookahead_limits_preview_to_acquisition_end_distance(self):
        p = np.array([[0.1, 0.99, 0.2, 0.3, 0.95]])
        reader = StreamingReader(
            word_ids=["w"],
            chunk_width=1,
            p=p,
            u=np.zeros((1, 5)),
            V=np.ones((5, 1)),
            tau=0.5,
            max_lookahead=1,
            lookahead_reference="acquired_end",
        )
        reader.audio_arrive(0, 5, 0)
        preview = reader.online_propose()
        self.assertTrue(preview.temporary)
        self.assertEqual(preview.boundary, 4)

        wide = StreamingReader(
            word_ids=["w"],
            chunk_width=1,
            p=p,
            u=np.zeros((1, 5)),
            V=np.ones((5, 1)),
            tau=0.5,
            max_lookahead=5,
            lookahead_reference="acquired_end",
        )
        wide.audio_arrive(0, 5, 0)
        self.assertEqual(wide.online_propose().boundary, 1)

    def test_sequential_chunk_feed_keeps_formal_cursor(self):
        p = np.array([[0.1, 0.2, 0.9, 0.1, 0.8]])
        reader = StreamingReader(
            word_ids=["w"], chunk_width=1, p=p, u=np.zeros((1, 5)), V=np.ones((5, 1)), tau=0.5
        )
        reader.audio_arrive(0, 2, 2)
        self.assertIsNone(reader.online_propose().boundary)
        reader.audio_arrive(2, 4, 4)
        proposal = reader.online_propose()
        self.assertEqual(proposal.boundary, 2)
        self.assertTrue(proposal.committable)
        self.assertTrue(reader.commit().accepted)
        self.assertEqual(reader.last_committed_boundary, 2)


if __name__ == "__main__":
    unittest.main()
