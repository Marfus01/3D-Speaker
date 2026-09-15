"""Small CPU checks; run with python -m unittest speakerlab.process.hmm.test_face_hmm."""

import itertools
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speakerlab.process.hmm.face_hmm import FaceHMM

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs/3dspeaker/speaker-diarization/local"


class FaceHMMTests(unittest.TestCase):
    def test_viterbi_and_episode_boundaries(self):
        model = FaceHMM().fit(np.array([[0]] * 10 + [[1]] * 10), [10, 10])
        channel = model.models_[0]
        channel.startprob_ = np.array([0.8, 0.2])
        channel.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
        channel.emissionprob_ = np.array([[0.8, 0.2], [0.1, 0.9]])
        observations = np.array([1, 1, 0, 0, 1])
        expected = []
        for obs in (observations[:3], observations[3:]):
            def probability(states):
                value = channel.startprob_[states[0]] * channel.emissionprob_[states[0], obs[0]]
                for t in range(1, len(obs)):
                    value *= channel.transmat_[states[t - 1], states[t]]
                    value *= channel.emissionprob_[states[t], obs[t]]
                return value
            expected.extend(max(itertools.product([0, 1], repeat=len(obs)), key=probability))
        np.testing.assert_array_equal(model.predict(observations[:, None], [3, 2])[:, 0], expected)

    def test_constant_channels_and_validation(self):
        observations = np.array([[0, 1]] * 12, dtype=np.uint8)
        validated, _ = FaceHMM._validate(observations, [5, 7])
        self.assertTrue(np.shares_memory(validated, observations))
        model = FaceHMM().fit(observations, [5, 7])
        decoded = model.predict(observations, [5, 7])
        self.assertEqual(decoded.dtype, np.uint8)
        np.testing.assert_array_equal(decoded, observations)
        with self.assertRaises(ValueError):
            model.fit(observations, [11])
        with self.assertRaises(ValueError):
            model.fit(np.array([[2]]))

    def test_fit_is_deterministic_and_probabilities_valid(self):
        observations = np.tile(np.array([[0]] * 8 + [[1]] * 8), (4, 1))
        first = FaceHMM().fit(observations, [32, 32])
        second = FaceHMM().fit(observations, [32, 32])
        np.testing.assert_allclose(first.A_F_, second.A_F_)
        for matrix in (first.A_F_, first.B_F_):
            self.assertTrue(np.isfinite(matrix).all())
            np.testing.assert_allclose(matrix.sum(axis=-1), 1)
        self.assertGreaterEqual(first.B_F_[0, 1, 1], first.B_F_[0, 0, 1])

    def test_face_correction_uses_existing_helper(self):
        sys.path.insert(0, str(LOCAL))
        from cluster_and_postprocess import correct_face_labels
        # Correct one crop to cluster 1, one to Others; preserve an ambiguous frame.
        decoded = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]])
        observed = np.array([[1, 0, 0]] * 3)
        segments = np.array(["E01-0", "E01-1", "E01-2"])
        result = correct_face_labels(decoded, observed, segments, segments,
                                     np.array([0, 0, 0]), [[0, 1], [0, -1], [0, 1]])
        np.testing.assert_array_equal(result, [1, -1, 0])

    def test_batched_candidates_match_dense(self):
        from speakerlab.process import cluster
        rng = np.random.RandomState(42)
        labels = np.repeat([-1, 2, 7, 11], 8)
        for dtype in (np.float32, np.float64):
            features = rng.normal(size=(32, 8)).astype(dtype)
            # Include tied centroids and zero embeddings, preserving the original argsort rule.
            features[16:] = 0
            features.setflags(write=False)
            expected = cluster.align_samples2clusters(labels, features, 2)
            with patch.object(cluster, 'cosine_similarity', wraps=cluster.cosine_similarity) as similarity:
                actual = cluster.align_samples2clusters(labels, features, 2, batch_size=3)
                self.assertTrue(all(len(call.args[0]) <= 3 for call in similarity.call_args_list))
            self.assertEqual(actual, expected)

    def test_ahc_negation_reuses_similarity_matrix(self):
        from speakerlab.process import cluster
        features = np.random.RandomState(42).normal(size=(12, 5)).astype(np.float32)
        matrix = cluster.cosine_similarity(features)
        expected = cluster.squareform(-matrix, checks=False)
        squareform = cluster.squareform
        def check_squareform(values, **kwargs):
            self.assertIs(values, matrix)
            condensed = squareform(values, **kwargs)
            np.testing.assert_array_equal(condensed, expected)
            return condensed
        with patch.object(cluster, 'cosine_similarity', return_value=matrix), \
                patch.object(cluster, 'squareform', side_effect=check_squareform):
            cluster.AHCluster(fix_cos_thr=0.15)(features)

    def test_visual_pipeline_and_group_metrics(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embs_video"
            embeddings.mkdir()
            timeline, cached, rows = {}, {}, []
            for record in ["E01", "E02"]:
                ids = [f"{record}-{i}" for i in range(12)]
                timeline.update({key: {"start": i, "stop": i + 1, "file": f"/{record}.wav"}
                                 for i, key in enumerate(ids)})
                # The final frame has no face; its zero observation must remain.
                features = np.array([[1., 0.] if i < 6 else [0., 1.] for i in range(11)])
                with open(embeddings / f"{record}_midframe.pkl", "wb") as stream:
                    pickle.dump(dict(feat=features, audio_seg_id=ids[:-1], face_idx=[0] * 11), stream)
                for i, key in enumerate(ids[:-1]):
                    cached[f"{key}_0"] = 10 if i < 6 else 30
                    rows.append({"audio_seg_id": key, "face index": 0,
                                 "face label": "Sheldon" if i < 6 else "Leonard",
                                 "x1": 0, "y1": 0, "x2": 100, "y2": 50 + 100 * (i % 5)})
            (root / "subseg.json").write_text(json.dumps(timeline))
            (root / "wav.list").write_text("/nonexistent/E01.wav\n/nonexistent/E02.wav\n")
            (root / "ahc.json").write_text(json.dumps(cached))
            pd.DataFrame(rows).to_excel(root / "annotation.xlsx", index=False)
            output = root / "result"
            common = [sys.executable, str(LOCAL / "cluster_and_postprocess_face.py"),
                      "--conf", str(ROOT / "egs/3dspeaker/speaker-diarization/conf/the big bang theory/diar_video.yaml"),
                      "--wavs", str(root / "wav.list"), "--visual_embs_dir", str(embeddings),
                      "--subseg_json", str(root / "subseg.json"), "--result_dir", str(output)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            def execute(command):
                process = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                return process.stdout
            execute(common + ["--initial_labels", str(root / "ahc.json")])
            self.assertIn("Reusing AHC labels", execute(common))
            self.assertEqual(json.loads((output / "faces_mid_frame_ahc.json").read_text()), cached)
            self.assertEqual(set(json.loads((output / "faces_mid_frame_face_hmm.json").read_text())), set(cached))
            info = json.loads((output / "run_info.json").read_text())
            self.assertEqual(info["lengths"], [12, 12])
            execute([sys.executable, str(LOCAL / "compute_acc_face.py"), "--result_dir", str(output),
                     "--ref_xlsx", str(root / "annotation.xlsx")])
            for name in ["ahc", "face_hmm"]:
                metrics = (output / f"faces_mid_frame_{name}_accuracy.txt").read_text()
                for key in ["overall_accuracy"] + [f"group_{i}_accuracy" for i in range(5)]:
                    self.assertIn(key, metrics)
            # Also exercise the unchanged AHC configuration on the same cached embeddings.
            (output / "faces_mid_frame_ahc.json").unlink()
            self.assertIn("AHC: fix_cos_thr_mf=0.15", execute(common))
            self.assertIsNone(json.loads((output / "run_info.json").read_text())["ahc_source"])
            # Run both submission scripts locally with synthetic inputs and no Slurm/GPU.
            recipe = LOCAL.parent
            for script, tv_name in [("run_BB.sh", "the big bang theory"), ("run_IL.sh", "I love my family")]:
                data = root / tv_name
                (data / "raw").mkdir(parents=True)
                (data / "annotation").mkdir()
                (data / "raw/wav.list").write_text((root / "wav.list").read_text())
                (data / "annotation/faces_annotation_with_loc_new.xlsx").write_bytes(
                    (root / "annotation.xlsx").read_bytes())
                env.update(RECIPE_ROOT=str(recipe), DATA_ROOT=str(root),
                           VISUAL_EMBS_DIR=str(embeddings), SUBSEG_JSON=str(root / "subseg.json"),
                           AHC_LABELS=str(root / "ahc.json"), RESULT_DIR=str(root / script),
                           PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
                execute(["bash", str(recipe / "face_baseline" / script)])
                self.assertTrue((root / script / "faces_mid_frame_face_hmm_accuracy.txt").exists())
            (embeddings / "E02_midframe.pkl").unlink()
            failed = subprocess.run(common, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("FileNotFoundError", failed.stderr)


if __name__ == "__main__":
    unittest.main()
