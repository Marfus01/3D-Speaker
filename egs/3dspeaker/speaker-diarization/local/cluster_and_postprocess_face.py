"""AHC followed by independent face HMMs, using cached visual embeddings only."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from speakerlab.process.cluster import align_samples2clusters, reset_cluster_ids
from speakerlab.process.hmm.face_hmm import FaceHMM
from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config
from cluster_and_postprocess import correct_face_labels, save_cluster_results_vision_mf


def load_faces(wavs, visual_embs_dir, subseg_json):
    """Keep all subtitle time points, including frames with no detected face."""
    records = sorted(Path(line.strip()).stem for line in Path(wavs).read_text().splitlines()
                     if line.strip())
    with open(subseg_json) as stream:
        segments = json.load(stream)
    timeline, lengths, features, face_segments, face_indices = [], [], [], [], []
    for record in records:
        ids = [key for key in segments if key.rsplit("-", 1)[0] == record]
        ids.sort(key=lambda key: (segments[key]["start"], int(key.rsplit("-", 1)[1])))
        if not ids:
            raise ValueError(f"No subtitle segments for {record}.")
        timeline.extend(ids)
        lengths.append(len(ids))
        # Missing files are errors; an empty, existing cache is a valid episode.
        with open(Path(visual_embs_dir) / f"{record}_midframe.pkl", "rb") as stream:
            data = pickle.load(stream)
        if not (len(data["feat"]) == len(data["audio_seg_id"]) == len(data["face_idx"])):
            raise ValueError(f"Inconsistent face cache for {record}.")
        if not set(data["audio_seg_id"]).issubset(ids):
            raise ValueError(f"Face cache and subtitle segments disagree for {record}.")
        if len(data["feat"]):
            if not np.isfinite(data["feat"]).all():
                raise ValueError(f"Nonfinite embeddings for {record}.")
            features.append(data["feat"])
            face_segments.extend(data["audio_seg_id"])
            face_indices.extend(data["face_idx"])
        print(f"[Load] {record}: {len(data['feat'])} faces, {len(ids)} frames", flush=True)
        del data
    if not features:
        raise ValueError("No cached face embeddings found.")
    features = np.concatenate(features)
    keys = [f"{seg}_{int(idx)}" for seg, idx in zip(face_segments, face_indices)]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate face keys.")
    return features, np.array(face_segments), np.array(face_indices), np.array(timeline), lengths


def run(args):
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    print("[1/5] Loading cached visual embeddings", flush=True)
    features, face_segments, face_indices, timeline, lengths = load_faces(
        args.wavs, args.visual_embs_dir, args.subseg_json)
    print(f"[Load] shape={features.shape}, dtype={features.dtype}, "
          f"embeddings={features.nbytes / 2**30:.2f} GiB", flush=True)
    keys = [f"{seg}_{int(idx)}" for seg, idx in zip(face_segments, face_indices)]
    initial_path = result_dir / "faces_mid_frame_ahc.json"
    source = Path(args.initial_labels) if args.initial_labels else initial_path
    reused_ahc = bool(args.initial_labels) or source.exists()
    if reused_ahc:
        with open(source) as stream:
            cached = json.load(stream)
        if set(cached) != set(keys):
            raise ValueError("AHC result must cover exactly the loaded face keys.")
        if any(not isinstance(value, int) or value < -1 for value in cached.values()):
            raise ValueError("AHC labels must be integers >= -1 (Others).")
        labels = np.array([cached[key] for key in keys])
        del cached
        print(f"[2/5] Reusing AHC labels: {source}", flush=True)
    else:
        config = build_config(args.conf)
        config.vision_cluster["args"]["fix_cos_thr"] = config.fix_cos_thr_mf
        cluster = build("vision_cluster", config)
        print(f"[2/5] Starting AHC: fix_cos_thr_mf={config.fix_cos_thr_mf}; "
              f"N x N matrix alone ~{len(features)**2 * features.dtype.itemsize / 2**30:.2f} GiB "
              "(excluding temporary arrays and linkage workspace)", flush=True)
        labels = reset_cluster_ids(cluster(features))
    save_cluster_results_vision_mf(labels, face_segments, face_indices, initial_path)
    print(f"[2/5] AHC labels saved: {initial_path}", flush=True)

    # Preserve every AHC cluster; reserve the last channel for the helper's Others convention.
    cluster_ids = sorted(set(labels) - {-1})
    label_map = {label: j for j, label in enumerate(cluster_ids)}
    label_map[-1] = -1
    mapped = np.array([label_map[label] for label in labels])
    print(f"[3/5] Building presence matrix and candidates: "
          f"T={len(timeline)}, K={len(cluster_ids) + 1}", flush=True)
    F_hat = np.zeros((len(timeline), len(cluster_ids) + 1), dtype=np.uint8)
    row_map = {seg: row for row, seg in enumerate(timeline)}
    for seg, label in zip(face_segments, mapped):
        F_hat[row_map[seg], label] = 1
    candidates = align_samples2clusters(mapped, features, candi_align_cluster_num=args.candidates, batch_size=1024)
    del features
    print("[4/5] Fitting independent face HMMs", flush=True)
    model = FaceHMM(n_iter=args.n_iter, tol=args.tol, random_state=args.random_state)
    model.fit(F_hat, lengths)
    print("[4/5] Decoding face states", flush=True)
    F_decode = model.predict(F_hat, lengths)
    print("[5/5] Correcting face labels", flush=True)
    corrected = correct_face_labels(F_decode, F_hat, timeline, face_segments, mapped, candidates)
    reverse_map = dict(enumerate(cluster_ids))
    reverse_map[-1] = -1
    corrected = np.array([reverse_map[label] for label in corrected])
    save_cluster_results_vision_mf(
        corrected, face_segments, face_indices, result_dir / "faces_mid_frame_face_hmm.json")
    np.savez(result_dir / "face_hmm_params.npz", alpha=model.alpha_,
             A_F=model.A_F_, B_F=model.B_F_, cluster_ids=np.array(cluster_ids + [-1]))
    info = dict(vars(args), ahc_source=str(source) if reused_ahc else None,
                n_faces=len(labels), n_frames=len(timeline), lengths=lengths,
                n_channels=F_hat.shape[1], changed_faces=int(np.sum(labels != corrected)))
    with open(result_dir / "run_info.json", "w") as stream:
        json.dump(info, stream, indent=2)
    print(f"Results: {result_dir}; corrected {info['changed_faces']} / {len(labels)} faces.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", required=True)
    parser.add_argument("--wavs", required=True, help="Episode list; waveforms are not read")
    parser.add_argument("--visual_embs_dir", required=True)
    parser.add_argument("--subseg_json", required=True, help="Original subtitle segment timeline")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--initial_labels", help="Existing AHC face-label JSON")
    parser.add_argument("--candidates", type=int, default=2, help="Nearest visual clusters per face")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--random_state", type=int, default=100)
    args = parser.parse_args()
    if args.candidates < 1 or args.n_iter < 1 or args.tol <= 0:
        parser.error("candidates, n_iter, and tol must be positive")
    run(args)


if __name__ == "__main__":
    main()
