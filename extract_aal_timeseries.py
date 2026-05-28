#!/usr/bin/env python
"""Extract AAL ROI time series from preprocessed HCP REST1 archives.

This follows the logic in subject_connectome.sh after AFNI bandpass filtering:
1. read f_1_LR.nii.gz and f_1_RL.nii.gz from each subject archive;
2. z-score each voxel time series within each run separately;
3. concatenate LR then RL in time;
4. average voxel-wise z-scored signals within each selected AAL ROI.
"""

from __future__ import annotations

import argparse
import csv
import tarfile
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from tqdm import tqdm


def read_aal_labels(label_file: Path) -> list[tuple[int, str]]:
    labels: list[tuple[int, str]] = []
    with label_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                continue
            labels.append((int(row[2]), row[1]))
    return labels


def subject_id_from_archive(archive: Path) -> str:
    name = archive.name
    suffix = "_fmri_preproc.tar.gz"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name.replace(".tar.gz", "")


def extract_member(archive: Path, member_name: str, target_dir: Path) -> Path:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(f"{member_name} not found in {archive}") from exc
        source = tar.extractfile(member)
        if source is None:
            raise FileNotFoundError(f"{member_name} is not a regular file in {archive}")
        target = target_dir / Path(member_name).name
        with target.open("wb") as f:
            f.write(source.read())
    return target


def load_labels_on_func_grid(atlas_file: Path, func_img: nib.Nifti1Image) -> np.ndarray:
    atlas_img = nib.load(str(atlas_file))
    same_shape = atlas_img.shape[:3] == func_img.shape[:3]
    same_affine = np.allclose(atlas_img.affine, func_img.affine)
    if not (same_shape and same_affine):
        atlas_img = resample_to_img(
            atlas_img,
            func_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    return np.asarray(atlas_img.dataobj, dtype=np.int32)


def run_roi_timeseries(
    func_file: Path, atlas_labels: np.ndarray, labels: list[tuple[int, str]]
) -> np.ndarray:
    img = nib.load(str(func_file))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"{func_file} is not a 4D image")

    n_time = data.shape[3]
    out = np.full((n_time, len(labels)), np.nan, dtype=np.float32)

    for col, (roi_value, _) in enumerate(labels):
        mask = atlas_labels == roi_value
        if not np.any(mask):
            continue

        roi_data = data[mask, :]
        mean = roi_data.mean(axis=1, keepdims=True)
        std = roi_data.std(axis=1, ddof=0, keepdims=True)
        valid = np.isfinite(std[:, 0]) & (std[:, 0] > 0)
        if not np.any(valid):
            continue

        z = (roi_data[valid, :] - mean[valid, :]) / std[valid, :]
        out[:, col] = z.mean(axis=0)

    return out


def process_archive(
    archive: Path,
    atlas_file: Path,
    labels: list[tuple[int, str]],
    output_dir: Path,
    roi_count: int,
    keep_extracted: bool,
) -> Path:
    subject = subject_id_from_archive(archive)
    output_file = output_dir / f"{subject}_AAL{roi_count}_timeseries.csv"
    if output_file.exists():
        return output_file

    with tempfile.TemporaryDirectory(prefix=f"{subject}_", dir=output_dir) as tmp:
        tmp_dir = Path(tmp)
        lr_file = extract_member(archive, f"{subject}/f_1_LR.nii.gz", tmp_dir)
        rl_file = extract_member(archive, f"{subject}/f_1_RL.nii.gz", tmp_dir)

        lr_img = nib.load(str(lr_file))
        atlas_labels = load_labels_on_func_grid(atlas_file, lr_img)

        lr_ts = run_roi_timeseries(lr_file, atlas_labels, labels)
        rl_ts = run_roi_timeseries(rl_file, atlas_labels, labels)
        ts = np.vstack([lr_ts, rl_ts])

        columns = [name for _, name in labels]
        index = pd.Index(np.arange(ts.shape[0]), name="timepoint")
        pd.DataFrame(ts, index=index, columns=columns).to_csv(output_file)

        if keep_extracted:
            keep_dir = output_dir / "extracted" / subject
            keep_dir.mkdir(parents=True, exist_ok=True)
            lr_file.replace(keep_dir / lr_file.name)
            rl_file.replace(keep_dir / rl_file.name)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AAL ROI time series from HCP REST1 preprocessed archives."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing *_fmri_preproc.tar.gz archives.",
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path("AAL_atlas/ROI_MNI_V4.nii"),
        help="AAL atlas NIfTI file.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("AAL_atlas/ROI_MNI_V4.txt"),
        help="AAL atlas label text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output CSV files. Default: roi_timeseries/AAL<roi-count>.",
    )
    parser.add_argument(
        "--roi-count",
        type=int,
        default=90,
        help="Number of labels to use from ROI_MNI_V4.txt. Use 90 to match group_connectome.sh, or 116 for the full atlas.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional subject IDs to process. Default: all matching archives.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted f_1_LR/RL files under output-dir/extracted.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(f"roi_timeseries/AAL{args.roi_count}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_aal_labels(args.labels)
    if not labels:
        raise ValueError(f"No labels found in {args.labels}")
    if args.roi_count < 1 or args.roi_count > len(labels):
        raise ValueError(f"--roi-count must be between 1 and {len(labels)}")
    labels = labels[: args.roi_count]

    archives = sorted(args.data_dir.glob("*_fmri_preproc.tar.gz"))
    if args.subjects:
        wanted = set(args.subjects)
        archives = [a for a in archives if subject_id_from_archive(a) in wanted]
    if not archives:
        raise FileNotFoundError("No matching *_fmri_preproc.tar.gz archives found")

    for archive in tqdm(archives, desc="Subjects"):
        out = process_archive(
            archive=archive,
            atlas_file=args.atlas,
            labels=labels,
            output_dir=args.output_dir,
            roi_count=args.roi_count,
            keep_extracted=args.keep_extracted,
        )
        tqdm.write(f"wrote {out}")


if __name__ == "__main__":
    main()
