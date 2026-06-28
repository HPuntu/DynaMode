from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
import h5py
import zarr
from tqdm import tqdm

from src.zarr_feature_extraction import (
    build_native_feature_bundle,
    decode_text,
    infer_shard_args,
    load_pdb_from_string,
    make_alignment_runtime,
    prepare_dynamic_recompute,
    process_backbone_replica,
    shard_items,
    write_static_features,
)


def search_bases(mdcath_path: str) -> tuple[Path, ...]:
    root = Path(mdcath_path)
    return (
        root,
        root / "data",
        root / "data" / "data",
    )


def find_domains(mdcath_path: str) -> list[str]:
    for base in search_bases(mdcath_path):
        src = base / "mdcath_source.h5"
        if src.exists():
            with h5py.File(src, "r") as handle:
                return sorted(handle.keys())

    for base in search_bases(mdcath_path):
        files = sorted(glob.glob(str(base / "mdcath_dataset_*.h5")))
        if files:
            return [
                Path(path).name.replace("mdcath_dataset_", "").replace(".h5", "")
                for path in files
            ]

    raise FileNotFoundError(f"No mdCATH domain source found under {mdcath_path}")


def find_h5(mdcath_path: str, domain_id: str) -> Path | None:
    fname = f"mdcath_dataset_{domain_id}.h5"
    for base in search_bases(mdcath_path):
        candidate = base / fname
        if candidate.exists():
            return candidate
    return None


def read_native_inputs(h5_path: Path, domain_id: str):
    with h5py.File(h5_path, "r") as handle:
        group = handle[domain_id]
        pdb_str = decode_text(group["pdbProteinAtoms"][()])
        raw_dssp = group["dssp"][()] if "dssp" in group else None
    return pdb_str, raw_dssp


def list_replicas(h5_path: Path, domain_id: str) -> list[tuple[str, str]]:
    replicas = []
    with h5py.File(h5_path, "r") as handle:
        domain_group = handle[domain_id]
        for temp_name, temp_obj in domain_group.items():
            if not isinstance(temp_obj, h5py.Group):
                continue
            for rep_name, rep_obj in temp_obj.items():
                if isinstance(rep_obj, h5py.Group) and "coords" in rep_obj:
                    replicas.append((str(temp_name), str(rep_name)))
    return replicas


def load_replica_coords(h5_path: Path, domain_id: str, temp_name: str, rep_name: str):
    with h5py.File(h5_path, "r") as handle:
        return handle[domain_id][temp_name][rep_name]["coords"][:]


def build_mdcath_zarr2(
    mdcath_path: str,
    zarr_path: str,
    window_size: int = 256,
    force: bool = False,
    force_static: bool = False,
    max_domains: int | None = None,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> None:
    print(f"Discovering mdCATH domains from: {mdcath_path}")
    domains = find_domains(mdcath_path)
    if max_domains is not None:
        domains = domains[:max_domains]
    domains = shard_items(domains, shard_idx, num_shards)
    print(f"Found {len(domains)} domains for shard {shard_idx}/{num_shards}.")

    print(f"Opening Zarr store at: {zarr_path} in append mode.")
    root = zarr.open(zarr_path, mode="a")
    device, aligner = make_alignment_runtime()
    print(f"Running alignment on: {device}")

    success_count = 0
    fail_count = 0
    written_reps = 0
    skipped_reps = 0
    h5_dssp_count = 0
    computed_dssp_count = 0

    for domain_id in tqdm(domains, desc="Processing mdCATH domains"):
        h5_path = find_h5(mdcath_path, domain_id)
        if h5_path is None:
            print(f"\n[Warning] No h5 file for {domain_id}. Skipping.")
            fail_count += 1
            continue

        try:
            pdb_str, raw_dssp = read_native_inputs(h5_path, domain_id)
            native_traj = load_pdb_from_string(pdb_str)
            features = build_native_feature_bundle(native_traj, raw_dssp=raw_dssp)
            if raw_dssp is None:
                computed_dssp_count += 1
            else:
                h5_dssp_count += 1

            domain_group = root.require_group(domain_id)
            write_static_features(domain_group, pdb_str, features, force=force_static)
            prepare_dynamic_recompute(domain_group, force=force)
            replicas = list_replicas(h5_path, domain_id)
        except Exception as exc:
            print(f"\n[Error] Failed to prepare {domain_id}: {exc}")
            fail_count += 1
            continue

        for temp_name, rep_name in replicas:
            try:
                coords = load_replica_coords(h5_path, domain_id, temp_name, rep_name)
                status = process_backbone_replica(
                    domain_group,
                    temp_name,
                    rep_name,
                    coords,
                    features,
                    aligner,
                    device,
                    window_size=window_size,
                    force=force,
                )
                if status == "skipped":
                    skipped_reps += 1
                else:
                    written_reps += 1
            except Exception as exc:
                print(f"\n[Error] Failed on {domain_id}/{temp_name}/{rep_name}: {exc}")
                continue

        success_count += 1

    print("\n=== mdCATH EXTRACTION V2 COMPLETE ===")
    print(f"Successfully processed domains: {success_count} | Failed: {fail_count}")
    print(f"Replicas written: {written_reps} | Skipped: {skipped_reps}")
    print(f"DSSP source counts: h5={h5_dssp_count} | computed_from_pdb={computed_dssp_count}")

    if success_count > 0:
        print("Reconsolidating Zarr metadata...")
        zarr.consolidate_metadata(zarr_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and featurise mdCATH h5 trajectories into a unified Zarr store."
    )
    parser.add_argument("shard_idx_pos", type=int, nargs="?")
    parser.add_argument("num_shards_pos", type=int, nargs="?")
    parser.add_argument("--mdcath_path", type=str, required=True)
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force_static",
        action="store_true",
        help="Rewrite static arrays such as dssp, res_type, native_angles, and pdbProteinAtoms.",
    )
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--max_domains", type=int, default=None)
    args, _ = parser.parse_known_args()

    shard_idx, num_shards = infer_shard_args(args)
    build_mdcath_zarr2(
        args.mdcath_path,
        args.zarr_path,
        window_size=args.window_size,
        force=args.force,
        force_static=args.force_static,
        max_domains=args.max_domains,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )


if __name__ == "__main__":
    main()
