from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
import mdtraj as md
import zarr
from tqdm import tqdm

from src.zarr_feature_extraction import (
    build_native_feature_bundle,
    infer_shard_args,
    make_alignment_runtime,
    prepare_dynamic_recompute,
    process_backbone_replica,
    shard_items,
    write_static_features,
)


def find_atlas_systems(atlas_root: str) -> dict[str, dict[str, list[str] | str]]:
    """Find ATLAS systems in either per-domain subdirectories or a flat layout."""
    root = Path(atlas_root)
    systems: dict[str, dict[str, list[str] | str]] = {}

    if root.exists():
        for entry in sorted(os.scandir(root), key=lambda item: item.name):
            if not entry.is_dir():
                continue
            domain_id = entry.name
            subdir = Path(entry.path)
            pdb_path = subdir / f"{domain_id}_topo.pdb"
            if not pdb_path.exists():
                pdb_path = subdir / f"{domain_id}.pdb"
            if not pdb_path.exists():
                continue
            xtcs = sorted(glob.glob(str(subdir / f"{domain_id}*.xtc")))
            if xtcs:
                systems[domain_id] = {"pdb": str(pdb_path), "xtcs": xtcs}

    for pdb_path_str in sorted(glob.glob(str(root / "*.pdb"))):
        pdb_path = Path(pdb_path_str)
        domain_id = pdb_path.name.replace("_topo.pdb", "").replace(".pdb", "")
        if domain_id in systems:
            continue
        xtcs = sorted(glob.glob(str(root / f"{domain_id}*.xtc")))
        if xtcs:
            systems[domain_id] = {"pdb": str(pdb_path), "xtcs": xtcs}

    return systems


def read_pdb_text(pdb_path: str) -> str:
    with open(pdb_path, "r") as handle:
        return handle.read()


def build_atlas_zarr2(
    atlas_raw_dir: str,
    zarr_path: str,
    window_size: int = 256,
    force: bool = False,
    force_static: bool = False,
    max_domains: int | None = None,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> None:
    print(f"Scanning ATLAS raw directory: {atlas_raw_dir}")
    systems = find_atlas_systems(atlas_raw_dir)
    domain_ids = sorted(systems.keys())
    if max_domains is not None:
        domain_ids = domain_ids[:max_domains]
    domain_ids = shard_items(domain_ids, shard_idx, num_shards)
    print(f"Found {len(domain_ids)} systems for shard {shard_idx}/{num_shards}.")

    print(f"Opening Zarr store at: {zarr_path} in append mode.")
    root = zarr.open(zarr_path, mode="a")
    device, aligner = make_alignment_runtime()
    print(f"Running alignment on: {device}")

    success_count = 0
    fail_count = 0
    written_reps = 0
    skipped_reps = 0

    for domain_id in tqdm(domain_ids, desc="Processing ATLAS systems"):
        system = systems[domain_id]
        pdb_path = str(system["pdb"])
        xtc_paths = list(system["xtcs"])

        try:
            native_traj = md.load_pdb(pdb_path)
            pdb_str = read_pdb_text(pdb_path)
            features = build_native_feature_bundle(native_traj)

            domain_group = root.require_group(domain_id)
            write_static_features(domain_group, pdb_str, features, force=force_static)
            prepare_dynamic_recompute(domain_group, force=force)
        except Exception as exc:
            print(f"\n[Error] Failed to prepare {domain_id}: {exc}")
            fail_count += 1
            continue

        for rep_idx, xtc_path in enumerate(xtc_paths):
            try:
                full_traj = md.load(xtc_path, top=pdb_path)
                coords_angstrom = full_traj.xyz * 10.0
                status = process_backbone_replica(
                    domain_group,
                    "300",
                    str(rep_idx),
                    coords_angstrom,
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
                print(f"\n[Error] Failed on {domain_id} run {xtc_path}: {exc}")
                continue

        success_count += 1

    print("\n=== ATLAS EXTRACTION V2 COMPLETE ===")
    print(f"Successfully processed systems: {success_count} | Failed: {fail_count}")
    print(f"Replicas written: {written_reps} | Skipped: {skipped_reps}")

    if success_count > 0:
        print("Reconsolidating Zarr metadata...")
        zarr.consolidate_metadata(zarr_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and featurise ATLAS PDB/XTC trajectories into a unified Zarr store."
    )
    parser.add_argument("shard_idx_pos", type=int, nargs="?")
    parser.add_argument("num_shards_pos", type=int, nargs="?")
    parser.add_argument("--atlas_dir", "--atlas_raw_dir", dest="atlas_dir", type=str, required=True)
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
    build_atlas_zarr2(
        args.atlas_dir,
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
