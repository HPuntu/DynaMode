import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import mdtraj as md
from tqdm import tqdm
import argparse
import multiprocessing
import os


def process_atlas_data(extract_path, domain_dir, atlas_id):
    '''
    Saves all trajectory replicas at full time resolution (10ps/frame, ~10000 frames).
    Outputs are saved directly into the specific domain_dir.
    '''
    traj_files = list(extract_path.glob("*.xtc"))
    topo_files = list(extract_path.glob("*.pdb")) or list(extract_path.glob("*.gro"))

    if not traj_files or not topo_files:
        raise FileNotFoundError("Missing .xtc or topology (.pdb/.gro) in ZIP.")

    topo_path = str(topo_files[0])

    # Save the topology just once inside the domain folder
    topo_out = domain_dir / f"{atlas_id}_topo.pdb"
    t_topo = md.load(str(traj_files[0]), top=topo_path)
    t_topo[0].save_pdb(str(topo_out))

    # Save every trajectory file at full resolution
    for traj_file in traj_files:
        repeat_name = traj_file.stem
        traj_out = domain_dir / f"{atlas_id}_{repeat_name}_sliced.xtc"
        t = md.load(str(traj_file), top=topo_path)
        t.save_xtc(str(traj_out))

    return True

def download_and_process(pdb_id, chain_id, outdir):
    atlas_id = f"{pdb_id.lower()}_{chain_id.upper()}"

    domain_dir = outdir / atlas_id

    if domain_dir.exists():
        existing_xtcs = list(domain_dir.glob(f"{atlas_id}_*_sliced.xtc"))
        if len(existing_xtcs) > 0 and (domain_dir / f"{atlas_id}_topo.pdb").exists():
            return "skipped", atlas_id

    url = f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{atlas_id}"

    tmp_zip = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_zip = Path(tmp.name)

        urllib.request.urlretrieve(url, str(tmp_zip))

        with tempfile.TemporaryDirectory() as tmp_extract_dir:
            tmp_path = Path(tmp_extract_dir)
            with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
                zip_ref.extractall(tmp_path)

            domain_dir.mkdir(parents=True, exist_ok=True)
            process_atlas_data(tmp_path, domain_dir, atlas_id)
            return "success", atlas_id

    except Exception:
        return "failed", atlas_id
    finally:
        if tmp_zip and tmp_zip.exists():
            tmp_zip.unlink()

def _worker_task(args):
    pdb_id, chain_id, outdir = args
    status, atlas_id = download_and_process(pdb_id, chain_id, outdir)
    return status, atlas_id

def main():
    parser = argparse.ArgumentParser(description='Download ATLAS trajectories at full time resolution (10ps/frame)')
    parser.add_argument("-f", "--file", type=str, required=True, help="Input .txt file of domain IDs")
    parser.add_argument("-o", "--outdir", type=str, default="atlas_data", help="Output directory")
    parser.add_argument("--nproc", type=int, default=min(4, os.cpu_count() or 1))

    args = parser.parse_args()

    input_txt = Path(args.file)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_txt.exists():
        print(f"Error: {input_txt} not found.")
        sys.exit(1)

    entries = []
    with open(input_txt, "r") as f:
        for line in f:
            val = line.strip()
            if not val: continue
            if "_" in val:
                p, c = val.split("_")
            else:
                p, c = val[:4], (val[4:] if len(val) > 4 else "A")
            entries.append((p.lower(), c.upper(), output_dir))

    stats = {"success": 0, "failed": 0, "skipped": 0}
    failed_list = []

    print(f"Processing {len(entries)} entries with {args.nproc} workers...")

    with multiprocessing.Pool(processes=args.nproc) as pool:
        for status, atlas_id in tqdm(pool.imap_unordered(_worker_task, entries), total=len(entries)):
            stats[status] += 1
            if status == "failed":
                failed_list.append(atlas_id)

    print(f"\n--- Final Status ---")
    print(f"  - Successfully processed: {stats['success']}")
    print(f"  - Already present (skipped): {stats['skipped']}")
    print(f"  - Failed/Missing on ATLAS: {stats['failed']}")

    if failed_list:
        print(f"\nFailed IDs: {', '.join(failed_list)}")

if __name__ == "__main__":
    main()
