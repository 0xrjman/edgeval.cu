"""
edgeval.cu CLI — GPU-accelerated edge detection evaluation tool.

Usage:
    edgeval eval <results-dir> --gt-dir <gt-dir> [--gpu] [--full]
    edgeval show <results-dir>
    edgeval nms <input-dir> <output-dir>
    edgeval info
"""
import os
import sys
import click


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
DATASETS = {"BSDS": 200, "BRIND": 200, "NYUD": 654, "BIPED": 50, "UDED": 30}


def _auto_detect_dataset(dir_path):
    """Try to determine which dataset dir_path belongs to."""
    upper = dir_path.upper()
    for name in DATASETS:
        if name in upper:
            return name
    return None


def _resolve_gt_dir(gt_dir, dataset):
    """Resolve GT directory from explicit path or dataset name."""
    if gt_dir:
        return gt_dir
    if dataset:
        # Check common locations
        candidates = [
            f"GT/{dataset}",
            f"GT/{dataset}/test",
            f"gt/{dataset}",
            f"data/GT/{dataset}",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        click.echo(f"⚠️  GT directory for dataset '{dataset}' not found. Use --gt-dir", err=True)
        sys.exit(1)
    click.echo("⚠️  Either --gt-dir or --dataset is required", err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
#  CLI Group
# ---------------------------------------------------------------------------
@click.group()
@click.version_option(version="0.1.0", prog_name="edgeval")
def cli():
    """edgeval.cu — GPU-accelerated edge detection evaluation.

    Compare your edge detection results against standard benchmarks
    (BSDS500, NYUD, BIPED, UDED) using either CPU CSA or GPU Auction
    algorithm.  Use --gpu for ~7.4× speedup.
    """


# ---------------------------------------------------------------------------
#  eval
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("result_dir", type=click.Path(exists=True))
@click.option("-g", "--gt-dir", type=click.Path(exists=True),
              help="Ground truth directory (with .mat files)")
@click.option("-d", "--dataset",
              type=click.Choice(list(DATASETS.keys()) + [k.lower() for k in DATASETS]),
              help="Dataset name (auto-detected from path if omitted)")
@click.option("--gpu", is_flag=True, help="Use GPU acceleration (default: CPU)")
@click.option("-f", "--full", is_flag=True, help="Full evaluation (99 thresholds, default)")
@click.option("--thrs", type=int, default=99, show_default=True,
              help="Number of thresholds (9 = light, 99 = full)")
@click.option("--max-dist", type=float, default=0.0075, show_default=True,
              help="Max matching distance (fraction of image diagonal)")
@click.option("--no-thin", is_flag=True, help="Skip morphological thinning")
@click.option("-nw", "--not-wait", is_flag=True,
              help="Do not wait for new results")
@click.option("--timeout", type=float, default=8, show_default=True,
              help="Max hours to wait for results")
@click.option("--workers", type=int, default=-1, show_default=True,
              help="CPU workers (-1 = all cores, only affects CPU mode)")
def eval_cmd(result_dir, gt_dir, dataset, gpu, full, thrs, max_dist,
             no_thin, not_wait, timeout, workers):
    """Evaluate edge detection results against ground truth.

    RESULT_DIR is a directory containing .png edge maps (0-255).
    Use --gt-dir to specify ground truth, or --dataset for auto location.
    """
    # Resolve dataset
    ds = dataset.upper() if dataset else _auto_detect_dataset(result_dir)
    if not ds:
        click.echo("⚠️  Could not auto-detect dataset. Use --dataset or --gt-dir", err=True)
        sys.exit(1)

    gt = _resolve_gt_dir(gt_dir, ds)
    thrs_val = 9 if not full and thrs == 99 else thrs

    click.echo(f"📊 Dataset:  {ds}")
    click.echo(f"📁 Results:  {result_dir}")
    click.echo(f"📁 GT:       {gt}")
    click.echo(f"⚙️  Thresholds: {'full (99)' if thrs_val == 99 else f'{thrs_val}'}")
    click.echo(f"🚀 Mode:     {'GPU' if gpu else 'CPU'}")
    click.echo("")

    if gpu:
        try:
            from .gpu_eval import gpu_edges_eval_dir
        except ImportError as e:
            click.echo(f"❌ GPU mode not available: {e}", err=True)
            click.echo("   Make sure CUDA toolkit is installed and auction_cuda.so is compiled.", err=True)
            sys.exit(1)

        scores = gpu_edges_eval_dir(
            result_dir, gt, thrs=thrs_val, max_dist=max_dist,
            thin=not no_thin,
        )
    else:
        from ._impl.edges_eval_dir import edges_eval_dir as cpu_eval

        # CPU mode runs the original pipeline
        if not_wait:
            # Direct eval
            cpu_eval(result_dir, gt, thrs=thrs_val, max_dist=max_dist,
                     thin=not no_thin, workers=workers)
        else:
            # Use eval_component for auto-monitoring
            from .eval_component import eval_one_epoch
            eval_one_epoch(result_dir, ds, full=(thrs_val == 99),
                           key="img", file_format=".mat")
        scores = None

    if gpu and scores:
        click.echo("")
        click.echo("═" * 40)
        click.echo(f"  ODS: {scores['ods_f']:.4f}")
        click.echo(f"  OIS: {scores['ois_f']:.4f}")
        if scores['ap'] > 0:
            click.echo(f"  AP:  {scores['ap']:.4f}")
        click.echo("═" * 40)


# ---------------------------------------------------------------------------
#  show
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("result_dir", type=click.Path(exists=True))
@click.option("-f", "--full", is_flag=True, help="Show full eval results")
def show_cmd(result_dir, full):
    """Show evaluation results (ODS/OIS rankings)."""
    from .show import main as show_main

    class Args:
        dir = result_dir
        full = full

    show_main(Args())


# ---------------------------------------------------------------------------
#  nms
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--key", default="img", show_default=True,
              help="Key for .mat files")
@click.option("--format", "file_format", type=click.Choice([".mat", ".npy"]),
              default=".mat", show_default=True, help="Input file format")
def nms_cmd(input_dir, output_dir, key, file_format):
    """Run non-maximum suppression on edge results.

    INPUT_DIR: directory with .mat/.npy edge maps.
    OUTPUT_DIR: where to save .png NMS results.
    """
    from .nms_process import nms_process
    click.echo(f"📁 Input:  {input_dir}")
    click.echo(f"📁 Output: {output_dir}")
    click.echo(f"🔑 Key:    {key}")
    nms_process(input_dir, output_dir, key=key, file_format=file_format)
    click.echo("✅ NMS complete.")


# ---------------------------------------------------------------------------
#  info
# ---------------------------------------------------------------------------
@cli.command()
def info_cmd():
    """Show system info, CUDA availability, and package details."""
    import platform
    click.echo("═" * 40)
    click.echo("  edgeval.cu — System Information")
    click.echo("═" * 40)
    click.echo(f"  Version:    0.1.0")
    click.echo(f"  Python:     {platform.python_version()}")
    click.echo(f"  Platform:   {platform.platform()}")

    # Check CUDA
    try:
        from . import _cuda_available
        click.echo(f"  CUDA:       {'✅ Available' if _cuda_available else '❌ Not available'}")
    except Exception:
        click.echo("  CUDA:       ❌ Not available")

    # Check CPU solver
    solver_path = os.path.join(os.path.dirname(__file__), 'cxx', 'lib', 'solve_csa.so')
    click.echo(f"  CPU Solver: {'✅ Found' if os.path.exists(solver_path) else '❌ Not found'}")

    # Check GPU solver
    gpu_solver_path = os.path.join(os.path.dirname(__file__), 'gpu_eval', 'auction_cuda.so')
    click.echo(f"  GPU Solver: {'✅ Found' if os.path.exists(gpu_solver_path) else '⏳ Not compiled'}")
    click.echo("═" * 40)


# ---------------------------------------------------------------------------
#  Entry
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    cli()
