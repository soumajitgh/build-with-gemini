"""
Phase 1 Orchestrator - Pit Detection Pipeline

This is the main entry point that orchestrates:
1. Pit detection from orthomosaic images
2. Visualization of results (charts + overlay)
"""

import argparse
import sys
from pathlib import Path

from src.phase1_pit_detection import detect_pits_from_file
from src.visualize import visualize_pits, visualize_overlay


def run_pipeline(input_file: str, visualize: bool = True, show_plot: bool = True):
    """
    Run the complete pit detection pipeline.
    
    Args:
        input_file: Path to orthomosaic .tif file (e.g., input/{sitename}/pits.tif)
        visualize: Whether to generate visualization
        show_plot: Whether to display the plot interactively
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Auto-resolve site name from path: input/{sitename}/pits.tif
    site_name = input_path.parent.name
    if site_name == "input" or site_name == ".":
        site_name = input_path.stem
    
    print(f"{'='*50}")
    print(f"Phase 1: Pit Detection Pipeline")
    print(f"{'='*50}")
    print(f"Site: {site_name}")
    print(f"Input: {input_path}")
    print()
    
    # Step 1: Detect pits
    print("[1/3] Running pit detection...")
    df = detect_pits_from_file(str(input_path), site_name=site_name)
    
    if df is None:
        print("Error: Detection failed.")
        sys.exit(1)
    
    if df.empty:
        print("Warning: No pits detected.")
        sys.exit(0)
    
    # Create output directory
    output_dir = Path("output") / site_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "map.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved coordinates to {csv_path}")
    
    # Step 2: Generate chart visualization
    if visualize:
        print()
        print("[2/3] Generating diagnostic charts...")
        viz_path = output_dir / "visualization.png"
        visualize_pits(str(csv_path), output_path=str(viz_path), show=False)
        
        # Step 3: Generate overlay visualization
        print("[3/3] Generating overlay map...")
        overlay_path = output_dir / "overlay.png"
        visualize_overlay(
            str(input_path),
            str(csv_path),
            output_path=str(overlay_path),
            show=show_plot
        )
    
    # Summary
    print()
    print(f"{'='*50}")
    print("Pipeline Complete!")
    print(f"{'='*50}")
    print(f"Total pits detected: {len(df)}")
    print(f"Output directory: {output_dir}")
    print(f"  - map.csv: GPS coordinates")
    if visualize:
        print(f"  - visualization.png: Diagnostic charts")
        print(f"  - overlay.png: Pit locations on orthomosaic")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Orchestrator - Pit Detection Pipeline"
    )
    parser.add_argument(
        "input_file",
        help="Path to orthomosaic .tif file (e.g., input/{sitename}/pits.tif)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization step"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plot interactively (still saves PNG)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        args.input_file,
        visualize=not args.no_visualize,
        show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()
