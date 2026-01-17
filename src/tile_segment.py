#!/usr/bin/env python3
"""
GeoTIFF Tiling Script

Splits large GeoTIFF files into smaller tiles while PRESERVING geospatial metadata.
This is critical for the Build with Gemini competition where coordinate matching
between OP1 (pits) and OP3 (saplings) is required.

Usage:
    python -m src.tile_segment --project debadihi --tile-size 1024
    python -m src.tile_segment --project debadihi --tile-size 1024 --overlap 200
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.windows import Window


def split_geotiff(
    input_path: str | Path,
    output_folder: str | Path,
    tile_size: int = 1024,
    overlap: int = 0,
) -> dict:
    """
    Splits a GeoTIFF into tiles while PRESERVING coordinate data.

    Args:
        input_path: Path to the input GeoTIFF file.
        output_folder: Directory to save the output tiles.
        tile_size: Size of each tile in pixels (default: 1024).
        overlap: Number of pixels to overlap between tiles (default: 0).
                 Use overlap to avoid cutting objects at tile boundaries.

    Returns:
        dict: Statistics about the tiling operation.
    """
    input_path = Path(input_path)
    output_folder = Path(output_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Calculate stride (step size between tiles)
    stride = tile_size - overlap

    stats = {
        "input_file": str(input_path),
        "output_folder": str(output_folder),
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": stride,
        "tiles_created": 0,
        "tiles": [],
    }

    with rasterio.open(input_path) as src:
        # Get image dimensions
        img_width = src.width
        img_height = src.height

        stats["input_width"] = img_width
        stats["input_height"] = img_height
        stats["crs"] = str(src.crs)
        stats["pixel_resolution"] = (src.res[0], src.res[1])

        print(f"Processing: {input_path}")
        print(f"  Dimensions: {img_width} x {img_height} pixels")
        print(f"  CRS: {src.crs}")
        print(f"  Pixel resolution: {src.res[0]:.4f} x {src.res[1]:.4f} units")
        print(f"  Tile size: {tile_size}px, Overlap: {overlap}px, Stride: {stride}px")
        print()

        # Calculate number of tiles
        num_tiles_x = (img_width + stride - 1) // stride
        num_tiles_y = (img_height + stride - 1) // stride
        total_tiles = num_tiles_x * num_tiles_y

        print(f"  Expected tiles: {num_tiles_x} x {num_tiles_y} = {total_tiles}")
        print()

        tile_count = 0

        # Loop through the image in steps of 'stride'
        for col_idx, i in enumerate(range(0, img_width, stride)):
            for row_idx, j in enumerate(range(0, img_height, stride)):
                # Define the window (crop area)
                # Use min() to handle the edges where the tile might be smaller
                width = min(tile_size, img_width - i)
                height = min(tile_size, img_height - j)

                # Skip very small edge tiles (less than 25% of tile_size)
                min_tile_dimension = tile_size // 4
                if width < min_tile_dimension or height < min_tile_dimension:
                    continue

                window = Window(col_off=i, row_off=j, width=width, height=height)

                # Calculate the new geospatial transform for this specific tile
                transform = src.window_transform(window)

                # Read the pixel data for this window
                data = src.read(window=window)

                # Update metadata for the new tile (height, width, transform)
                profile = src.profile.copy()
                profile.update(
                    {
                        "driver": "GTiff",
                        "height": height,
                        "width": width,
                        "transform": transform,
                        # Optimize for smaller file sizes
                        "compress": "lzw",
                        # Don't use tiled=True for edge tiles as it requires 16-multiple dimensions
                    }
                )

                # Create tile filename with row and column info
                tile_filename = f"tile_{row_idx:04d}_{col_idx:04d}.tif"
                output_path = output_folder / tile_filename

                # Save the tile
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(data)

                # Calculate bounds for this tile
                bounds = rasterio.windows.bounds(window, src.transform)

                tile_info = {
                    "filename": tile_filename,
                    "row": row_idx,
                    "col": col_idx,
                    "pixel_x": i,
                    "pixel_y": j,
                    "width": width,
                    "height": height,
                    "bounds": {
                        "left": bounds[0],
                        "bottom": bounds[1],
                        "right": bounds[2],
                        "top": bounds[3],
                    },
                }
                stats["tiles"].append(tile_info)

                tile_count += 1
                if tile_count % 100 == 0:
                    print(f"  Created {tile_count} tiles...")

        stats["tiles_created"] = tile_count
        print(f"\nProcessing complete!")
        print(f"  Total tiles created: {tile_count}")
        print(f"  Output folder: {output_folder}")

    return stats


def process_all_tiffs_in_project(
    project_path: str | Path,
    tile_size: int = 1024,
    overlap: int = 0,
) -> list[dict]:
    """
    Process all TIFF files in a project directory.

    Args:
        project_path: Path to the project directory containing TIFF files.
        tile_size: Size of each tile in pixels.
        overlap: Number of pixels to overlap between tiles.

    Returns:
        list: List of statistics dicts for each processed file.
    """
    project_path = Path(project_path)
    all_stats = []

    # Find all TIFF files in the project directory
    tiff_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    tiff_files = []
    for pattern in tiff_patterns:
        tiff_files.extend(project_path.glob(pattern))

    if not tiff_files:
        print(f"No TIFF files found in {project_path}")
        return all_stats

    print(f"Found {len(tiff_files)} TIFF file(s) in {project_path}")
    print("=" * 60)

    for tiff_file in sorted(tiff_files):
        # Create output folder based on the input filename
        output_folder_name = f"{tiff_file.stem}_tiles"
        output_folder = project_path / output_folder_name

        print(f"\nProcessing: {tiff_file.name}")
        print("-" * 40)

        try:
            stats = split_geotiff(
                input_path=tiff_file,
                output_folder=output_folder,
                tile_size=tile_size,
                overlap=overlap,
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {tiff_file}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Summary:")
    total_tiles = sum(s.get("tiles_created", 0) for s in all_stats)
    print(f"  Total files processed: {len(all_stats)}")
    print(f"  Total tiles created: {total_tiles}")

    return all_stats


def main():
    """Main entry point for the tile segmentation script."""
    parser = argparse.ArgumentParser(
        description="Split GeoTIFF files into tiles while preserving geospatial metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all TIFFs in a project with default 1024px tiles
    python -m src.tile_segment --project debadihi

    # Use 512px tiles with 100px overlap for better edge detection
    python -m src.tile_segment --project debadihi --tile-size 512 --overlap 100

    # Process a single file
    python -m src.tile_segment --input input/debadihi/weeding.tif --output output/tiles
        """,
    )

    parser.add_argument(
        "--project",
        type=str,
        help="Project name (folder inside 'input/' directory). Processes all TIFFs in the project.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a single GeoTIFF file to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for tiles (required if --input is specified).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Size of each tile in pixels (default: 1024).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between tiles in pixels (default: 0). Use for ML to avoid edge artifacts.",
    )

    args = parser.parse_args()

    # Determine base path
    base_path = Path(__file__).parent.parent

    if args.project:
        # Process all TIFFs in the project
        project_path = base_path / "input" / args.project
        if not project_path.exists():
            print(f"Error: Project directory not found: {project_path}")
            return 1

        process_all_tiffs_in_project(
            project_path=project_path,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
    elif args.input:
        # Process a single file
        if not args.output:
            print("Error: --output is required when using --input")
            return 1

        split_geotiff(
            input_path=args.input,
            output_folder=args.output,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
