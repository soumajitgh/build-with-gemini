"""
Visualization module for pit detection results.

Creates diagnostic charts to visualize pit locations and quality metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio


def visualize_overlay(
    image_path: str,
    csv_path: str,
    output_path: str | None = None,
    show: bool = True,
    dot_size: int = 20,
    dot_color: str = "red",
    brighten: float = 1.0
):
    """
    Overlay detected pits on the original orthomosaic image.
    
    Args:
        image_path: Path to the orthomosaic .tif file
        csv_path: Path to the pit coordinates CSV file
        output_path: Optional path to save the figure (PNG)
        show: Whether to display the plot interactively
        dot_size: Size of the pit markers
        dot_color: Color of the pit markers
        brighten: Brightness multiplier for the image (1.0 = original)
    """
    image_path = Path(image_path)
    csv_path = Path(csv_path)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Load pit coordinates
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pits from {csv_path}")
    
    if df.empty:
        print("No pits to visualize.")
        return
    
    # Load and display the orthomosaic
    with rasterio.open(image_path) as src:
        print(f"Loading image: {src.width} x {src.height} pixels")
        
        if src.count >= 3:
            # Read RGB bands
            r = src.read(1)
            g = src.read(2)
            b = src.read(3)
            # Stack into (Height, Width, 3) format
            img = np.dstack((r, g, b))
            
            # Optional brightness adjustment
            if brighten != 1.0:
                img = np.clip(img * brighten, 0, 255).astype(np.uint8)
        else:
            # Grayscale
            img = src.read(1)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 14))
        
        # Show the map as background
        ax.imshow(img, origin='upper')
        
        # Overlay pit locations as red dots
        ax.scatter(
            df['pixel_x'],
            df['pixel_y'],
            c=dot_color,
            s=dot_size,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
            label=f'Detected Pits ({len(df)})'
        )
        
        # Styling
        site_name = df["site"].iloc[0] if "site" in df.columns else "Unknown"
        ax.set_title(f"Pit Detection Overlay - {site_name} ({len(df)} Pits)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved overlay visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def visualize_pits(csv_path: str, output_path: str | None = None, show: bool = True):
    """
    Generate diagnostic visualization for detected pits.
    
    Args:
        csv_path: Path to the pit coordinates CSV file
        output_path: Optional path to save the figure (PNG)
        show: Whether to display the plot interactively
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pits from {csv_path}")
    
    if df.empty:
        print("No data to visualize.")
        return
    
    # Set up figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    site_name = df["site"].iloc[0] if "site" in df.columns else "Unknown"
    fig.suptitle(f"Pit Detection Analysis - {site_name}", fontsize=14, fontweight="bold")
    
    # Plot 1: Spatial Map (Latitude vs Longitude)
    sns.scatterplot(
        data=df,
        x="lon",
        y="lat",
        hue="area_px",
        size="area_px",
        sizes=(20, 200),
        palette="viridis",
        ax=axes[0],
        legend="brief"
    )
    axes[0].set_title("Spatial Distribution (Lat/Lon)")
    axes[0].set_xlabel("Longitude (UTM)")
    axes[0].set_ylabel("Latitude (UTM)")
    axes[0].set_aspect("equal")
    
    # Plot 2: Area Distribution
    sns.histplot(data=df, x="area_px", bins=20, kde=True, ax=axes[1], color="skyblue")
    axes[1].set_title("Pit Size Distribution (Area in px)")
    axes[1].axvline(df["area_px"].mean(), color="red", linestyle="--", label=f'Mean: {df["area_px"].mean():.1f}')
    axes[1].axvline(df["area_px"].median(), color="green", linestyle="--", label=f'Median: {df["area_px"].median():.1f}')
    axes[1].legend()
    
    # Plot 3: Solidity Check
    if "solidity" in df.columns:
        sns.histplot(data=df, x="solidity", bins=15, kde=True, ax=axes[2], color="orange")
        axes[2].set_title("Shape Quality (Solidity)")
        axes[2].set_xlim(0, 1.1)
        axes[2].axvline(df["solidity"].mean(), color="red", linestyle="--", label=f'Mean: {df["solidity"].mean():.2f}')
        axes[2].legend()
    else:
        # Fallback: show aspect ratio if available
        if "aspect_ratio" in df.columns:
            sns.histplot(data=df, x="aspect_ratio", bins=15, kde=True, ax=axes[2], color="orange")
            axes[2].set_title("Shape Aspect Ratio")
        else:
            axes[2].text(0.5, 0.5, "No solidity data", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_title("Shape Quality")
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """CLI entry point for visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize pit detection results"
    )
    parser.add_argument("csv_file", help="Path to pit coordinates CSV")
    parser.add_argument("--output", "-o", help="Output PNG file path")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot interactively")
    parser.add_argument("--image", "-i", help="Path to orthomosaic .tif for overlay visualization")
    parser.add_argument("--dot-size", type=int, default=20, help="Size of pit markers (default: 20)")
    parser.add_argument("--brighten", type=float, default=1.0, help="Brightness multiplier (default: 1.0)")
    
    args = parser.parse_args()
    
    if args.image:
        # Overlay visualization
        visualize_overlay(
            args.image,
            args.csv_file,
            output_path=args.output,
            show=not args.no_show,
            dot_size=args.dot_size,
            brighten=args.brighten
        )
    else:
        # Chart visualization
        visualize_pits(
            args.csv_file,
            output_path=args.output,
            show=not args.no_show
        )


if __name__ == "__main__":
    main()
