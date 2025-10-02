#!/usr/bin/env python3
"""
NII Preprocessing Script

This script processes 3D NII files by extracting individual slices and saving them as separate 2D NII files.
Useful for preparing datasets where lazy loading expects 2D images.

Usage:
    python scripts/preprocess_nii.py <input_directory> [options]

Examples:
    # Process all NII files in a directory (non-recursive)
    python scripts/preprocess_nii.py data/input/

    # Process recursively with custom output directory
    python scripts/preprocess_nii.py data/input/ -r -o data/output/

    # Process in-place recursively, keeping original files
    python scripts/preprocess_nii.py data/input/ -r --keep-originals
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ct_clf_hack.shared.file_utils import preprocess_nii_directory
from ct_clf_hack.shared.logger_utils import default_logger


def main():
    logger = default_logger
    logger.info("Starting NII preprocessing script")

    parser = argparse.ArgumentParser(
        description="Preprocess 3D NII files by extracting individual slices as 2D NII files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/input/                           # Process directory (non-recursive)
  %(prog)s data/input/ -r                       # Process recursively
  %(prog)s data/input/ -r -o data/output/       # Process with custom output directory
  %(prog)s data/input/ -r --keep-originals      # Keep original 3D files
        """
    )

    # Required arguments
    parser.add_argument(
        "input_directory",
        type=str,
        help="Input directory containing NII files to process"
    )

    # Optional arguments
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=False,
        help="Process subdirectories recursively (default: False)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for processed files (default: process in-place)"
    )

    parser.add_argument(
        "--keep-originals",
        action="store_true",
        default=False,
        help="Keep original 3D NII files after processing (default: remove originals)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_directory)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Validate output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        if output_dir.exists() and not output_dir.is_dir():
            logger.error(f"Output path exists but is not a directory: {output_dir}")
            sys.exit(1)

    # Log processing parameters
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir if output_dir else 'In-place processing'}")
    logger.info(f"Recursive processing: {args.recursive}")
    logger.info(f"Keep original files: {args.keep_originals}")

    try:
        # Process the directory
        stats = preprocess_nii_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            recursive=args.recursive,
            remove_originals=not args.keep_originals
        )

        # Print summary
        print("\n" + "="*50)
        print("NII PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Input directory:        {input_dir}")
        print(f"Output directory:       {output_dir if output_dir else 'In-place'}")
        print(f"Recursive processing:   {args.recursive}")
        print(f"Keep originals:         {args.keep_originals}")
        print("-"*50)
        print(f"Total files found:      {stats['total_files']}")
        print(f"3D files processed:     {stats['processed_3d_files']}")
        print(f"2D files skipped:       {stats['skipped_2d_files']}")
        print(f"Total slices created:   {stats['total_slices_created']}")
        print(f"Errors encountered:     {stats['errors']}")
        print("="*50)

        if stats['errors'] > 0:
            logger.warning(f"Processing completed with {stats['errors']} errors. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("NII preprocessing completed successfully!")
            print("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
