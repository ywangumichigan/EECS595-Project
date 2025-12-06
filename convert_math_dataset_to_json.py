#!/usr/bin/env python3
"""
Convert MATH dataset from parquet files to JSON format with level and type information.

This script reads all test parquet files from math_dataset directory and converts them
to a single JSON file with the following fields:
- level: Difficulty level (e.g., "Level 1", "Level 2", etc.)
- type: Subject type (e.g., "algebra", "geometry", etc.)
- problem: The math problem/question
- instruction: Combined system prompt + problem (for compatibility with evaluation)
- output: The solution with boxed answer (for compatibility with evaluation)

Output format matches the Aliyun SFT format while preserving metadata.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def convert_parquet_to_json(math_dataset_dir: str, output_file: str, split: str = "test"):
    """
    Convert MATH dataset parquet files to JSON format.
    
    Args:
        math_dataset_dir: Path to math_dataset directory
        output_file: Path to output JSON file
        split: Which split to process ("test" or "train")
    """
    math_dataset_path = Path(math_dataset_dir)
    
    if not math_dataset_path.exists():
        raise FileNotFoundError(f"Math dataset directory not found: {math_dataset_dir}")
    
    # Subject types (directories)
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]
    
    SYSTEM_PROMPT = (
        "You are a math assistant. Solve the problem step by step, "
        "explain your reasoning, and box the final answer using \\boxed{}."
    )
    
    all_data = []
    
    print(f"Processing {split} split from {math_dataset_dir}...")
    
    for subject in subjects:
        subject_dir = math_dataset_path / subject
        parquet_file = subject_dir / f"{split}-00000-of-00001.parquet"
        
        if not parquet_file.exists():
            print(f"Warning: {parquet_file} not found, skipping {subject}")
            continue
        
        print(f"\nProcessing {subject}...")
        
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        print(f"  Found {len(df)} examples")
        
        # Convert each row to the desired format
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Converting {subject}"):
            problem = row.get('problem', '')
            level = row.get('level', '')
            type_field = row.get('type', subject)  # Use directory name as fallback
            solution = row.get('solution', '')
            
            # Create instruction (system prompt + problem)
            instruction = f"{SYSTEM_PROMPT}\n\n{problem}"
            
            # Create entry (removed redundant "solution" field, using only "output")
            entry = {
                "level": level,
                "type": type_field,
                "problem": problem,
                "instruction": instruction,
                "output": solution  # Solution with boxed answer
            }
            
            all_data.append(entry)
    
    # Write to JSON file
    print(f"\nWriting {len(all_data)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Conversion complete! Generated {output_file}")
    print(f"  Total entries: {len(all_data)}")
    
    # Print statistics
    if all_data:
        levels = {}
        types = {}
        for entry in all_data:
            level = entry.get('level', 'Unknown')
            type_field = entry.get('type', 'Unknown')
            levels[level] = levels.get(level, 0) + 1
            types[type_field] = types.get(type_field, 0) + 1
        
        print("\nStatistics:")
        print(f"  Levels: {dict(sorted(levels.items()))}")
        print(f"  Types: {dict(sorted(types.items()))}")
    
    return len(all_data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MATH dataset from parquet files to JSON format with level and type"
    )
    parser.add_argument(
        '--math_dataset_dir',
        type=str,
        default='./math_dataset',
        help='Path to math_dataset directory (default: ./math_dataset)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['test', 'train'],
        default='test',
        help='Which split to process: test or train (default: test)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: test_math.json or train_math.json in root)'
    )
    
    args = parser.parse_args()
    
    # Default output file name
    if args.output is None:
        args.output = f"{args.split}_math.json"
    
    print("=" * 60)
    print("MATH Dataset Converter")
    print("=" * 60)
    
    count = convert_parquet_to_json(
        args.math_dataset_dir,
        args.output,
        args.split
    )
    
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Total examples: {count}")
    print(f"Output file: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

