#!/usr/bin/env python3
"""
Convert MATH dataset from JSONL format to Aliyun SFT training format.

Input format (JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Output format (JSON):
[
    {
        "instruction": "system message + user message",
        "output": "assistant message"
    },
    ...
]
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_math_to_aliyun_format(input_file: str, output_file: str):
    """
    Convert MATH dataset from JSONL to Aliyun format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file
    """
    converted_data = []
    
    # Read JSONL file
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Extract messages
                messages = data.get('messages', [])
                if len(messages) < 3:
                    print(f"Warning: Line {line_num} has less than 3 messages, skipping")
                    continue
                
                # Find system, user, and assistant messages
                system_msg = None
                user_msg = None
                assistant_msg = None
                
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    if role == 'system':
                        system_msg = content
                    elif role == 'user':
                        user_msg = content
                    elif role == 'assistant':
                        assistant_msg = content
                
                # Validate we have all required messages
                if not system_msg:
                    print(f"Warning: Line {line_num} missing system message, skipping")
                    continue
                if not user_msg:
                    print(f"Warning: Line {line_num} missing user message, skipping")
                    continue
                if not assistant_msg:
                    print(f"Warning: Line {line_num} missing assistant message, skipping")
                    continue
                
                # Combine system and user messages into instruction
                instruction = f"{system_msg}\n\n{user_msg}"
                
                # Create Aliyun format entry
                converted_entry = {
                    "instruction": instruction,
                    "output": assistant_msg
                }
                
                converted_data.append(converted_entry)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Write output JSON file
    print(f"\nWriting {len(converted_data)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Conversion complete! Generated {output_file}")
    print(f"  Total entries: {len(converted_data)}")
    
    return len(converted_data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MATH dataset from JSONL to Aliyun SFT format"
    )
    parser.add_argument(
        '--train_input',
        type=str,
        default='../../MATH_train_full.jsonl',
        help='Path to training JSONL file (default: ../../MATH_train_full.jsonl)'
    )
    parser.add_argument(
        '--test_input',
        type=str,
        default='../../MATH_test_full.jsonl',
        help='Path to test JSONL file (default: ../../MATH_test_full.jsonl)'
    )
    parser.add_argument(
        '--train_output',
        type=str,
        default='train_math.json',
        help='Output path for training JSON file (default: train_math.json)'
    )
    parser.add_argument(
        '--test_output',
        type=str,
        default='test_math.json',
        help='Output path for test JSON file (default: test_math.json)'
    )
    
    args = parser.parse_args()
    
    # Convert training set
    print("=" * 60)
    print("Converting TRAINING set...")
    print("=" * 60)
    train_count = convert_math_to_aliyun_format(args.train_input, args.train_output)
    
    # Convert test set
    print("\n" + "=" * 60)
    print("Converting TEST set...")
    print("=" * 60)
    test_count = convert_math_to_aliyun_format(args.test_input, args.test_output)
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Training examples: {train_count}")
    print(f"Test examples: {test_count}")
    print(f"Total examples: {train_count + test_count}")
    print("\nOutput files:")
    print(f"  - Training: {args.train_output}")
    print(f"  - Test: {args.test_output}")


if __name__ == "__main__":
    main()

