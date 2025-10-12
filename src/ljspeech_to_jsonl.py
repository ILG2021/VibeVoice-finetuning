#!/usr/bin/env python3
"""
LJSpeech to JSONL Converter

将LJSpeech格式的数据集转换为JSONL格式
LJSpeech格式: metadata.csv文件，每行格式为 "filename|text"
JSONL格式: 每行是一个JSON对象，包含audio和text字段
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional


def convert_ljspeech_to_jsonl(
    metadata_path: str,
    output_path: Optional[str] = "prompts.jsonl",
    audio_dir: Optional[str] = None,
    audio_extension: str = ".wav",
    speaker: str = '',
) -> None:
    """
    将LJSpeech格式转换为JSONL格式
    
    Args:
        metadata_path: metadata.csv文件路径
        output_path: 输出的JSONL文件路径
        audio_dir: 音频文件目录，如果为None则使用相对路径
        audio_extension: 音频文件扩展名
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_count = 0
    with open(metadata_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            
            # LJSpeech metadata.csv格式通常是用|分隔的
            csv_reader = csv.reader(infile, delimiter='|')
            
            for row in csv_reader:
                if len(row) < 2:
                    print(f"Warning: Skipping invalid row: {row}")
                    continue
                
                filename = row[0].strip()
                text = speaker + ': ' + row[1].strip()
                
                # 构建音频文件路径
                if audio_dir:
                    # 使用指定的音频目录
                    audio_file = os.path.join(audio_dir, filename + audio_extension)
                else:
                    # 使用相对路径
                    parent = Path(metadata_path).parent  # Get parent directory
                    audio_file = (parent / 'wavs' / filename).as_posix()

                # 创建JSONL条目
                jsonl_entry = {
                    "audio": audio_file,
                    "text": text,
                }

                # 写入JSONL文件
                outfile.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
                processed_count += 1
    
    print(f"Successfully converted {processed_count} entries from {metadata_path} to {output_path}")


def validate_jsonl_file(jsonl_path: str, audio_dir: Optional[str] = None) -> None:
    """
    验证生成的JSONL文件
    
    Args:
        jsonl_path: JSONL文件路径
        audio_dir: 音频文件目录，用于验证文件是否存在
    """
    print(f"\nValidating JSONL file: {jsonl_path}")
    
    valid_count = 0
    invalid_count = 0
    missing_audio_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                entry = json.loads(line.strip())
                
                # 检查必需字段
                if 'audio' not in entry or 'text' not in entry:
                    print(f"Line {line_num}: Missing required fields")
                    invalid_count += 1
                    continue
                
                # 如果提供了音频目录，检查音频文件是否存在
                if audio_dir:
                    audio_path = os.path.join(audio_dir, os.path.basename(entry['audio']))
                    if not os.path.exists(audio_path):
                        missing_audio_count += 1
                
                valid_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
                invalid_count += 1
    
    print(f"Validation results:")
    print(f"  Valid entries: {valid_count}")
    print(f"  Invalid entries: {invalid_count}")
    if audio_dir:
        print(f"  Missing audio files: {missing_audio_count}")


def main():
    parser = argparse.ArgumentParser(description='Convert LJSpeech format to JSONL')
    parser.add_argument('metadata', help='Path to metadata.csv file')
    parser.add_argument('--output', help='Output JSONL file path')
    parser.add_argument('--audio-dir', help='Audio files directory')
    parser.add_argument('--audio-ext', default='.wav', help='Audio file extension (default: .wav)')
    parser.add_argument('--speaker', default='Speaker 0', help='Speaker for text (default: Speaker 0)')
    parser.add_argument('--validate', action='store_true', help='Validate the output JSONL file')
    
    args = parser.parse_args()

    # 转换文件
    convert_ljspeech_to_jsonl(
        metadata_path=args.metadata,
        output_path=args.output,
        audio_dir=args.audio_dir,
        audio_extension=args.audio_ext,
        speaker=args.speaker,
    )

    # 如果需要，验证输出文件
    if args.validate:
        validate_jsonl_file(args.output, args.audio_dir)

    
    return 0


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) == 1:
        # 如果没有提供参数，显示示例用法
        print("LJSpeech to JSONL Converter")
        print("\nUsage examples:")
        print("python ljspeech_to_jsonl.py metadata.csv output.jsonl")
        print("python ljspeech_to_jsonl.py metadata.csv output.jsonl --audio-dir /path/to/wavs")
        print("python ljspeech_to_jsonl.py metadata.csv output.jsonl --validate")
        print("\nFor full help: python ljspeech_to_jsonl.py --help")
    else:
        exit(main())

