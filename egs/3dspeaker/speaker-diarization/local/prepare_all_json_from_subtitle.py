# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will process subtitle files to generate VAD JSON files.
Usage:
    python voice_activity_detection.py --wavs $wav_list --out_file_subseg $audio_segment_json --out_file_vad $vad_json
"""

import os
import sys
import json
import argparse
import re

def time_to_seconds(time_str):
    """Convert time string (HH:MM:SS.ss) to seconds"""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def parse_subtitle_line(line):
    """Parse a subtitle line and return episode_idx, line_idx, start_time, end_time, text"""
    parts = line.strip().split('|')
    assert len(parts) >= 5, f"Subtitle line format error: {line}"
    
    episode_idx = parts[0]
    line_idx = parts[1]
    start_time_str = parts[2]
    end_time_str = parts[3]
    text = '|'.join(parts[4:])  # In case text contains '|'
    
    start_time = time_to_seconds(start_time_str)
    end_time = time_to_seconds(end_time_str)
    
    return episode_idx, line_idx, start_time, end_time, text

def normalize_path(path):
    """Normalize file paths to use the correct separator for the current OS."""
    return os.path.normpath(path)

def main():
    parser = argparse.ArgumentParser(description='Voice activity detection from subtitle files')
    parser.add_argument('--wavs', default='', type=str, help='Wav list file')
    parser.add_argument('--out_file_subseg', default='', type=str, help='Output file for individual segments')
    parser.add_argument('--out_file_vad', default='', type=str, help='Output file for merged segments')
    
    args = parser.parse_args()
    
    # Read wav.list to a list
    wavs = []
    if args.wavs.endswith('.wav'):
        # input is a wav path
        wavs.append(args.wavs)
    else:
        try:
            # input is wav list
            with open(args.wavs,'r') as f:
                wav_list = f.readlines()
        except:
            raise Exception('Input should be a wav file or a wav list.')
        for wav_path in wav_list:
            wav_path = wav_path.strip()
            wavs.append(wav_path)

    for wav_path in wav_list:
        wav_path = normalize_path(wav_path.strip())
        wavs.append(wav_path)
    
    json_dict = {}
    
    print(f'[INFO]: Start processing subtitle files...')
    
    # Process each wav file
    for wpath in wavs:
        # Convert wav path to subtitle path
        # Example: '/f/data/tv_series_plus/tv_data/the big bang theory/raw/E01.wav'
        # ->       '/f/data/tv_series_plus/tv_data/the big bang theory/speaker_text/E01.txt'
        subtitle_path = wpath.replace(os.path.join('raw', ''), os.path.join('speaker_text', '')).replace('.wav', '.txt')
        subtitle_path = normalize_path(subtitle_path)
        
        if not os.path.exists(subtitle_path):
            print(f'[WARNING]: Subtitle file not found: {subtitle_path}')
            continue
        
        # Get episode name (e.g., "E01")
        episode_name = os.path.basename(wpath).rsplit('.', 1)[0]
        
        # Read and parse subtitle file
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            parsed = parse_subtitle_line(line)
            if parsed is None:
                continue
            
            episode_idx, line_idx, start_time, end_time, text = parsed
            
            # Create segment ID
            segment_id = f"{episode_name}-{line_idx}"
            
            json_dict[segment_id] = {
                'file': wpath,
                'start': start_time,
                'stop': end_time,
            }
    
    # Save individual segments
    if not json_dict:
        raise ValueError("json_dict is empty. No valid subtitle data was processed.")
    os.makedirs(os.path.dirname(args.out_file_subseg), exist_ok=True)
    with open(args.out_file_subseg, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f'[INFO]: Audio segments info saved to {args.out_file_subseg}')
    
    # Merge segments that are close in time (within 1 second)
    merged_dict = {}
    
    # Group by file
    file_groups = {}
    for seg_id, seg_data in json_dict.items():
        file_path = seg_data['file']
        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append((seg_id, seg_data))
    
    # Process each file group
    for file_path, segments in file_groups.items():
        # Sort segments by start time
        segments.sort(key=lambda x: x[1]['start'])
        
        episode_name = os.path.basename(file_path).rsplit('.', 1)[0]
        
        i = 0
        while i < len(segments):
            current_seg_id, current_seg = segments[i]
            merge_start = current_seg['start']
            merge_end = current_seg['stop']
            merged_indices = [int(current_seg_id.split('-')[-1])]
            
            # Look for consecutive segments to merge
            j = i + 1
            while j < len(segments):
                next_seg_id, next_seg = segments[j]
                # Check if next segment starts within 1 second of current segment end
                if next_seg['start'] - merge_end <= 1.0:
                    merge_end = next_seg['stop']
                    merged_indices.append(int(next_seg_id.split('-')[-1]))
                    j += 1
                else:
                    break
            
            # Create merged segment
            if len(merged_indices) > 1:
                # Multiple segments merged
                start_idx = merged_indices[0]
                end_idx = merged_indices[-1]
                merged_seg_id = f"{episode_name}-{start_idx}-{end_idx}"
            else:
                # Single segment
                merged_seg_id = f"{episode_name}-{merged_indices[0]}"
            
            merged_dict[merged_seg_id] = {
                'file': file_path,
                'start': merge_start,
                'stop': merge_end,
            }
            
            i = j  # Move to next unprocessed segment
    
    # Save merged segments
    os.makedirs(os.path.dirname(args.out_file_vad), exist_ok=True)
    with open(args.out_file_vad, 'w') as f:
        json.dump(merged_dict, f, indent=2)
    
    print(f'[INFO]: Visual segments saved to {args.out_file_vad}')
    print(f'[INFO]: Number of audio segments: {len(json_dict)}, number of visual segments: {len(merged_dict)}')

if __name__ == '__main__':
    main()