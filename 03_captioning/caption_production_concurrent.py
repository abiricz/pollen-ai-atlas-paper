#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
High-Throughput Concurrent Captioning Script - V2

Key features:
- Uses build_prompt_morphology from lib/utils.py
- Loads masks from filtered H5 for accurate size measurements
- Async aiohttp for concurrent vLLM requests
- Resume capability with progress checkpoints

With 8 servers × 20 concurrency = 160 parallel captions!

Usage:
    python caption_production_concurrent.py \
        --h5_path /path/to/slide_filtered.h5 \
        --wsi_path /path/to/slide.tif \
        --query_image /path/to/query.png \
        --vllm_ports 11446,11447,11448 \
        --output_dir /path/to/output \
        --species "Alnus glutinosa" \
        --family "Betulaceae" \
        --pixel_um 0.22 \
        --concurrency 20 \
        --resume
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import asyncio
import aiohttp
import json
import re
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import base64
import io
from PIL import Image
import time
from tqdm import tqdm
import openslide

# Import the exact same prompt builder used in validation
from lib.utils import build_prompt_morphology, mask_diameters


class ConcurrentCaptioner:
    """Concurrent captioning matching caption_validation.py exactly"""
    
    def __init__(
        self,
        h5_path: str,
        wsi_path: str,
        query_image: str,
        vllm_ports: List[int],
        output_dir: str,
        species: str,
        family: str,
        pixel_um: float = 0.22,
        anchor_path: Optional[str] = None,
        taxon_hint: Optional[str] = None,
        concurrency: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 150,
        max_retries: int = 5,
        max_grains: Optional[int] = None,
        resume: bool = False
    ):
        self.h5_path = Path(h5_path)
        self.wsi_path = Path(wsi_path)
        self.query_image_path = Path(query_image)
        self.vllm_ports = vllm_ports
        self.output_dir = Path(output_dir)
        self.species = species
        self.family = family
        self.pixel_um = pixel_um
        self.concurrency = concurrency
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_grains = max_grains
        self.resume = resume
        
        # Output files
        self.slide_name = self.h5_path.stem.replace('_filtered', '')
        self.output_jsonl = self.output_dir / f"{self.slide_name}_captions.jsonl"
        self.progress_file = self.output_dir / f"{self.slide_name}_progress.json"
        self.summary_file = self.output_dir / f"{self.slide_name}_summary.json"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load WSI using openslide (lightweight, no ML models needed)
        print(f"[INFO] Loading WSI: {self.wsi_path}")
        self.wsi = openslide.OpenSlide(str(self.wsi_path))
        print(f"[INFO] WSI dimensions: {self.wsi.dimensions[0]} x {self.wsi.dimensions[1]}")
        
        # Load query image
        print(f"[INFO] Loading query image: {self.query_image_path}")
        self.query_img = Image.open(self.query_image_path).convert("RGB")
        
        # Load anchor and taxon_hint
        if anchor_path and os.path.exists(anchor_path):
            with open(anchor_path, 'r') as f:
                self.anchor = f.read().strip()
            print(f"[INFO] Loaded anchor: {anchor_path}")
        else:
            self.anchor = f"Scientific morphological description of {species} ({family}) pollen grain."
            print(f"[INFO] Using default anchor")
        
        self.taxon_hint = taxon_hint or f"{species} ({family})"
        
        # Detect model from vLLM server
        import requests
        first_port = vllm_ports[0]
        try:
            resp = requests.get(f"http://localhost:{first_port}/v1/models", timeout=10)
            self.model_name = resp.json()['data'][0]['id']
            print(f"[INFO] Detected model: {self.model_name}")
        except Exception as e:
            print(f"[WARN] Could not detect model: {e}")
            self.model_name = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
        
        # Pre-encode query image
        self.query_b64 = self._image_to_b64(self.query_img)
        
        # Statistics
        self.total_grains = 0
        self.completed_grains = 0
        self.failed_grains = 0
        self.start_time = None
    
    def _image_to_b64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    
    def _extract_crop_b64(self, bbox: np.ndarray, margin: int = 5) -> str:
        """Extract crop from WSI and return as base64 (same as wsi_patch_b64)"""
        x0, y0, x1, y1 = map(int, bbox.astype(float))
        w, h = x1 - x0, y1 - y0
        
        # Expand bbox with margin but clamp to slide dims
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        w = min(w + 2 * margin, self.wsi.dimensions[0] - x0)
        h = min(h + 2 * margin, self.wsi.dimensions[1] - y0)
        
        rgba = self.wsi.read_region((x0, y0), 0, (w, h)).convert("RGB")
        buf = io.BytesIO()
        rgba.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    
    def load_filtered_data(self) -> Dict[str, Any]:
        """Load all data from filtered H5"""
        print(f"[INFO] Loading filtered data from: {self.h5_path}")
        
        data = {}
        with h5py.File(self.h5_path, 'r') as f:
            r = f['results']
            data['bboxes'] = r['bbox'][:]
            data['masks'] = r['mask'][:]
            data['confidence'] = r['confidence'][:]
            data['confidence_refined'] = r['confidence_refined'][:]
            data['class_ids'] = r['class_id'][:]
            data['class_names'] = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                   for name in r['class_name'][:]]
            data['class_probs'] = r['class_prob'][:]
            data['points'] = r['point'][:]
            data['toplefts'] = r['topleft'][:]
        
        self.total_grains = len(data['bboxes'])
        print(f"[INFO] Loaded {self.total_grains} grains")
        return data
    
    def load_progress(self) -> set:
        """Load completed grain indices from progress file"""
        if not self.resume or not self.progress_file.exists():
            return set()
        
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            completed = set(progress.get('completed_indices', []))
            print(f"[INFO] Resuming: {len(completed)} grains already completed")
            return completed
        except Exception as e:
            print(f"[WARN] Could not load progress: {e}")
            return set()
    
    def save_progress(self, completed_indices: set):
        """Save progress checkpoint"""
        progress = {
            'completed_indices': sorted(list(completed_indices)),
            'total_grains': self.total_grains,
            'completed_grains': len(completed_indices),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    async def caption_grain(
        self,
        session: aiohttp.ClientSession,
        vllm_url: str,
        idx: int,
        data: Dict[str, Any]
    ) -> Optional[Dict]:
        """Caption a single grain - same logic as caption_via_vllm"""
        try:
            bbox = data['bboxes'][idx]
            mask = data['masks'][idx]
            conf = data['confidence_refined'][idx]
            
            # Compute morphology measurements (same as caption_via_vllm)
            eq_px, pol_px = mask_diameters(mask)
            pol_um = pol_px * self.pixel_um
            eq_um = eq_px * self.pixel_um
            area_um2 = np.count_nonzero(mask) * self.pixel_um**2
            
            # Extract images
            crop_b64 = self._extract_crop_b64(bbox)
            
            # Build prompt (same as validation)
            prompt_body = build_prompt_morphology(
                pol_um,
                eq_um,
                area_um2,
                taxon_hint=self.taxon_hint,
                confidence=conf,
                anchor=self.anchor,
            )
            
            # Build messages (same structure as caption_via_vllm)
            messages = [
                {
                    "role": "system",
                    "content": "You are a palynologist image-analysis assistant, describing pollen morphology from microscopic images.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_body},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.query_b64}"}},
                    ],
                },
            ]
            
            # API request
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Disable thinking mode for models with enable_thinking in chat template
            if any(tag in self.model_name.lower() for tag in ['qwen3.5', 'qwen3.6', 'qwen3_5', 'gemma-4', 'gemma4', 'gemma_4']):
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            
            async with session.post(
                f"{vllm_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"[ERROR] vLLM error {response.status}: {error_text[:200]}")
                    return None
                
                result = await response.json()
                caption_text = result['choices'][0]['message']['content'].strip()
                
                # Strip any residual <think>...</think> blocks from output
                caption_text = re.sub(r'<think>.*?</think>\s*', '', caption_text, flags=re.DOTALL).strip()
                
                # Build output record (matching validation format)
                record = {
                    "id": f"{self.slide_name}_{idx:06d}",
                    "species": self.species,
                    "family": self.family,
                    "source_slide": self.slide_name,
                    "caption_model": self.model_name,
                    "bbox": bbox.tolist(),
                    "similarity_confidence": float(data['confidence'][idx]),
                    "confidence_refined": float(conf),
                    "class_id": int(data['class_ids'][idx]),
                    "class_name": str(data['class_names'][idx]),
                    "class_prob": float(data['class_probs'][idx]),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "mask_index": int(idx),
                    "sam_point": data['points'][idx].tolist(),
                    "wsi_topleft": data['toplefts'][idx].tolist(),
                    "prompt": prompt_body,
                    "caption": caption_text,
                    "polar_diameter_um": pol_um,
                    "equatorial_diameter_um": eq_um,
                    "area_um2": area_um2,
                    "timestamp": datetime.now().isoformat()
                }
                
                return record
                
        except asyncio.TimeoutError:
            # Return None with error info for retry logic
            return None
        except Exception as e:
            # Return None with error info for retry logic  
            return None
    
    async def caption_grain_with_retry(
        self,
        session: aiohttp.ClientSession,
        vllm_urls: List[str],
        idx: int,
        data: Dict[str, Any],
        max_retries: int = 3,
        start_server_idx: int = 0
    ) -> Optional[Dict]:
        """Caption a grain with retry logic - tries different servers on failure"""
        last_error = None
        
        for attempt in range(max_retries):
            # Start from assigned server, rotate on retry for load balancing
            server_idx = (start_server_idx + attempt) % len(vllm_urls)
            vllm_url = vllm_urls[server_idx]
            
            try:
                record = await self.caption_grain(session, vllm_url, idx, data)
                if record is not None:
                    if attempt > 0:
                        record['retry_attempts'] = attempt
                    return record
                    
                # If None returned, try next server
                last_error = "vLLM returned error"
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Backoff
                    
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Backoff
        
        print(f"[ERROR] Grain {idx} failed after {max_retries} attempts: {last_error}")
        return None
    
    async def process_all(
        self,
        data: Dict[str, Any],
        completed_indices: set
    ):
        """Process all grains with bounded concurrency"""
        total_concurrency = len(self.vllm_ports) * self.concurrency
        print(f"\n[INFO] Starting concurrent captioning")
        print(f"       vLLM servers: {len(self.vllm_ports)}")
        print(f"       Concurrency per server: {self.concurrency}")
        print(f"       Total parallel requests: {total_concurrency}")
        
        # Filter out completed grains
        pending_indices = [i for i in range(self.total_grains) if i not in completed_indices]
        
        # Apply max_grains limit if specified (for benchmarking)
        if self.max_grains is not None and len(pending_indices) > self.max_grains:
            pending_indices = pending_indices[:self.max_grains]
            print(f"       [BENCHMARK] Limited to {self.max_grains} grains")
        
        print(f"       Grains to process: {len(pending_indices)}")
        
        if not pending_indices:
            print("[INFO] All grains already completed!")
            return
        
        # Single global semaphore for bounded concurrency
        semaphore = asyncio.Semaphore(total_concurrency)
        
        # Build vLLM URLs
        vllm_urls = [f"http://localhost:{port}/v1" for port in self.vllm_ports]
        
        # Open JSONL for appending
        output_file = open(self.output_jsonl, 'a')
        
        # Progress tracking
        pbar = tqdm(total=len(pending_indices), desc="Captioning", unit="grains")
        completed_count = len(completed_indices)
        retried_count = 0  # Track retries
        
        self.start_time = time.time()
        
        # Results lock for thread-safe writing
        results_lock = asyncio.Lock()
        
        async def process_grain(idx: int, server_idx: int) -> None:
            """Process a single grain with bounded concurrency and retry logic"""
            nonlocal completed_count, retried_count
            
            async with semaphore:
                try:
                    # Use retry wrapper with assigned server for load balancing
                    record = await self.caption_grain_with_retry(
                        session, vllm_urls, idx, data, 
                        max_retries=self.max_retries,
                        start_server_idx=server_idx
                    )
                    
                    async with results_lock:
                        if record:
                            # Track retries
                            if 'retry_attempts' in record:
                                retried_count += 1
                            
                            # Write to JSONL
                            output_file.write(json.dumps(record) + '\n')
                            output_file.flush()
                            
                            # Update progress
                            completed_indices.add(idx)
                            completed_count += 1
                            
                            # Save checkpoint every 100 grains
                            if completed_count % 100 == 0:
                                self.save_progress(completed_indices)
                        else:
                            self.failed_grains += 1
                        
                        pbar.update(1)
                        
                        # Update stats
                        elapsed = time.time() - self.start_time
                        rate = pbar.n / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({
                            'rate': f'{rate:.1f}/s',
                            'failed': self.failed_grains,
                            'retried': retried_count
                        })
                        
                except Exception as e:
                    async with results_lock:
                        print(f"[ERROR] Task error for grain {idx}: {e}")
                        self.failed_grains += 1
                        pbar.update(1)
        
        async with aiohttp.ClientSession() as session:
            # Process in batches to avoid memory issues
            batch_size = 10000
            
            for batch_start in range(0, len(pending_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(pending_indices))
                batch_indices = pending_indices[batch_start:batch_end]
                
                if batch_start > 0:
                    print(f"\n[INFO] Processing batch {batch_start//batch_size + 1}: grains {batch_start}-{batch_end}")
                
                # Create tasks for this batch
                tasks = []
                for i, idx in enumerate(batch_indices):
                    server_idx = (batch_start + i) % len(vllm_urls)
                    tasks.append(process_grain(idx, server_idx))
                
                # Wait for all tasks in batch to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Checkpoint after each batch
                self.save_progress(completed_indices)
        
        pbar.close()
        output_file.close()
        
        # Final checkpoint
        self.save_progress(completed_indices)
        
        # Print summary
        elapsed = time.time() - self.start_time
        print(f"\n[INFO] Captioning complete!")
        print(f"       Total time: {elapsed/60:.1f} minutes")
        print(f"       Completed: {completed_count}")
        print(f"       Failed: {self.failed_grains}")
        print(f"       Rate: {completed_count/elapsed:.2f} grains/second")
    
    def save_summary(self):
        """Save summary statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            'slide': self.slide_name,
            'species': self.species,
            'family': self.family,
            'total_grains': self.total_grains,
            'completed_grains': self.completed_grains,
            'failed_grains': self.failed_grains,
            'elapsed_seconds': elapsed,
            'rate_grains_per_second': self.completed_grains / elapsed if elapsed > 0 else 0,
            'vllm_servers': len(self.vllm_ports),
            'concurrency_per_server': self.concurrency,
            'total_concurrency': len(self.vllm_ports) * self.concurrency,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[INFO] Summary saved: {self.summary_file}")
    
    async def run(self):
        """Main execution"""
        print(f"\n{'='*70}")
        print(f"CONCURRENT PRODUCTION CAPTIONING V2")
        print(f"{'='*70}")
        print(f"Slide: {self.slide_name}")
        print(f"Species: {self.species}")
        print(f"Family: {self.family}")
        print(f"Pixel size: {self.pixel_um} µm")
        print(f"vLLM servers: {len(self.vllm_ports)}")
        print(f"Concurrency: {self.concurrency} per server")
        print(f"Total parallel: {len(self.vllm_ports) * self.concurrency}")
        print(f"{'='*70}\n")
        
        # Load data
        data = self.load_filtered_data()
        
        # Load progress
        completed_indices = self.load_progress()
        
        # Process
        await self.process_all(data, completed_indices)
        
        # Save summary
        self.completed_grains = len(completed_indices)
        self.save_summary()
        
        print(f"\n[SUCCESS] Output: {self.output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="High-throughput concurrent captioning V2 - uses exact validation prompts"
    )
    parser.add_argument('--h5_path', type=str, required=True,
                       help='Path to filtered H5 file')
    parser.add_argument('--wsi_path', type=str, required=True,
                       help='Path to WSI file')
    parser.add_argument('--query_image', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--vllm_ports', type=str, required=True,
                       help='Comma-separated vLLM ports (e.g., 11446,11447,11448)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--species', type=str, required=True,
                       help='Species name')
    parser.add_argument('--family', type=str, required=True,
                       help='Family name')
    parser.add_argument('--pixel_um', type=float, default=None,
                       help='Pixel size in micrometers (default: from config or 0.22)')
    parser.add_argument('--pixel_config', type=str, default=None,
                       help='Path to pixel_config.yaml for per-slide pixel_um values')
    parser.add_argument('--anchor', type=str, default=None,
                       help='Path to anchor text file')
    parser.add_argument('--taxon_hint', type=str, default=None,
                       help='Descriptive one-liner for pollen (from *_hint.txt), e.g., '
                            '"small to medium spheroidal grain, tricolporate, dense spines, 16–28 µm"')
    parser.add_argument('--concurrency', type=int, default=20,
                       help='Concurrent requests per vLLM server (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=150,
                       help='Max tokens per caption')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Max retry attempts per failed grain (default: 3)')
    parser.add_argument('--max_grains', type=int, default=None,
                       help='Max grains to process (for benchmarking, default: all)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Parse ports
    ports = [int(p.strip()) for p in args.vllm_ports.split(',')]
    
    # Resolve pixel_um: CLI > config file > default
    slide_name = Path(args.h5_path).stem.replace('_filtered', '')
    pixel_um = args.pixel_um  # CLI value (may be None)
    
    if pixel_um is None and args.pixel_config:
        # Try to load from config file
        import yaml
        try:
            with open(args.pixel_config) as f:
                pixel_config = yaml.safe_load(f)
            
            # Check per-slide override first
            slides = pixel_config.get('slides', {})
            if slide_name in slides:
                pixel_um = slides[slide_name]
                print(f"[Config] pixel_um={pixel_um} (from slides.{slide_name})")
            else:
                # Try dataset default based on slide prefix
                defaults = pixel_config.get('defaults', {})
                if slide_name.startswith('hun_'):
                    pixel_um = defaults.get('hungarian', 0.22)
                    print(f"[Config] pixel_um={pixel_um} (from defaults.hungarian)")
                elif slide_name.startswith('med_'):
                    pixel_um = defaults.get('mediterranean', 0.22)
                    print(f"[Config] pixel_um={pixel_um} (from defaults.mediterranean)")
                elif slide_name.startswith(('swe_', 'sw_')):
                    pixel_um = defaults.get('swedish', 0.22)
                    print(f"[Config] pixel_um={pixel_um} (from defaults.swedish)")
                else:
                    pixel_um = defaults.get('french', 0.242797)
                    print(f"[Config] pixel_um={pixel_um} (from defaults.french)")
        except Exception as e:
            print(f"[Warning] Could not read pixel_config: {e}")
            pixel_um = None
    
    # Final fallback
    if pixel_um is None:
        pixel_um = 0.22
        print(f"[Default] pixel_um={pixel_um}")
    
    # Create captioner
    captioner = ConcurrentCaptioner(
        h5_path=args.h5_path,
        wsi_path=args.wsi_path,
        query_image=args.query_image,
        vllm_ports=ports,
        output_dir=args.output_dir,
        species=args.species,
        family=args.family,
        pixel_um=pixel_um,
        anchor_path=args.anchor,
        taxon_hint=args.taxon_hint,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_grains=args.max_grains,
        resume=args.resume
    )
    
    # Run async
    asyncio.run(captioner.run())


if __name__ == '__main__':
    main()
