# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm
import openslide
from scipy.ndimage import gaussian_filter
from collections import Counter
import math, sys
import heapq

import logging
# Suppress everything below WARNING for all loggers
logging.basicConfig(level=logging.WARNING)
# Optional: get specific loggers by name if needed
logging.getLogger("sam").setLevel(logging.ERROR)  # or whichever module logs this

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from lib.model import model_selection, ViTBackbone
from lib.loader import WSITileDataset


class WSIPipeline:
    def __init__(self,
                 wsi_path,
                 device,
                 patch_size=518,
                 overlap=0,
                 batch_size=8,
                 sam2_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                 sam2_ckpt="./checkpoints/sam2.1_hiera_large.pt",
                 vit_ckpt="vit_small_lvd_finetuned_100epochs.pth",
                 vit_name="vit_small_patch14_dinov2.lvd142m",
                 divisor_for_sam_query=2, # default, point to middle
                 sam_multi_point_query=False # whether to add multi point query for query img segmentation
                 ):
        
        print("[Init] Initializing WSI pipeline...")
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.sam2_cfg = sam2_cfg
        self.sam2_ckpt = sam2_ckpt
        self.vit_ckpt = vit_ckpt
        self.vit_name = vit_name
        self.device = device
        self.divisor_for_sam_query = divisor_for_sam_query
        self.sam_multi_point_query = sam_multi_point_query

        self._init_wsi()
        self._build_vit()
        self._build_sam2()
        self._build_dataloader()
        print("[Init] Pipeline ready.")

    def _init_wsi(self):
        print("[WSI] Loading WSI from disk...")
        self.wsi = openslide.OpenSlide(self.wsi_path)
        self.full_w, self.full_h = self.wsi.dimensions
        print(f"[WSI] Dimensions: {self.full_w} x {self.full_h}")

    def _build_vit(self):
        print("[ViT] Loading ViT backbone and weights...")
        vit_model, vit_input_size = model_selection(self.vit_name)
        if self.vit_ckpt is not None:
            state = torch.load(self.vit_ckpt, map_location='cpu')
            vit_model.load_state_dict(state, strict=False)
            print('[ViT] Successfully loaded weights')
        self.vit = ViTBackbone(vit_model, model_img_input_size=vit_input_size, conv_reduce=False)
        self.vit = self.vit.to(self.device).eval()
        self.transform_vit = transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        print("[ViT] Model ready. Clearing GPU cache.")

    def _build_sam2(self):
        print("[SAM2] Loading SAM2 model and predictor...")
        sam2_model = build_sam2(self.sam2_cfg, self.sam2_ckpt).to(self.device).eval()
        # use autocast for the heavy image encoder while keeping decoder numerics stable
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("[SAM2] Predictor ready. Clearing GPU cache.")

    def _build_dataloader(self):
        print("[Data] Building dataset and dataloader...")
        transform_dummy = transforms.Compose([transforms.ToTensor()])
        self.dataset = WSITileDataset(
            wsi_path=self.wsi_path,
            transform=transform_dummy,
            tile_size=self.patch_size,
            overlap=self.overlap
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.custom_collate_fn,
            num_workers=4
        )
        print(f"[Data] {len(self.dataset)} tiles loaded.")
    
    def _topk_pca_vector(self, tokens: np.ndarray,
                         k: int = 3,
                         eps: float = 1e-10) -> np.ndarray:
        """
        Collapse the top-k principal components of `tokens` into a single D-vector.
    
        1) Center tokens.
        2) Compute SVD → singular values S and Vt (shape (D,D)).
        3) Take the first k rows of Vt → the top-k PCs.
        4) Align each PC so that it points roughly toward the mean token-vector.
        5) Weight each PC by its singular value and sum.
        6) Normalize to unit-length.
    
        Args:
            tokens: (N, D) array of feature-vectors.
            k     : how many PCs to include (e.g. 3).
            eps   : numeric safeguard for zero-division.
    
        Returns:
            comb_norm : (D,) unit-vector combining the top k PCs.
        """
        # 1) center
        mean_vec = tokens.mean(axis=0, keepdims=True)   # (1, D)
        X        = tokens - mean_vec                    # (N, D)
        
        # 2) SVD
        #    shapes: U:(N,D), S:(min(N,D),), Vt:(D,D)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # 3) grab top-k PCs
        pcs = Vt[:k]    # (k, D)
        
        # 4) align signs
        mv = mean_vec.ravel()
        for i in range(k):
            if np.dot(pcs[i], mv) < 0:
                pcs[i] = -pcs[i]
        
        # 5) weighted sum by singular values
        #    S[0] weights PC1, S[1] weights PC2, …
        eig = S**2                           # eigenvalues ∝ variance
        comb = np.sum(pcs * eig[:k, None], axis=0)
        comb /= np.linalg.norm(comb) + eps
        #comb = np.sum(pcs * S[:k, None], axis=0)  # (D,)
    
        # 6) normalize
        comb_norm = comb / (np.linalg.norm(comb) + eps)

        # 7) explained-variance ratio
        total_var   = np.sum(S**2)
        var_fraction = np.sum(S[:k]**2) / (total_var + eps)
        
        return comb_norm, var_fraction
        
    def compute_vit_embeddings(self):
        print("[ViT] Computing embeddings for all tiles...")
        
        # 1) Figure out total tokens and dims
        # We know each tile produces Hf*Wf tokens, and there are len(self.dataset) tiles:
        # We’ll peek at one batch to get Hf, Wf, C:
        sample_batch = next(iter(self.dataloader))
        with torch.no_grad():
            emb_map = self.vit(sample_batch['tiles_tensors'].to(self.device))
        _, C, Hf, Wf = emb_map.shape
        emb_map = None
        torch.cuda.empty_cache()
    
        total_tiles = len(self.dataset)
        tokens_per_tile = Hf * Wf
        total_tokens = total_tiles * tokens_per_tile
    
        # 2) Pre-allocate big arrays
        self.embeddings = np.empty((total_tokens, C), dtype=np.float32)
        self.coords     = np.empty((total_tokens,   2), dtype=np.int32)
    
        # 3) Fill them batch by batch
        idx = 0
        for batch in tqdm(self.dataloader, desc="ViT Embedding"):
            imgs = batch['tiles_tensors'].to(self.device)
            with torch.no_grad():
                emb_map = self.vit(imgs)  # (B, C, Hf, Wf)
            B, _, Hf, Wf = emb_map.shape
    
            # flatten B×Hf×Wf → (B*Hf*Wf, C)
            tokens = emb_map.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            n = tokens.shape[0]
            self.embeddings[idx:idx+n] = tokens
    
            # build coords for these tokens
            coords_chunk = []
            stride = self.patch_size // Wf  # should be 14
            for (x0, y0) in batch['coords']:
                for dy in range(Hf):
                    for dx in range(Wf):
                        px = x0 + dx * stride
                        py = y0 + dy * stride
                        coords_chunk.append((px, py))
            self.coords[idx:idx+n] = np.array(coords_chunk, dtype=np.int32)
    
            idx += n
    
            # cleanup
            del emb_map, imgs
            torch.cuda.empty_cache()
            
        print(f"[ViT] Embeddings computed: {total_tokens} tokens.")

    
    def compute_similarity_chunked(self, q_norm, chunk_size=200_000, eps=1e-10, smooth_sigma=1, vit_patch_size=14):
        # Determine output grid size from token coords
        max_x = (self.coords[:, 0].max() // vit_patch_size) + 1
        max_y = (self.coords[:, 1].max() // vit_patch_size) + 1
        sim_map = np.full((max_y, max_x), -np.inf, dtype=np.float64)
    
        N, D = self.embeddings.shape
    
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            emb_chunk = self.embeddings[start:end]
            coords_chunk = self.coords[start:end]
    
            # Normalize
            norms = np.linalg.norm(emb_chunk, axis=1, keepdims=True) + eps
            emb_norm = emb_chunk / norms
    
            # Cosine similarity
            sim_chunk = emb_norm @ q_norm  # (chunk_size,)
    
            # Map coords to grid
            x_idx = (coords_chunk[:, 0] // vit_patch_size).astype(int)
            y_idx = (coords_chunk[:, 1] // vit_patch_size).astype(int)
    
            # Scatter similarities into sim_map
            sim_map[y_idx, x_idx] = sim_chunk  # this assumes no duplicate tokens per index
    
        # Normalize final sim_map to [0, 1]
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + eps)
        
        # (Optional) spatially smooth away grid-artifacts
        sim_map = gaussian_filter(sim_map, sigma=smooth_sigma)

        # Normalize final sim_map to [0, 1]
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + eps)
        sim_map /= sim_map.sum() # sim_norm now sums to 1, so it’s a valid probability distribution for Shannon entropy
        return sim_map

    def compute_similarity_chunked_gpu(self,
        q_norm:       np.ndarray,   # (D,)   float32 unit vector
        vit_stride=14,
        chunk_size=100_000,         # bump this up until you fill your 24 GB
        eps=1e-10,
        smooth_sigma=1
    ):
        # 1) Precompute grid dims
        ys = self.coords[:,1] // vit_stride
        xs = self.coords[:,0] // vit_stride
        H, W = int(ys.max())+1, int(xs.max())+1
    
        # 2) Allocate sim_map on GPU in full precision
        sim_map = torch.full((H, W), float("-inf"),
                             device=self.device, dtype=torch.float32)
    
        # 3) Upload & normalize query once in fp16
        t_q = torch.from_numpy(q_norm).to(device=self.device, dtype=torch.float16)
        t_q = F.normalize(t_q, p=2, dim=0)  # (D,)
    
        N, D = self.embeddings.shape
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
    
            # a) slice CPU chunk
            emb_chunk = self.embeddings[start:end]             # (M, D), CPU
            coord_chunk = self.coords[start:end]               # (M, 2), CPU
    
            # b) move & normalize in fp16 (non_blocking if you pinned your CPU arrays)
            t_emb = (torch.from_numpy(emb_chunk)
                            .to(device=self.device, dtype=torch.float16, non_blocking=True))
            t_emb = F.normalize(t_emb, p=2, dim=1)        # (M, D)
    
            # c) dot-product  → float16 mat-vec is super fast
            t_sim = (t_emb @ t_q).to(dtype=torch.float32) # (M,)
    
            # d) get token-grid indices on GPU
            t_y = torch.from_numpy(coord_chunk[:,1] // vit_stride)\
                       .to(device=self.device)
            t_x = torch.from_numpy(coord_chunk[:,0] // vit_stride)\
                       .to(device=self.device)
            # flatten index
            flat_idx = t_y * W + t_x                     # (M,)
    
            # e) scatter into sim_map.view(-1) on GPU
            flat_map = sim_map.view(-1)
            flat_map.index_put_((flat_idx,), t_sim, accumulate=False)
    
            # cleanup for next chunk
            del t_emb, t_sim, t_y, t_x, flat_idx
            # (no torch.cuda.empty_cache())
    
        # 4) bring back to CPU and normalize
        sim_map = sim_map.cpu().numpy()
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + eps)
        
        # (Optional) spatially smooth away grid-artifacts
        sim_map = gaussian_filter(sim_map, sigma=smooth_sigma)

        # Normalize final sim_map to [0, 1]
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + eps)
        sim_map /= sim_map.sum() # sim_norm now sums to 1, so it’s a valid probability distribution for Shannon entropy
        return sim_map
    
    
    def compute_token_level_similarity(self, query_image_path, mode="cpu_mean", vit_patch_size=14):
        """
        Compute token-level similarity map.
        Modes:
          - "cpu_mean": collapse query to mean vector, compute on CPU.
          - "gpu_conv": perform convolutional cosine-sim on GPU (may OOM for large slides).
        """
        print(f"[Sim] Computing token-level similarity for query: {query_image_path} in mode={mode}...")

        # 1) Run ViT on GPU to get query tokens, then move to CPU
        query_img = Image.open(query_image_path).convert("RGB")
        w, h = query_img.size

        # save for later use
        self.query_img = query_img
        self.query_area = w * h
        
        pad_r = self.patch_size - w
        pad_b = self.patch_size - h
        query_pad = ImageOps.expand(query_img, (0, 0, pad_r, pad_b), fill=(255,255,255))
        
        if self.sam_multi_point_query:                                                                             ##### MODIFIED !!!
            # ---- positive grid around the centre ---------------------
            #   fractions of width / height → 3×3 = 9 positive points
            #print('[SAM] multi point query.')
            grid_frac = [0.33, 0.50, 0.67]
            pos_pts   = [(int(w * fx), int(h * fy))            # (x, y)
                         for fy in grid_frac for fx in grid_frac]
        
            # ---- optional negative ring (discourages merge) ----------
            ring_frac = [0.02, 0.98]                           # near corners
            neg_pts   = [(int(w * fx), int(h * fy))
                         for fx in ring_frac for fy in ring_frac]
        
            point_coords = np.array(pos_pts + neg_pts, np.float32)
            point_labels = np.concatenate([
                np.ones (len(pos_pts), dtype=np.int32),        # 1 = positive
                np.zeros(len(neg_pts), dtype=np.int32)         # 0 = negative
            ])
        else:
            # single positive at the geometric centre
            point_coords = np.array([[w // self.divisor_for_sam_query, h // self.divisor_for_sam_query]], np.float32)  # (1,2)
            point_labels = np.array([1], dtype=np.int32)
        
        # Quick SAM2 single-click segmentation (centre of image)
        #cx, cy = w//self.divisor_for_sam_query, h//self.divisor_for_sam_query
        
        dev_type = self.device.type if isinstance(self.device, torch.device) else "cpu"
        with torch.no_grad(), \
            torch.amp.autocast(device_type=dev_type, dtype=torch.float16):
            self.sam2_predictor.set_image(query_pad)
            masks, _, _ = self.sam2_predictor.predict(
                point_coords   = point_coords, # np.array([[cx, cy]], np.float32),
                point_labels   = point_labels, # np.array([1],    np.int32),
                multimask_output=False
            )
        fg_mask = masks[0]
        
        q_tensor = self.transform_vit(query_pad).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_feat_map = self.vit(q_tensor)  # (1, C, Hq, Wq)
        tok_map = q_feat_map.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (Hq, Wq, D)
        # Clean GPU
        del q_tensor, q_feat_map
        torch.cuda.empty_cache()

        tok_idx = np.unique( np.argwhere( fg_mask ) // vit_patch_size, axis=0 )
        tok_y, tok_x = tok_idx[:,0], tok_idx[:,1]
        self.query_tokens = tok_map[tok_y, tok_x]
        self.query_mask = fg_mask
        qy = np.argwhere(self.query_mask)    # list of (row, col) in full-res pixels
        (y0, x0), (y1, x1) = qy.min(0), qy.max(0)
        self.query_bbox_area_px = (y1 - y0 + 1) * (x1 - x0 + 1)
        self.query_mask_area_px = self.query_mask.sum()  # now also in pixels
        
        # Build and normalize full feature map on CPU
        eps = 1e-10
        max_x = (self.coords[:,0].max() // vit_patch_size) + 1
        max_y = (self.coords[:,1].max() // vit_patch_size) + 1

        if mode in ["gpu_mean"]:
            q_vec = self.query_tokens.mean(axis=0)
            q_norm = q_vec / (np.linalg.norm(q_vec) + eps)
            self.query_vec  = q_vec    # raw, for confidence reference
            self.query_norm = q_norm   # unit, for similarity-map construction
            self.sim_map = self.compute_similarity_chunked_gpu(q_norm)
            print("[Sim] GPU chunked mean-vector similarity computed.")

        elif mode in ["cpu_mean"]:
            q_vec = self.query_tokens.mean(axis=0)
            q_norm = q_vec / (np.linalg.norm(q_vec) + eps)
            self.query_vec  = q_vec    # raw, for confidence reference
            self.query_norm = q_norm   # unit, for similarity-map construction
            self.sim_map = self.compute_similarity_chunked(q_norm)
            print("[Sim] CPU chunked mean-vector similarity computed.")

        else:
            raise ValueError(f"Unknown mode '{mode}' for similarity computation.")
        
    def compute_augmented_query_tokens(self, query_image_path, 
                                         n_shifts=8,
                                         vit_patch = 14,
                                         eps: float = 1e-10):

        query_img = Image.open(query_image_path).convert("RGB")
        w, h = query_img.size

        # save for later use
        self.query_img = query_img
        self.query_img_area = w * h

        augment = transforms.Compose([
            # random rotation among multiples of 90°
            transforms.RandomChoice([
                transforms.Lambda(lambda im: im),
                transforms.Lambda(lambda im: im.rotate( 90, fillcolor=(255,255,255))),
                transforms.Lambda(lambda im: im.rotate(180, fillcolor=(255,255,255))),
                transforms.Lambda(lambda im: im.rotate(270, fillcolor=(255,255,255))),
            ]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.2),  # 20% chance to remove color
            transforms.RandomResizedCrop(size=max(query_img.size), scale=(0.75, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.0)),
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
            transforms.Resize(query_img.size)
        ])
        # augmentations may crash at some point if object is shifted out of canvas -> todo 
        
        patch_size = self.patch_size
        
        max_x = patch_size - query_img.width
        max_y = patch_size - query_img.height
        xs = np.linspace(0, max_x, n_shifts, dtype=int)
        ys = np.linspace(0, max_y, n_shifts, dtype=int)
        
        shifted_query_tokens = []
        
        for i, y_off in enumerate(ys):
            for j, x_off in enumerate(xs):
        
                # 1) augment the **query** crop first
                if i == 0 and j == 0:
                    aug_img = query_img # skip augmentations for the first image
                    print('[Sim] First element is augmented with identity.')
                else:
                    aug_img = augment(query_img)
        
                # 2) build a blank canvas and paste the augmented crop
                canvas = Image.new('RGB', (patch_size, patch_size), (255,255,255))
                canvas.paste(aug_img, (x_off, y_off))
        
                # 3) SAM2 single-click at the *shifted+augmented* center
                if self.sam_multi_point_query:
                    #print('[SAM] multi point query.')
                    # shift every positive / negative point by (x_off, y_off)
                    grid_frac = [0.33, 0.50, 0.67]
                    pos_pts = [(x_off + int(w * fx),            #  <-- x_off added
                                y_off + int(h * fy))            #  <-- y_off added
                               for fy in grid_frac for fx in grid_frac]
                
                    ring_frac = [0.02, 0.98]
                    neg_pts = [(x_off + int(w * fx),            #  <-- x_off added
                                y_off + int(h * fy))            #  <-- y_off added
                               for fx in ring_frac for fy in ring_frac]
                
                    point_coords = np.array(pos_pts + neg_pts, np.float32)
                    point_labels = np.concatenate([
                        np.ones (len(pos_pts), dtype=np.int32),
                        np.zeros(len(neg_pts), dtype=np.int32)
                    ])
                else:
                    # centre point also needs the same shift
                    cx = x_off + w // 2                          #  <-- x_off added
                    cy = y_off + h // 2                          #  <-- y_off added
                    point_coords = np.array([[cx, cy]], np.float32)
                    point_labels = np.array([1], dtype=np.int32)
                
                #cx = x_off + aug_img.width//self.divisor_for_sam_query # this is for pointing to the exact middle of the query image -> may be unstable sometimes, due to surface patterns
                #cy = y_off + aug_img.height//self.divisor_for_sam_query
                
                canvas_np = np.asarray(canvas, np.float32) / 255.0

                dev_type = self.device.type if isinstance(self.device, torch.device) else "cpu"
                with torch.no_grad(), \
                    torch.amp.autocast(device_type=dev_type, dtype=torch.float16):
                    self.sam2_predictor.set_image(canvas_np)
                    masks, _, _ = self.sam2_predictor.predict(
                        point_coords   = point_coords, # np.array([[cx, cy]], np.float32),
                        point_labels   = point_labels, # np.array([1], np.int32),
                        multimask_output=False
                    )
                mask = masks[0] # H×W boolean 
        
                # 4) ViT forward to get token-feature map
                qt = self.transform_vit(canvas).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    fmap = self.vit(qt)        # (1, C, Hq, Wq)
                tok_map = fmap.squeeze(0).permute(1,2,0).cpu().numpy()  # (Hq, Wq, D)
        
                # 5) select tokens inside the mask
                coords   = np.argwhere(mask)           # list of (y, x)
                tok_idxs = np.unique(coords // vit_patch, axis=0)
                ty, tx   = tok_idxs[:,0], tok_idxs[:,1]
        
                # 6) collect those embeddings
                q_tokens = tok_map[ty, tx]
                shifted_query_tokens.append({
                    'mask': mask,
                    'coords': coords,
                    'shift':   (x_off, y_off),
                    'tokens':  q_tokens,
                    'indices': tok_idxs
                })
        
        return shifted_query_tokens

    def compute_token_level_similarity_augmented(self, query_image_path,
                                                 sim_map_creation_mode="gpu_mean",
                                                 sim_map_aggregation_mode='median',
                                                 n_shifts=4,
                                                 pca_comps=3,
                                                 vit_patch_size = 14, eps = 1e-10):

        shifted_query_tokens = self.compute_augmented_query_tokens(query_image_path, n_shifts=n_shifts)
        self.query_tokens = shifted_query_tokens[0]['tokens'] # get this first element for dS entropy estimation
        self.query_mask = shifted_query_tokens[0]['mask']
        self.query_mask_area_px = shifted_query_tokens[0]['coords'].shape[0] # number of pixels for query image for precise filterint ### ADDED

        qy = np.argwhere(self.query_mask)    # list of (row, col) in full-res pixels
        (y0, x0), (y1, x1) = qy.min(0), qy.max(0)
        self.query_bbox_area_px = (y1 - y0 + 1) * (x1 - x0 + 1)
        self.query_mask_area_px = self.query_mask.sum()  # now also in pixels

        q_vec = self.query_tokens.mean(axis=0)
        q_norm = q_vec / (np.linalg.norm(q_vec) + eps)
        self.query_vec  = q_vec    # raw, for confidence reference
        self.query_norm = q_norm   # unit, for similarity-map construction
        
        # 1) Compute one sim-map per shifted query
        sim_maps = []
        for shift in tqdm(shifted_query_tokens):
            toks = shift['tokens']              # (K_i, D)

            try:
                if sim_map_creation_mode in ["gpu_mean"]:
                    q_vec = toks.mean(axis=0)
                    q_norm = q_vec / (np.linalg.norm(q_vec) + eps)
                    sim_map = self.compute_similarity_chunked_gpu(q_norm)
                    print("[Sim] GPU chunked mean-vector similarity computed.")

                elif sim_map_creation_mode in ["cpu_mean"]:
                    q_vec = toks.mean(axis=0)
                    q_norm = q_vec / (np.linalg.norm(q_vec) + eps)
                    sim_map = self.compute_similarity_chunked(q_norm)
                    print("[Sim] CPU chunked mean-vector similarity computed.")
        
                elif sim_map_creation_mode == "cpu_pca":
                    q_norm, var_fraction = self._topk_pca_vector(toks,  k=pca_comps)
                    sim_map = self.compute_similarity_chunked(q_norm)
                    print(f"[Sim] CPU chunked *PCA-PC{pca_comps}* similarity computed.")
        
                elif sim_map_creation_mode == "gpu_pca":
                    #print('GPU PCA', toks.shape, pca_comps)
                    q_norm, var_fraction = self._topk_pca_vector(toks,  k=pca_comps)
                    sim_map = self.compute_similarity_chunked_gpu(q_norm)
                    print(f"[Sim] GPU chunked *PCA-PC{pca_comps}* similarity computed.")

                sim_maps.append(sim_map)
                
            except Exception as e:
                print('Exception occured due to augmented image not having \"enough\" PCA components. Skipping..')
        
        # 2) pick your aggregation
        agg_map = self.aggregate_sim_maps(sim_maps, mode=sim_map_aggregation_mode, smooth_sigma=1.0)
        
        # 3) final sim_map.
        self.sim_map = agg_map
        self.var_fraction = var_fraction
        
    def aggregate_sim_maps(self, sim_maps, mode='max', smooth_sigma=None, eps=1e-8):
        """
        sim_maps : list of 2D np.arrays all the same shape
        mode     : 'max', 'mean', or 'median'
        smooth_sigma : if not None, apply a gaussian_filter(sim, sigma=smooth_sigma)
        """
        # stack into (K, H, W)
        stack = np.stack(sim_maps, axis=0)
    
        if mode == 'max':
            agg = np.max(stack, axis=0)
        elif mode == 'mean':
            agg = np.mean(stack, axis=0)
        elif mode == 'median':
            agg = np.median(stack, axis=0)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
    
        # optional smoothing to further blur out grid-artifacts
        if smooth_sigma is not None:
            agg = gaussian_filter(agg, sigma=smooth_sigma)
    
        # renormalize to [0,1] and sum to 1
        agg = agg - agg.min()
        agg = agg / (agg.max() - agg.min() + eps)
        agg = agg / (agg.sum() + eps)
        return agg
    
    
    def estimate_object_entropy_drop(self, patch_size=518, stride=14, eps=1e-10):
        """
        Estimate the expected entropy drop from a typical detected object,
        using the query vector and sim_map.
        """
    
        sim_map = self.sim_map.copy()
        sim_map = sim_map.astype(np.float64)
        sim_map = np.clip(sim_map, a_min=0, a_max=None)
        sim_map /= sim_map.sum() + eps
    
        # 1. Estimate object size in tokens (based on query mask)
        query_token_count = len(self.query_tokens)
    
        # 2. Assume an ideal match: find the top-K highest similarity scores
        sim_flat = sim_map.flatten()
        top_k = np.partition(sim_flat, -query_token_count)[-query_token_count:]
        top_k.sort()
    
        # 3. Compute entropy reduction from removing those K values
        P_obj = top_k.sum()
        H     = -np.sum(sim_flat * np.log(sim_flat + eps))
        h_obj = -np.sum(top_k    * np.log(top_k    + eps))
        dH    = -h_obj + P_obj * (np.log(1.0 - P_obj + eps) - H)
        self.estimate_entropy_drop_per_object = dH

    def get_similarity_pil(self):
        """
        Normalize self.sim_map to [0, 255] and return a single-channel (L) PIL image.
        """
        if not hasattr(self, 'sim_map'):
            raise ValueError("sim_map not found. Run compute_token_level_similarity() first.")
        
        sim_map_np = self.sim_map
        sim_norm = (sim_map_np - sim_map_np.min()) / (sim_map_np.max() - sim_map_np.min() + 1e-8)
        sim_img = (sim_norm * 255).astype(np.uint8)
        return Image.fromarray(sim_img, mode='L')
    
    def segment_tile_with_point(self, tile_np, point):
        """
        half precision-accelerated single-click SAM-2 segmentation on one tile.
        `point` is (x, y) in tile coordinates.
        """
        # 1. ensure float32, [0‒1] range
        if tile_np.dtype == np.uint8:
            tile_np = tile_np.astype(np.float32) / 255.0
        else:
            tile_np = tile_np.astype(np.float32)

        # 2. encode & decode under autocast
        dev_type = self.device.type if isinstance(self.device, torch.device) else "cpu"
        with torch.amp.autocast(device_type=dev_type, dtype=torch.float16), torch.no_grad():
            self.sam2_predictor.set_image(tile_np)             # encoder already .half()
            masks, _, _ = self.sam2_predictor.predict(
                point_coords     = np.asarray([point], np.float32),   # (1,2)
                point_labels     = np.asarray([1],     np.int32),
                multimask_output = False
            )
        return masks[0]                              # H×W boolean mask
    
    
    def segment_tile_with_point_multi_functionality(
        self,
        tile_np,
        point,
        spray = False,
        radius_frac = 0.20,
        point_labels=None,
        multimask_output: bool = False
    ):
        """
        half precision-accelerated SAM-2 segmentation on one tile.

        If `spray=True`, nine positive clicks are generated around `point`
        (centre + 8 compass neighbours) with radius =
            radius_frac · R_grain,
        R_grain = sqrt(query_mask_area_px / pi).  Falls back to 1/4 tile area
        if query_mask_area_px is unavailable.
        Returns (mask, score) or (masks, scores) depending on multimask_output.
        """

        # ---------- normalise tile to float32 [0,1] -------------------
        img = tile_np.astype(np.float32)
        if img.max() > 1.01:            # uint8 range
            img /= 255.0

        # ---------- build click array --------------------------------
        if spray:
            cx, cy = map(int, point)
            # clamp radius_frac
            radius_frac = max(0.01, min(radius_frac, 0.9))

            grain_px = getattr(self, "query_mask_area_px",
                            (img.shape[0] * img.shape[1]) // 4)
            R = int(radius_frac * math.sqrt(grain_px / math.pi))

            offsets = [(0,0),
                    (-R, 0), (R,0), (0,-R), (0,R),
                    (-R,-R), (-R,R), (R,-R), (R,R)]
            pts = np.array([(cx+dx, cy+dy) for dx,dy in offsets],
                        dtype=np.float32)

            # clip to valid tile range
            h, w = img.shape[:2]
            pts[:,0] = np.clip(pts[:,0], 0, w-1)
            pts[:,1] = np.clip(pts[:,1], 0, h-1)

            labels = np.ones(len(pts), np.int32)

        else:                            # single click
            pts = np.asarray(point, np.float32).reshape(1, 2)
            labels = (np.asarray(point_labels, np.int32).reshape(-1)
                    if point_labels is not None
                    else np.array([1], np.int32))

        # ---------- SAM forward (encoder+decoder) --------------------
        dev_type = (self.device.type
                    if isinstance(self.device, torch.device) else "cpu")
        with torch.amp.autocast(device_type=dev_type, dtype=torch.float16), \
            torch.no_grad():
            self.sam2_predictor.set_image(img)
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords     = pts,
                point_labels     = labels,
                multimask_output = multimask_output
            )

        # ensure NumPy outputs
        scores = np.asarray(scores, dtype=np.float32)

        return (masks, scores) if multimask_output else (masks[0], scores[0])
    
    
    def extract_patch(self, x_tok, y_tok, patch_size=518, vit_patch_size=14):
        px_center = x_tok * vit_patch_size + vit_patch_size // 2
        py_center = y_tok * vit_patch_size + vit_patch_size // 2
        crop_size = patch_size
        half_crop = crop_size // 2
        x0 = max(px_center - half_crop, 0)
        y0 = max(py_center - half_crop, 0)

        tile_np = np.array(self.wsi.read_region((x0, y0), 0, (crop_size, crop_size)))[:, :, :3]
        px_local = px_center - x0
        py_local = py_center - y0

        # Absolute pixel map for entire tile
        H, W, _ = tile_np.shape
        x_coords = np.arange(x0, x0 + W)
        y_coords = np.arange(y0, y0 + H)
        coord_x, coord_y = np.meshgrid(x_coords, y_coords)
        coord_map = np.stack([coord_x, coord_y], axis=-1)

        return tile_np, (px_local, py_local), coord_map
    

## HELPER FUNCTIONS

def get_ranked_similarity_coords(sim_map, percentile_cutoff=90):
    """
    Returns coordinates of all tokens above the specified percentile threshold,
    sorted from highest to lowest similarity.
    """
    sim_flat = sim_map.flatten()
    threshold = np.percentile(sim_flat, percentile_cutoff)
    mask = sim_map >= threshold
    y_coords, x_coords = np.where(mask)

    scores = sim_map[mask]
    sorted_indices = np.argsort(-scores)
    coords = list(zip(x_coords[sorted_indices], y_coords[sorted_indices]))

    return coords


def get_global_bbox_from_mask(mask, coord_map):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None  # skip degenerate masks
    selected_coords = coord_map[ys, xs]  # (N, 2)
    x_min, y_min = selected_coords.min(axis=0)
    x_max, y_max = selected_coords.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]  # [x1, y1, x2, y2]


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# STOPPING CRITERIA HELPERS

# -----------------------------------------------------------------
#  A)  Non-maximum suppression + running class ratios
# -----------------------------------------------------------------
def flush_nms_gpu(candidate_bboxes, candidate_scores,
                  n_candidates, since_flush,
                  detection_status, nms_class_indices,
                  class_ratios, smoothed_ratios, ema_ratios,
                  nms_class_map,
                  iou_th=0.3, momentum=0.9):

    if since_flush == 0:
        return n_candidates, since_flush

    device = candidate_bboxes.device

    # 1) GPU NMS ------------------------------------------------------
    keep = torchvision.ops.nms(
        candidate_bboxes[:n_candidates],
        candidate_scores[:n_candidates],
        iou_threshold=iou_th
    ).cpu().tolist()

    old_n     = n_candidates - since_flush
    new_idxs  = list(range(old_n, n_candidates))

    kept_new   = {i for i in keep if i >= old_n}
    dropped_old= set(range(old_n)) - {i for i in keep if i < old_n}
    n_rep      = min(len(dropped_old), len(kept_new))

    kept_scores = [(i, candidate_scores[i].item()) for i in kept_new]
    kept_sorted = [i for i, _ in sorted(kept_scores, key=lambda x: x[1], reverse=True)]
    replacements = set(kept_sorted[:n_rep])
    true_new     = kept_new - replacements
    discarded    = set(new_idxs) - kept_new

    # 2) Patch bookkeeping ------------------------------------------
    for i in new_idxs:
        if   i in true_new:     cls = 'new'
        elif i in replacements: cls = 'replacement'
        else:                   cls = 'discarded'

        detection_status[i]  = cls
        nms_class_indices[i] = nms_class_map[cls]

    # 3) Compact GPU tensors ----------------------------------------
    keep_idx = torch.tensor(keep, device=device, dtype=torch.long)
    n_keep   = keep_idx.numel()
    candidate_bboxes[:n_keep]  = candidate_bboxes[keep_idx]
    candidate_scores[:n_keep]  = candidate_scores[keep_idx]
    n_candidates = n_keep

    # 4) Over-write ratio placeholders ------------------------------
    counts = Counter(detection_status)
    tot    = len(detection_status)
    for c in ['new', 'replacement', 'discarded']:
        raw = counts[c] / tot if tot else 0.0
        ema = raw if ema_ratios[c] is None else \
              momentum * ema_ratios[c] + (1 - momentum) * raw
        ema_ratios[c] = ema

        class_ratios   [c][-since_flush:] = [raw] * since_flush
        smoothed_ratios[c][-since_flush:] = [ema] * since_flush

    return n_candidates, 0  # reset since_flush


# -----------------------------------------------------------------
#  B)  Entropy & Δ-entropy tracker
# -----------------------------------------------------------------
def entropy_update(sim_map,
                   entropy_list, delta_entropy, smoothed_delta,
                   window,
                   *,                         # ← only keyword params below
                   w_removed=None,            # 1-D np.array or scalar
                   w_new=0.0,                 # same shape or scalar
                   eps=1e-10):
    """
    Constant-time update of  S = -Σ w log w

    Parameters
    ----------
    sim_map      : 2-D numpy array   (the global similarity map)
    entropy_list : list[float]       (history of S)
    delta_entropy, smoothed_delta
                  : lists used unchanged by the stop-rule code
    window       : int               (ΔS window size)
    w_removed    : 1-D array or scalar of the *old* weights you just lowered
    w_new        : matching array/scalar of the *new* weights you wrote back
                   (defaults to 0.0, which is correct for the “accept” branch)
    eps          : float, numerical safety

    If `w_removed is None`          → do a full recompute (first call / debug).
    """

    # ── A. current entropy value ─────────────────────────────────
    if w_removed is None or not entropy_list:
        # full scan (first call or explicit request)
        S = -np.sum(sim_map * np.log(sim_map + eps))
    else:
        # scalar → array
        if np.isscalar(w_removed):
            w_removed = np.array([w_removed], dtype=np.float32)
        if np.isscalar(w_new):
            w_new = np.full_like(w_removed, w_new, dtype=np.float32)

        S_prev = entropy_list[-1]
        S = S_prev + np.sum(w_removed * np.log(w_removed + eps)
                            - w_new     * np.log(w_new     + eps))

    entropy_list.append(S)

    # ── B. windowed ΔS and its EMA (unchanged logic) ────────────
    if len(entropy_list) > window:
        dS = S - entropy_list[-window-1]
    else:
        dS = 0.0
    delta_entropy.append(dS)

    ema_prev = smoothed_delta[-1] if smoothed_delta else dS
    alpha = 0.01                       # EMA span ≈ 100×WINDOW
    smoothed_delta.append((1-alpha)*ema_prev + alpha*dS)

    return dS

# -----------------------------------------------------------------
#  C)  Confidence + EMA tracker
# -----------------------------------------------------------------
def confidence_update(confidence,
                      confidence_list, smoothed_conf,
                      momentum=0.9):
    """
    Logs raw and exponentially-smoothed confidence.
    """
    if np.isnan(confidence):
        confidence = confidence_list[-1] if confidence_list else 0.0

    confidence_list.append(confidence)

    ema_prev = smoothed_conf[-1] if smoothed_conf else confidence
    smoothed_conf.append(momentum*ema_prev + (1-momentum)*confidence)
    
    # ------------------------------------------------------------------
#  steady_counter helper
# ------------------------------------------------------------------
def update_steady_entropy_counter(dS_window: float,
                          step_list_len: int,
                          steady_hist: list,
                          window: int,
                          thresh: float,
                          #confidence: float,
                          #confidence_thresh: float
                          ) -> int:
    """
    Update the 'flat-entropy' counter and append it to steady_hist.

    Returns the **new** steady_counter.
    """
    steady_counter = steady_hist[-1] if len(steady_hist) > 0 else 0
    if step_list_len > window:                # only after first full window
        if dS_window > thresh:                # entropy flat / increasing
            steady_counter += 1
        elif dS_window < thresh: #and confidence > confidence_thresh:                # mild decay toward zero
            steady_counter = max(steady_counter - 2, 0)

    steady_hist.append(steady_counter)
    return steady_counter


# ------------------------------------------------------------------
#  stop-rule helper
# ------------------------------------------------------------------
def check_convergence(step_count: int,
                      steady_value: int,
                      window_slope:      int,
                      patience_slope:    int,
                      smoothed_ratios: dict,
                      bad_mask_counter: int,
                      badmask_thresh: float,
                     ) -> tuple[bool,str]:
    """
    Decide whether to stop the mining loop.

    Parameters
    ----------
    step_count        : current length of `step_list`
    steady_value      : last element of `steady_hist`  (flat-entropy counter)
    discard_counter   : running counter for high discard ratio
    window_slope      : WINDOW_SLOPE  (objects per entropy window)
    patience_slope    : PATIENCE_SLOPE  (windows to accept as "flat")
    patience_discard  : PATIENCE_DISCARD (iterations for discard criterion)

    Returns
    -------
    stop   : bool
    reason : str  (empty if stop == False)
    """
    # --- criterion A: flat entropy for N windows ------------------
    flat_limit = patience_slope * window_slope      # measured in iterations
    if step_count > window_slope and steady_value >= flat_limit:
        print(f"[STOPPING] Entropy curve flattened → stopping mining.")
        return True, "flat entropy"

    # --- criterion B: high discard ratio --------------------------
    if smoothed_ratios["discarded"][-1] > 0.8 if len(smoothed_ratios["discarded"]) > 0 else 0:
        print(f"[STOPPING] Discarded object ratio exceeds 0.8 → stopping mining.")
        return True,  "high NMS-discard ratio"

    # --- criterion C: high proposed mask discard ratio --------------------------
    bad_mask_ratio = bad_mask_counter / step_count
    if step_count > 10_000 and bad_mask_ratio > badmask_thresh: # avoids an early abort, triggers after 10k iterations
        return True, f"[STOPPING] Bad mask ratio {bad_mask_ratio:.2%} exceeds {badmask_thresh:.0%}"    

    return False, ""

def maybe_add_to_cluster(reservoir, conf, emb, K):
    """
    Keep a min-heap of size ≤ K with the highest-confidence embeddings.
    Returns True if the reservoir changed (so the mean must be recomputed).
    """
    changed = False
    if len(reservoir) < K:
        heapq.heappush(reservoir, (conf, emb))
        changed = True
    elif conf > reservoir[0][0]:           # higher than current minimum
        heapq.heapreplace(reservoir, (conf, emb))
        changed = True
    return changed


## utils for postprocessing and LLM captioning

from sklearn.metrics.pairwise import cosine_similarity
from skimage.measure import regionprops
import io
import base64
from typing import Optional, Tuple, Dict, Any
from textwrap import dedent


def multi_query_attention_confs(
    query_tokens: np.ndarray,
    all_inds: list[np.ndarray],
    embeddings: np.ndarray,
    top_k: int = None,
    batch_size: int = 512,
    device: str = "cuda",
    low_memory: bool = False
) -> np.ndarray:
    """
    For each object j, pool its embeddings K_j against each of the Q query_tokens,
    then average the Q pooled vectors to get vec_j, and compute cos(vec_j, token_i).
    Returns confs[j] in [-1,1].
    
    Args:
        low_memory: If True, process with smaller batches and explicit gc to reduce memory
    """
    import gc
    
    # 1) Prepare query tokens
    Q, D = query_tokens.shape
    qt = torch.from_numpy(query_tokens).to(device=device, dtype=torch.float32)  # (Q, D)
    qt = F.normalize(qt, p=2, dim=1)                                            # unit-length per query token
    qt_cpu = qt.cpu().numpy()  # Keep CPU copy for top-k filtering

    N = len(all_inds)
    confs = np.empty(N, dtype=np.float32)
    pooled_vecs = np.empty((N, D), dtype=np.float32)
    
    # Use specified batch size (low_memory mode only affects gc, not batch size)
    actual_batch_size = batch_size
    
    pbar = tqdm(range(0, N, actual_batch_size), desc="Attention rerank", disable=(N < 1000))
    for start in pbar:
        end = min(N, start + actual_batch_size)
        batch_inds = all_inds[start:end]
        B = end - start

        # 2) Gather and optionally top-k filter per object
        lengths = []
        embs_list = []
        for inds in batch_inds:
            vecs = embeddings[inds]                     # (K_j, D)
            if top_k is not None and len(vecs) > top_k:
                sims = vecs @ qt_cpu.T                 # (K_j, Q)
                # take the top_k tokens by highest max match across queries
                best = np.argpartition(sims.max(axis=1), -top_k)[-top_k:]
                vecs = vecs[best]
            embs_list.append(torch.from_numpy(vecs).to(device=device))
            lengths.append(vecs.shape[0])

        Kmax = max(lengths)
        # 3) Pad into batch tensor + mask
        batch = torch.zeros((B, Kmax, D), device=device)
        mask  = torch.zeros((B, Kmax), device=device, dtype=torch.bool)
        for i, (e, L) in enumerate(zip(embs_list, lengths)):
            batch[i, :L, :] = e
            mask [i, :L]    = True
        
        # Free embs_list immediately
        del embs_list
        if low_memory:
            gc.collect()
            torch.cuda.empty_cache()

        # 4) Compute attention per query token
        #    scores: (B, Kmax, Q) = batch.unsqueeze(2) @ qt.t().unsqueeze(0)
        #    then softmax over tokens dim=1
        # First, expand for bmm: reshape batch to (B*Q, Kmax, 1) and qt to (B*Q, D, 1)? 
        # Simpler: compute for each query separately in a loop (Q is small).
        pooled_per_Q = []
        for qv in qt:  # qv: (D,)
            # (B, Kmax) scores
            scores = (batch @ qv.unsqueeze(-1)).squeeze(-1) / (D**0.5)
            scores.masked_fill_(~mask, float('-inf'))
            a = F.softmax(scores, dim=1)                    # (B, Kmax)
            # pool: (B, D)
            p = (batch * a.unsqueeze(-1)).sum(dim=1)
            pooled_per_Q.append(p)                          # list of Q x (B, D)

        # stack: (B, Q, D)
        stacked = torch.stack(pooled_per_Q, dim=1)
        del pooled_per_Q, batch, mask
        
        # 5) collapse Q → 1 by mean, then normalize: (B, D)
        vecs = stacked.mean(dim=1)
        del stacked
        vecs = F.normalize(vecs, p=2, dim=1)

        # 6) collapse query tokens down to one vector? We want cos(vecs, query mean)
        qmean = qt.mean(dim=0)
        qmean = qmean / qmean.norm()

        # 7) compute cosine similarity: (B,)
        conf_batch = (vecs @ qmean).cpu().numpy()
        confs[start:end] = conf_batch
        pooled_vecs[start:end, :] = vecs.cpu().numpy()   # (B, D)
        
        del vecs
        if low_memory:
            gc.collect()
            torch.cuda.empty_cache()

    return confs, pooled_vecs


def _chunked_medoid_index(subset, chunk_size=5000):
    """
    Find the medoid index using chunked pairwise cosine similarity.
    Avoids creating a (k, k) matrix in memory.
    
    The medoid is the point that maximizes the average cosine similarity
    to all other points. Instead of computing full (k, k) matrix, we compute
    row-wise sums in chunks.
    
    NOTE: This could be GPU-accelerated for further speedup. The computation
    is embarrassingly parallel - each (chunk, chunk) block can be computed
    on GPU and summed. Left as CPU for now since it's already fast enough
    (~seconds vs minutes for the full matrix approach).
    """
    k, D = subset.shape
    row_sums = np.zeros(k, dtype=np.float64)  # Use float64 for numerical stability
    
    # For each chunk of rows, compute their similarities to ALL points
    for i_start in range(0, k, chunk_size):
        i_end = min(k, i_start + chunk_size)
        chunk_rows = subset[i_start:i_end]  # (chunk, D)
        
        # Compute similarities to all points in chunks to avoid memory spike
        for j_start in range(0, k, chunk_size):
            j_end = min(k, j_start + chunk_size)
            chunk_cols = subset[j_start:j_end]  # (chunk2, D)
            
            # (chunk, D) @ (D, chunk2) = (chunk, chunk2)
            sims_block = chunk_rows @ chunk_cols.T
            row_sums[i_start:i_end] += sims_block.sum(axis=1)
    
    # Medoid is the point with highest average similarity (equiv to max sum)
    return np.argmax(row_sums)


def refine_medoid_proto_with_history(grains, init_proto, top_frac=0.25, max_iter=100, tol=1e-4, low_memory=False, max_subset_for_full=50000):
    """
    Iteratively refine a medoid-based prototype, recording its trajectory.

    Args:
        grains    : np.ndarray of shape (N, D), unit-length grain vectors
        init_proto: np.ndarray of shape (D,), unit-length initial prototype
        top_frac  : fraction of top-similar grains to consider each iteration
        max_iter  : maximum number of iterations
        tol       : convergence tolerance on prototype movement
        low_memory: If True, use chunked medoid computation for large subsets
        max_subset_for_full: Use full matrix for subsets smaller than this

    Returns:
        final_proto : np.ndarray shape (D,), the converged prototype
        history      : list of np.ndarray, the prototype at each iteration (including init)
    """
    proto   = init_proto.copy()
    history = [proto.copy()]

    N = grains.shape[0]
    k = max(1, int(N * top_frac))
    
    # Decide whether to use chunked computation
    use_chunked = low_memory and (k > max_subset_for_full)
    if use_chunked:
        print(f"[Medoid] Using chunked computation for k={k:,} subset (would be {k*k*4/1e9:.1f}GB matrix)")

    for it in range(1, max_iter+1):
        # 1) score & pick top-k
        sims    = grains.dot(proto)               # (N,)
        top_idx = np.argsort(-sims)[:k]           # highest-sim grains
        subset  = grains[top_idx]                 # (k, D)

        # 2) find subset medoid
        if use_chunked:
            medoid_i = _chunked_medoid_index(subset, chunk_size=5000)
        else:
            cs       = cosine_similarity(subset, subset)  # (k, k)
            medoid_i = np.argmax(cs.mean(axis=1))
            del cs  # Free immediately
            
        new_proto = subset[medoid_i].copy()
        new_proto /= (np.linalg.norm(new_proto) + 1e-8)

        history.append(new_proto.copy())

        # 3) check convergence
        if np.linalg.norm(new_proto - proto) < tol:
            print(f"Converged at iteration {it}")
            proto = new_proto
            break

        proto = new_proto

    return proto, history


def mask_diameters(mask):
    """
    Return (equatorial_px, polar_px) using RegionProps axes.
    axis_major_length ≈ longest Feret  → polar axis
    axis_minor_length ≈ shortest Feret → equatorial axis
    """
    props = regionprops(mask.astype(np.uint8))[0]
    pol_px = props.axis_major_length
    eq_px  = props.axis_minor_length
    return eq_px, pol_px


def wsi_patch_b64(pipeline, bbox, margin=5):
    """
    bbox: tuple/list (x_min, y_min, x_max, y_max) in level-0 pixels.
    margin: extra pixels around bbox to give the model a little context.
    Returns base64-encoded PNG bytes.
    """
    #print(bbox)
    x0, y0, x1, y1 = map(int, bbox.astype(float))
    w, h = x1 - x0, y1 - y0

    # expend bbox with margin but clamp to slide dims
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    w  = min(w + 2 * margin, pipeline.wsi.dimensions[0] - x0)
    h  = min(h + 2 * margin, pipeline.wsi.dimensions[1] - y0)

    rgba = pipeline.wsi.read_region((x0, y0), 0, (w, h)).convert("RGB")
    buf  = io.BytesIO()
    rgba.save(buf, format="PNG", optimize=True)

    rgba_query = pipeline.query_img
    buf_query  = io.BytesIO()
    rgba_query.save(buf_query, format="PNG", optimize=True)
    
    return base64.b64encode(buf_query.getvalue()).decode(), base64.b64encode(buf.getvalue()).decode()


def build_prompt_morphology(
    pol_um: float,
    eq_um: float,
    area_um2: float,
    taxon_hint: Optional[str] = None,
    confidence: Optional[float] = None,
    target_len: Tuple[int, int] = (60, 80),
    anchor: Optional[str] = None,
) -> str:
    """Return a morphology-aware prompt suitable for Qwen 2.5-VL-32B.

    The function injects *qualitative* size information, optional genus-level
    hints and a miner-loop similarity score, while forbidding numeric leakage
    and taxon names in the final caption.
    """

    # ---- qualitative size cue --------------------------------------------
    mean_diam_um = 0.5 * (pol_um + eq_um)
    if mean_diam_um < 20:
        size_qual = "small"
    elif mean_diam_um < 50:
        size_qual = "medium-sized"
    else:
        size_qual = "large"

    # ---- assemble prompt --------------------------------------------------
    return (
        dedent(
f"""You are a specialist palynologist image-analysis assistant examining a bright-field transmitted-light micrograph.
First describe only what is visible in the first image of a pollen grain or possible debris/artifact located by an object detector. 
Then, use unified, precise scientific language and summarize your output into a single paragraph between {target_len[0]} and {target_len[1]} words.

Use the second image, which shows a retrieved reference pollen grain, as the primary visual comparator. Describe only the first image; do not describe the second.
Map observed features to the reference description terminology only if they are unambiguously present; otherwise write exactly: not resolved, partially resolved, consistent with or suggestive of. Do not assume unseen features.

The pollen grain in the first image may be complete or fractional, physically damaged, partially obscured, or imaged such that some features are hidden (most likely due to 2D imaging).
Some areas may show structures consistent with the reference image and reference description, such as one or two visible pores, partial colpi, or sections of reticulate exine, while others are hidden, missing, obscured, or deformed. 
If this partial match is observed, note this in your output, but always generate a full caption describing the visible features and, if evident, the missing features and their likely cause. Avoid inferring unseen structures.

The object in the first image may be pollen grain or non-pollen material (e.g., dust, fiber, bubble, plant fragment, crystal).
If the object clearly does not match the reference image or pollen morphology in any respect, begin your output with exactly: Debris, dust or artifact detected; no matching pollen grain present.
Then describe what is visible, without using pollen-specific terminology unless unambiguously observed in the first image. 
If the object shows clear pollen morphology but differs from the reference image and reference caption (e.g., different aperture system or wall sculpture), do not use the debris opener. Describe it using only features that are visible and state: hint or exemplar inconsistent.

Write in fluent scientific English: no bullet points, hedging language, contradictions, numerical values, taxon names, stain colour, background, or field-of-view details.

Short taxon hint: {taxon_hint}

Reference description derived from Palynological Database (PalDat) for verification—use as terminology/style only and for feature matching when visible:

{anchor}

Current grain size class: {size_qual}.

Similarity to canonical cluster centre: {confidence:.2f} (0 = atypical, 1 = very typical). Use only as a prior.

(Context for your internal use - do **not** quote:  polar axis approx. {pol_um:.1f} microns, equatorial axis approx. {eq_um:.1f} microns, pollen grain area approx. {area_um2:.0f} square microns)

"""
        ).strip()
    )

def caption_via_vllm(
    pipeline: Any,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    conf: float,
    PIXEL_UM: float, 
    client: Any,
    model: Any,
    TEMP: float,
    MAX_TOK: np.uint32,
    taxon_hint: Optional[str],
    anchor: Optional[str],
) -> Dict[str, Any]:
    """Generate a morphology caption for a single grain crop.

    Returns a dict that already conforms to the record schema used by
    :func:`caption_slide`.
    """

    # ---- geometry ---------------------------------------------------------
    eq_px, pol_px = mask_diameters(mask)
    pol_um = pol_px * PIXEL_UM
    eq_um = eq_px * PIXEL_UM
    area_um2 = np.count_nonzero(mask) * PIXEL_UM**2

    # ---- images -----------------------------------------------------------
    query_img_b64, img_b64 = wsi_patch_b64(pipeline, bbox)

    # ---- prompt -----------------------------------------------------------
    prompt_body = build_prompt_morphology(
        pol_um,
        eq_um,
        area_um2,
        taxon_hint=taxon_hint,
        confidence=conf,
        anchor=anchor,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a palynologist image-analysis assistant, describing pollen morphology from microscopic images.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_body},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_img_b64}"}},
            ],
        },
    ]

    rsp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
    )

    caption_text = rsp.choices[0].message.content.strip()

    return {
        "prompt": prompt_body,
        "img_b64": img_b64,
        "caption": caption_text,
        "polar_diameter": pol_um,
        "equatorial_diameter": eq_um,
        "area_um2": area_um2,
    }
