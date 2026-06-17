# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

import os

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import torch
torch.set_num_threads(16)

import numpy as np
import random
from tqdm import tqdm
import h5py
import time
import argparse

import skimage
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.utils import (WSIPipeline, 
                   get_ranked_similarity_coords,
                   get_global_bbox_from_mask,
                   cosine_sim,
                   flush_nms_gpu,
                   entropy_update,
                   confidence_update,
                   update_steady_entropy_counter,
                   check_convergence,
                   maybe_add_to_cluster
                   )

import logging
# Suppress everything below WARNING for all loggers
logging.basicConfig(level=logging.WARNING)

# Optional: get specific loggers by name if needed
logging.getLogger("sam").setLevel(logging.ERROR)


### argparse needed !!!  -> add code with default values
parser = argparse.ArgumentParser()

# path args
parser.add_argument('--output', type=str, default="./outdir/", help="Output directory")
parser.add_argument('--wsi_path', type=str, required=True, help="Path to WSI file")
parser.add_argument('--query_image', type=str, required=True, help="Query image path")

# model based args
parser.add_argument('--vit_ckpt', type=str, default="../01_initialization/weights_vit_small_lvd_20250620_0312.pth")
parser.add_argument('--vit_name', type=str, default="vit_small_patch14_dinov2.lvd142m")
parser.add_argument('--vit_patch_size', type=int, default=14)
parser.add_argument('--patch_size', type=int, default=518)
parser.add_argument('--sam2_ckpt', type=str, required=True)
parser.add_argument('--sam2_cfg', type=str, required=True)
parser.add_argument('--device', type=str, default="cuda:0")

# mining loop general args
parser.add_argument('--percentile', type=float, default=90)
parser.add_argument('--min_mask_ratio', type=float, default=1/4)
parser.add_argument('--max_mask_ratio', type=float, default=2.5)
parser.add_argument('--badmask_ratio', type=float, default=0.7) # tracks discarded mask ratio based on size

# nms based args
parser.add_argument('--max_nms_objs', type=int, default=250_000) # an upper bound on how many proposals to look for
parser.add_argument('--nms_flush_every', type=int, default=256) # tune for your hardware and current sample's density: 64-256 is a good sweet spot - important if large number of objects present, then set it to larger value
parser.add_argument('--momentum_nms', type=float, default=0.9)
parser.add_argument('--iou_nms', type=float, default=0.3)

# entropy based args
parser.add_argument('--window_slope_entropy', type=int, default=100) # window used in entropy update
parser.add_argument('--dS_multiplier', type=float, default=100) # dS entropy estimation threshold multiplier - shift to any direction for stricter or looser stopping rule in S
parser.add_argument('--patience_slope_entropy', type=int, default=100) # need N consecutive flat windows (~10_000 objs)

# confidence based args
parser.add_argument('--confidence_thresh', type=float, default=0.65) # accepts new objects to cluster calc and against flat entropy counter - less strict to seed cluster
parser.add_argument('--conf_cluster_size', type=int, default=1000) # keep at most 1000 best grains

# sim map refinement related args
parser.add_argument('--n_shifts_augments', type=int, default=4) # keep at most 1000 best grains
parser.add_argument('--n_pca_comps', type=int, default=10) # keep at most 1000 best grains
parser.add_argument('--divisor_for_sam_query', type=int, default=2) # use to align point query for sam mask init, if middle point is bad for query image

# SAM mask generation from single or multiple points 
parser.add_argument('--sam_multi_point_query', action='store_true', default=False) # use to create much better mask for query image, span the object better with query points
parser.add_argument('--sam_multimask_output', action='store_true', default=False) # use to create much better masks for current object, select best from 3

parser.add_argument('--tile_size_multiplier', type=float, default=2.5) # align if needed, tile size centered to current query point based on query image


# parser.add_argument('--restart', action='store_true', help='Resume mining from an existing detections.h5')  # consider adding restart possibility if needed

args = parser.parse_args()

# 1) Paths
output_directory = args.output
wsi_path = args.wsi_path
query_image_path = args.query_image

# 2) General settings
patch_size = args.patch_size

vit_ckp = args.vit_ckpt
vit_name = args.vit_name
vit_patch_size = args.vit_patch_size

sam2_ckpt = args.sam2_ckpt
sam2_cfg = args.sam2_cfg

device = torch.device(args.device)
torch.cuda.set_device(device)


# 3) Mining loop settings

## upper limit for mining loop, maximal iterations in percentile
percentile = args.percentile

## mask accept contants
MIN_MASK_RATIO = args.min_mask_ratio
MAX_MASK_RATIO = args.max_mask_ratio
BADMASK_THRESH = args.badmask_ratio


## [NMS] maximum object number and nms check parameters
max_nms_objs = args.max_nms_objs
nms_flush_every = args.nms_flush_every
momentum_nms = args.momentum_nms
iou_nms = args.iou_nms

## [Entropy] entropy related parameters, stopping rule
window_slope_entropy = args.window_slope_entropy
dS_multiplier = args.dS_multiplier
patience_slope_entropy = args.patience_slope_entropy 

## [Confidence] confidence related parameters - clustering, stopping
confidence_thresh = args.confidence_thresh  
conf_cluster_size = args.conf_cluster_size

## [Sim map]
n_shifts_augments = args.n_shifts_augments
n_pca_comps = args.n_pca_comps
divisor_for_sam_query = args.divisor_for_sam_query

## [SAM] mask generation
sam_multi_point_query = args.sam_multi_point_query
sam_multimask_output = args.sam_multimask_output

tile_size_multiplier = args.tile_size_multiplier

#restart = args.restart # True or false    # consider adding restart possibility if needed

# 4) Additional ones

# Fix seeds, python, numpy, pytorch (both CPU and CUDA)
SEED = 42 # pick your favorite seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


### Processing
if __name__ == "__main__":

    # Create output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Generate output file name based on WSI filename
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    # consider adding restart possibility if needed
    #if not restart:
    save_file = os.path.join(output_directory, f"{wsi_name}_detections.h5")
    #else:
    #    load_file = os.path.join(output_directory, f"{wsi_name}_detections.h5") # load back prev h5
    #    save_file = os.path.join(output_directory, f"{wsi_name}_detections_restart.h5")

    # init object
    try:
        # --- init pipeline ------------------------------------------------
        print('[WSI] path:', wsi_path)
        pipeline = WSIPipeline(wsi_path, 
                            vit_ckpt=vit_ckp,
                            sam2_ckpt=sam2_ckpt, 
                            sam2_cfg=sam2_cfg,
                            device=device,
                            divisor_for_sam_query=divisor_for_sam_query,
                            sam_multi_point_query=sam_multi_point_query)

        # Compute ViT embeddings
        pipeline.compute_vit_embeddings() # Embed all tiles with ViT

        # Check
        invalid = pipeline.coords[(pipeline.coords[:, 0] % vit_patch_size != 0) | (pipeline.coords[:, 1] % vit_patch_size != 0)]
        print(f"Found {len(invalid)} invalid token coords:", invalid[:10])

        # Computing token-level similarity for the query img
        # HERE MAYBE add a fallback to only mean based rule without augmentation and shift
        #pipeline.compute_token_level_similarity(query_image_path)
        
        pipeline.compute_token_level_similarity_augmented(query_image_path,
                                                  sim_map_creation_mode='gpu_pca',
                                                  pca_comps=n_pca_comps,
                                                  n_shifts=n_shifts_augments)

        # Estimate entropy drop if relevant object found
        pipeline.estimate_object_entropy_drop()
        print('[entropy] Estimated entropy drop:', pipeline.estimate_entropy_drop_per_object)

        # Get highest marked pixels based on similarity and sort in descending order
        ranked_coords = get_ranked_similarity_coords(pipeline.sim_map, percentile_cutoff=percentile)

        # Fill this value into sim map to replace values of bad masks found
        sim_floor = np.percentile(pipeline.sim_map, 0.5)   # robust floor
        print(f"[sim floor] Using {sim_floor:.10f} as background replacement value")

        # Init variables for object mining loop
        H_tok, W_tok = pipeline.sim_map.shape
        index_map = np.full((H_tok, W_tok), -1, dtype=np.int32)      # -1 = empty
        yx = pipeline.coords[:, 1] // vit_patch_size   # token-row
        xx = pipeline.coords[:, 0] // vit_patch_size   # token-col
        index_map[yx, xx] = np.arange(len(pipeline.coords), dtype=np.int32)
        visited_tokens = np.zeros((H_tok, W_tok), np.bool_)

        # --- tracking lists ------------------------------------------------
        step_list, conf_steps, confidence_list, smoothed_conf = [], [], [], []
        entropy_list, delta_entropy, smoothed_delta = [], [], []
        detection_status, nms_class_indices = [], []
        nms_class_map = {'new': 0, 'replacement': 1, 'discarded': 2}
        class_ratios = {'new':[], 'replacement':[], 'discarded':[]}
        smoothed_ratios = {'new':[], 'replacement':[], 'discarded':[]}
        ema_ratios = {'new':None,'replacement':None,'discarded':None}

        since_flush = 0                       # how many boxes added since last flush
        candidate_bboxes = torch.zeros((max_nms_objs, 4),  device=pipeline.device, dtype=torch.float32) # preferably on the GPU
        candidate_scores = torch.zeros((max_nms_objs,),     device=pipeline.device, dtype=torch.float32) # preferably on the GPU
        n_candidates = 0                      # how many objects found and written so far to nms buffer

        # --- STOP-RULE variables  ------------------------------------------------
        pipeline.estimate_object_entropy_drop()
        dS_est = pipeline.estimate_entropy_drop_per_object
        slope_thresh_dS = -abs(dS_est) / dS_multiplier     # e.g. -2.3e-5 if dS_est ≈ -0.0023
        print('SLOPE THRESH:', slope_thresh_dS)
        
        steady_hist = []               # tracks flat entopy history
        bad_mask_counter = 0           # tracks high discard ratio based on proposed mask size

        # Cluster related variables
        cluster_mean  = pipeline.query_vec.copy()     # raw vector
        reservoir = []                    # min-heap of (confidence, embedding)
        cluster_frozen = False            # will turn True once mean converges
        prev_cluster_mean = cluster_mean.copy() # init
        converged_counter = 0             # counts successive flat updates

        # Calculate mask area to skip if bad mask is proposed
        query_area_px = pipeline.query_mask_area_px
        min_mask_area = MIN_MASK_RATIO * query_area_px
        max_mask_area = MAX_MASK_RATIO * query_area_px
        
        print('MASK AREAS:', query_area_px, min_mask_area, max_mask_area, pipeline.query_tokens.shape)
        
        query_bbox_area_px = pipeline.query_bbox_area_px
        min_bbox_area = MIN_MASK_RATIO * query_bbox_area_px
        max_bbox_area = MAX_MASK_RATIO * query_bbox_area_px

        tile_size_mining = int(max(np.array(pipeline.query_img.size) * tile_size_multiplier)) # tile size used for object mining        ##### CONSIDER ADDING THIS AS ARGUMENT   !!! 

        # --- Containers and variables  ------------------------------------------------
        found_objects_num = 0
        
        # ── PRE-ALLOCATE EVERYTHING ─────────────────────────────
        all_masks = np.zeros((max_nms_objs, tile_size_mining, tile_size_mining), dtype=np.uint8)
        all_points = np.zeros((max_nms_objs, 2), dtype=np.int32)
        all_toplefts = np.zeros((max_nms_objs, 2), dtype=np.int32)
        all_bboxes = np.zeros((max_nms_objs, 4), dtype=np.int32)
        all_confs = np.zeros((max_nms_objs,), dtype=np.float32)
        all_timestamps = np.zeros((max_nms_objs,), dtype=np.float32)
        all_indices = [] # not known !! variable shaped array
        
        # --- Miner LOOP ------------------------------------------------
        start_time = time.time()
        pbar = tqdm(ranked_coords, ascii=True, ncols=100)
            
        for x_tok, y_tok in pbar:
            
            # ── 1) SKIP: token already mined ───────────────────────────────
            if visited_tokens[y_tok, x_tok]:
                dS_window = entropy_update(pipeline.sim_map, entropy_list,
                            delta_entropy, smoothed_delta, window_slope_entropy,
                            w_removed=np.empty(0, dtype=np.float32))

                update_steady_entropy_counter(
                    dS_window=dS_window,
                    step_list_len=len(step_list),
                    steady_hist=steady_hist,
                    window=window_slope_entropy,
                    thresh=slope_thresh_dS,
                )
                
                step_list.append(len(step_list) + 1)

                stop, reason = check_convergence(
                    step_count      = len(step_list),
                    steady_value    = steady_hist[-1],
                    window_slope    = window_slope_entropy,
                    patience_slope  = patience_slope_entropy,
                    smoothed_ratios = smoothed_ratios,
                    bad_mask_counter = bad_mask_counter,
                    badmask_thresh = BADMASK_THRESH
                )
                if stop:
                    pbar.write(f"[STOP] {reason} - terminating mining.")
                    break
                
                continue
            
            
            # ── 2) SEGMENT current peak ───────────────────────────────────
            tile_np, point, coord_map = pipeline.extract_patch(
                x_tok, y_tok,
                patch_size=tile_size_mining )

            # MULTIMASK output from SAM
            if sam_multimask_output:
                with torch.no_grad():
                    masks, scores = pipeline.segment_tile_with_point_multi_functionality(
                                                tile_np,
                                                point,
                                                spray = False,
                                                radius_frac = 0.20,
                                                multimask_output = True
                                                )    
                
                # choose the best mask that satisfies the area limits
                mask_areas = np.array([m.sum() for m in masks], dtype=int)
                ok_idx = np.where((min_mask_area <= mask_areas) & (mask_areas <= max_mask_area))[0]

                # --- select mask -------------------------------------------------
                if ok_idx.size:                                # at least one valid area
                    best = ok_idx[np.argmax(scores[ok_idx])]
                else:                                          # fallback: best score overall
                    best = int(np.argmax(scores))
                mask = masks[best].astype(bool)
                confidence = float(scores[best])

                # drop tiny speckles (<20% of query area)
                min_obj = int(0.20 * query_area_px)
                mask = remove_small_objects(mask, min_size=min_obj)
                
                # fill small holes (<20% of query area)
                min_hole = int(0.20 * query_area_px)
                mask = remove_small_holes(mask, area_threshold=min_hole)
                
                #if not (min_bbox_area <= bbox_area_px <= max_bbox_area):
                #    # no viable mask → skip this token
                #    continue
                
                mask_area = int(mask.sum())
            
            # DEFAULT BRANCH: segment with one click query - all simpler grains are done this way
            else:
                with torch.no_grad():
                    mask = pipeline.segment_tile_with_point(tile_np, point)
                mask_area = mask.sum()
            
            torch.cuda.empty_cache()
            
            # 3) compute bounding‐box area in pixels
            if mask_area == 0:                             # mask is empty
                bbox_area_px = 0                           # sentinel → will fail size test
            else:
                qy = np.argwhere(mask)                     # list of (row, col) in full-res pixels
                (y0, x0), (y1, x1) = qy.min(0), qy.max(0)
                bbox_area_px = (y1 - y0 + 1) * (x1 - x0 + 1)
            
            if not ( (min_mask_area <= mask_area <= max_mask_area) and
                     (min_bbox_area <= bbox_area_px <= max_bbox_area) ):
                # ── 2a) DISCARD branch (bad mask size) ────────────────────
                visited_tokens[y_tok, x_tok]   = True

                # grab the old similarity value *before* you replace it
                w_removed = np.array([pipeline.sim_map[y_tok, x_tok]], dtype=np.float32)
                pipeline.sim_map[y_tok, x_tok] = sim_floor          # flatten peak
                w_new = np.array([sim_floor], dtype=np.float32)

                # log as bad mask
                bad_mask_counter += 1

                dS_window = entropy_update(
                    pipeline.sim_map, entropy_list,
                    delta_entropy, smoothed_delta, window_slope_entropy,
                    w_removed=w_removed,            # old value
                    w_new=w_new                     # new value
                )

                update_steady_entropy_counter(
                    dS_window=dS_window,
                    step_list_len=len(step_list),
                    steady_hist=steady_hist,
                    window=window_slope_entropy,
                    thresh=slope_thresh_dS,
                )
                
                step_list.append(len(step_list) + 1)

                stop, reason = check_convergence(
                    step_count      = len(step_list),
                    steady_value    = steady_hist[-1],
                    window_slope    = window_slope_entropy,
                    patience_slope  = patience_slope_entropy,
                    smoothed_ratios = smoothed_ratios,
                    bad_mask_counter = bad_mask_counter,
                    badmask_thresh = BADMASK_THRESH
                )
                if stop:
                    pbar.write(f"[STOP] {reason} - terminating mining.")
                    break
                
                continue
            
            
            # ── 3) ACCEPT branch: evaluate mask tokens ────────────────────
            wsi_coords = coord_map[mask.astype(bool)] # Update after segmentation
            # Get token coordinates and index out from flattened array
            tok_x = (wsi_coords[:, 0] // vit_patch_size).astype(int)
            tok_y = (wsi_coords[:, 1] // vit_patch_size).astype(int)

            # unique token positions for this mask
            tok_flat = tok_y * W_tok + tok_x               # 1-D IDs
            unique_tok_flat = np.unique(tok_flat)          # 1-D numpy array
            ty = unique_tok_flat // W_tok
            tx = unique_tok_flat %  W_tok

            # bounds filter
            inside = (ty >= 0) & (ty < H_tok) & (tx >= 0) & (tx < W_tok)
            ty = ty[inside]
            tx = tx[inside]
            if ty.size == 0:               # the mask fell completely outside the embedded grid
                continue
            
            indices = index_map[ty, tx]
            indices = indices[indices != -1]      # drop padding
            # Check if indices is non-empty before mean
            if indices.size == 0:
                continue  # Skip this object

            # --- Confidence calculation
            mean_token_this = pipeline.embeddings[indices].mean(axis=0) # Get mean token of current object

            # Numerical stability
            if np.isnan(mean_token_this).any():
                continue
            
            # Compare to cluster
            # mean_token_this is raw (unnormalised)       shape (D,)
            # cluster_mean   is raw - initialised once with pipeline.query_vec
            confidence = cosine_sim(mean_token_this, cluster_mean)

            # ----- more robust cluster update -----------------------
            if (confidence > confidence_thresh) and (not cluster_frozen):
                # try to insert; recompute mean only if reservoir changed
                if maybe_add_to_cluster(reservoir, confidence, mean_token_this, conf_cluster_size):
                    cluster_mean = np.mean([e for _, e in reservoir], axis=0)
            
                    # ---- convergence check ----
                    if cosine_sim(cluster_mean, prev_cluster_mean) > 0.999: # convergence threshold
                        converged_counter += 1
                    else:
                        converged_counter = 0
                    # freeze after N consecutive negligible changes
                    prev_cluster_mean = cluster_mean.copy()
            
                    if converged_counter >= conf_cluster_size//10 or len(reservoir) == conf_cluster_size:
                        cluster_frozen = True
                        print(f"[cluster] mean stabilised after {len(reservoir)} samples; "
                            "further updates frozen.")

            # ignore low-conf objects fully, continue with next
            if confidence < 0.5: # mostly background or irrelevant object -> do not track
                continue  # skip entropy and sim_map update
            
            w_removed = pipeline.sim_map[ty, tx].astype(np.float32).copy() # grab the old weights *before* zeroing -> dS calc
            
            visited_tokens[ty, tx] = True
            pipeline.sim_map[ty, tx] = 0.0
            
            # --- NMS & bookkeeping ----------------------------------------
            bbox = get_global_bbox_from_mask(mask, coord_map)

            # write the new box + score into your GPU buffer
            idx = n_candidates

            candidate_bboxes[idx] = torch.tensor(bbox, dtype=torch.float32, device=pipeline.device)
            candidate_scores[idx] = float(confidence)
            since_flush += 1      # one more box since last flush
            n_candidates += 1     # always grow (we haven’t filtered yet)
            
            # provisionally mark it “new” so list lengths stay in sync
            detection_status.append('pending')          # will be patched at flush
            nms_class_indices.append(-1)          # placeholder, any value; will patch later
            for c in ['new', 'replacement', 'discarded']:
                if class_ratios[c]:
                    class_ratios[c].append(class_ratios   [c][-1])
                    smoothed_ratios[c].append(smoothed_ratios[c][-1])
                else:
                    class_ratios[c].append(0.0)
                    smoothed_ratios[c].append(0.0)
            
            # --- one-call GPU NMS & bookkeeping -------------------------------
            if since_flush >= nms_flush_every:
                n_candidates, since_flush = flush_nms_gpu(
                    candidate_bboxes, candidate_scores,
                    n_candidates, since_flush,
                    detection_status, nms_class_indices,
                    class_ratios, smoothed_ratios, ema_ratios,
                    nms_class_map,
                    iou_th=iou_nms, momentum=momentum_nms
                )

            # --- Entropy update ----------------------------------------
            dS_window = entropy_update(pipeline.sim_map, entropy_list,
                        delta_entropy, smoothed_delta, window_slope_entropy,
                        w_removed=w_removed)          # incremental path
            
            update_steady_entropy_counter(
                    dS_window=dS_window,
                    step_list_len=len(step_list),
                    steady_hist=steady_hist,
                    window=window_slope_entropy,
                    thresh=slope_thresh_dS,
                )
            
            # --- Confidence update ----------------------------------------
            confidence_update(confidence, confidence_list, smoothed_conf)


            # --- Increment variables and save ----------------------------------------
            conf_steps.append(len(conf_steps))          # x-axis for confidence only
            step_list.append(len(step_list) + 1)
            
            top_left = tuple(coord_map[0, 0])  # infer top-left directly, no function change
            
            now = time.time()
            elapsed_since_start = now - start_time  # seconds since beginning
            
            # Save object found
            all_masks[found_objects_num] = mask                     # (H,W) uint8
            all_points[found_objects_num] = point                   # [x,y]
            all_toplefts[found_objects_num] = top_left              # [x,y]
            all_bboxes[found_objects_num] = bbox                    # [x1,y1,x2,y2]
            all_confs[found_objects_num] = confidence               # float
            all_indices.append(indices)
            all_timestamps[found_objects_num] = elapsed_since_start
            
            found_objects_num += 1
            
            # --- Convergence update based on stop rules -------------------------------
            stop, reason = check_convergence(
                    step_count = len(step_list),
                    steady_value = steady_hist[-1],
                    window_slope = window_slope_entropy,
                    patience_slope = patience_slope_entropy,
                    smoothed_ratios = smoothed_ratios,
                    bad_mask_counter = bad_mask_counter,
                    badmask_thresh = BADMASK_THRESH
            )
            if stop:
                pbar.write(f"[STOP] {reason} - terminating mining.")
                break
            
            
            # --- Timing checkpoint log -------------------------------
            if found_objects_num % 200 == 0:
                now = time.time()
                elapsed = now - start_time
                rate = found_objects_num / elapsed
                current_entropy = entropy_list[-1] if entropy_list else np.nan
                current_smoothed_delta = smoothed_delta[-1] if smoothed_delta else np.nan
                steady_entropy_count = steady_hist[-1]
                discarded_ratio = smoothed_ratios['discarded'][-1] if smoothed_ratios['discarded'] else 0.0
                current_smoothed_conf = smoothed_conf[-1]       if smoothed_conf       else np.nan
                print(
                    f"[{elapsed/60:.1f}m]  Found: {found_objects_num},  "
                    f"Rate: {rate:.1f} obj/s,  "
                    f"Entropy: {current_entropy:.4f},  "
                    f"ΔS (EMA): {current_smoothed_delta:.4e},  "
                    f"Steady iterations: {steady_entropy_count},  "
                    f"Discarded (EMA): {discarded_ratio:.2f},  "
                    f"Conf (EMA): {current_smoothed_conf:.2f}"
                )        


        # --- Saving variables to disk after LOOP -------------------------------
        # flatten + offsets for indices
        flat_inds = np.concatenate(all_indices)
        offs = np.concatenate([[0], np.cumsum([len(x) for x in all_indices])])

        # Initialize HDF5 file for saving masks and metadata    
        with h5py.File(save_file, "w") as hf:
            h5_grp = hf.create_group("results")

            # SAVE TO H5 after exit
            h5_grp.create_dataset("mask", data=all_masks[:found_objects_num].astype(np.uint8), compression="gzip")
            h5_grp.create_dataset("point", data=all_points[:found_objects_num])
            h5_grp.create_dataset("topleft", data=all_toplefts[:found_objects_num])
            h5_grp.create_dataset("bbox", data=all_bboxes[:found_objects_num])
            h5_grp.create_dataset("confidence", data=all_confs[:found_objects_num])
            h5_grp.create_dataset("indices_flat", data=flat_inds, dtype=np.int32)
            h5_grp.create_dataset("indices_offs", data=offs, dtype=np.int32)
            h5_grp.create_dataset("timestamp", data=all_timestamps[:found_objects_num])


            # ---------- SAVE RUN-LEVEL METRICS ----------
            h5_grp.create_dataset("entropy_list", data=np.array(entropy_list)) # entropy hist
            h5_grp.create_dataset("delta_entropy", data=np.array(delta_entropy)) # delta entropy hist
            h5_grp.create_dataset("smoothed_delta", data=np.array(smoothed_delta)) # smoothed delta entropy hist
            h5_grp.create_dataset("confidence_list", data=np.array(confidence_list)) # confidence hist
            h5_grp.create_dataset("smoothed_conf", data=np.array(smoothed_conf)) # smoothed confidence hist
            h5_grp.create_dataset("nms_class_indices", data=np.array(nms_class_indices)) # nms hist
            h5_grp.create_dataset("detection_status", data=np.array(detection_status, dtype='S')) # nms status hist

    finally:
        # close file -> Final cleanup
        pipeline.wsi.close()
