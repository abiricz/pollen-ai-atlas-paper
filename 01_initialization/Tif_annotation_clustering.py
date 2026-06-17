# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

import timm

from PIL import Image, ImageOps
import argparse

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.preprocessing import *

class CollectionsDataset(Dataset):
    def __init__(self,
                 data,
                 labels,
                 num_classes, 
                 transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return {'image': image,
                'label': label
                }

def preprocess_query_image(image_path, transform, target_size=224):
    """
    Load and preprocess a query image by center cropping or padding to the target size.

    Args:
        image_path (str): Path to the query image.
        target_size (int): Target width and height for the image.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for embedding.
    """
    # Load the image
    query_image = Image.open(image_path).convert("RGB")
    
    # Resize to fit within target_size while keeping aspect ratio
    query_image = np.array( ImageOps.fit(query_image, (target_size, target_size), method=Image.BICUBIC) )
    
    # Apply transformation
    preprocessed_image = transform(query_image)
    return preprocessed_image


## MAIN
if __name__ == "__main__":
    # ARG handling
    
    ## define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str, required=True) # external data root
    parser.add_argument("--wsi_root_dir", type=str, required=True) # directory containing input WSI files
    parser.add_argument("--hdf5_path", type=str, required=True) # "one_shot_annotations_french/corylus_2_edf.h5"
    parser.add_argument("--current_wsi", type=str, required=True) # e.g. betula_2_edf.tif
    parser.add_argument("--query_image_path", type=str, required=True) # "query_images/betula_2_edf.png"
    parser.add_argument("--output_images_path", type=str, required=False, default='')
    parser.add_argument("--output_hdf5_path", type=str, required=False, default='')
    parser.add_argument("--cuda_num", type=str, required=False, default='0')
    parser.add_argument("--lower_ellipse_axis_ratio", type=int, required=False, default=0.85) # default 0.85
    parser.add_argument("--upper_ellipse_axis_ratio", type=int, required=False, default=1.15) # default 1.15
    parser.add_argument("--vit_embedder_weights_path", type=str, required=False, default='')
    parser.add_argument("--score_threshold", type=float, required=False, default=0.0)

    # parse arguments
    args = parser.parse_args()

    project_root_dir = args.project_root_dir
    wsi_root_dir = args.wsi_root_dir
    hdf5_path = args.hdf5_path
    output_hdf5_path = args.output_hdf5_path
    current_wsi = args.current_wsi
    query_image_path = args.query_image_path
    output_images_path = args.output_images_path
    cuda_num = args.cuda_num
    
    lower_ellipse_axis_ratio = args.lower_ellipse_axis_ratio
    upper_ellipse_axis_ratio = args.upper_ellipse_axis_ratio
    
    vit_embedder_weights_path = args.vit_embedder_weights_path ## CAN be used to get fine-tuned weights
    
    score_threshold = args.score_threshold
    
    # Fix seed
    device = seed_torch(42, cuda_num)
    
    # Load data
    
    ## Paths to WSI and HDF5
    wsi_path = f"{wsi_root_dir}{current_wsi}"
    hdf5_path = f"{project_root_dir}{hdf5_path}"
    print(wsi_path, hdf5_path)

    ## Load and map annotations to global WSI coordinates
    global_annotations = load_and_map_annotations_to_global(hdf5_path)

    ## Apply NMS on overlapping annotations
    filtered_annotations = apply_nms(global_annotations, iou_threshold=0.5, score_threshold=score_threshold)
    #filtered_annotations = apply_nms_cuda(global_annotations, iou_threshold=0.5, score_threshold=score_threshold, device=device)
    print('Annotation numbers before and after NMS filtering:', len(global_annotations), len(filtered_annotations))

    ## Extract patches and centroids
    centroids_and_patches = collect_bounding_boxes_from_wsi(wsi_path, filtered_annotations) #, num_workers=6)

    ## Get the maximum width and height among all image patches
    max_width = max(patch['patch'].size[0] for patch in centroids_and_patches)
    max_height = max(patch['patch'].size[1] for patch in centroids_and_patches)

    ## For uniform square padding
    max_dim = max(max_width, max_height)
    print(f"Padding size (square): {max_dim} x {max_dim}")

    patches, labels, scores = give_back_padded_dataset(centroids_and_patches, max_width=max_dim, max_height=max_dim) #, num_workers=8)

    # Save or further process the patches and labels as needed
    print("Processed patches:", patches.shape)
    print("Labels:", labels.shape)
    
    # Load query image to adjust statistics
    query_image = Image.open(query_image_path).convert("RGB")
    query_array = np.array(query_image)  # Convert to NumPy array if it's a PIL image

    # Compute mean and std per channel
    query_mean = query_array.mean(axis=(0, 1))
    query_std = query_array.std(axis=(0, 1))

    print(f"Mean per channel: {query_mean}")
    print(f"Std per channel: {query_std}")
    
    # Load model
    model = timm.create_model(
            'vit_small_patch14_dinov2.lvd142m', # vit_large_patch14_dinov2
            pretrained=True,
            img_size=518,
            init_values=1e-5,
            num_classes=0,  # Remove classifier nn.Linear
            )
    
    if len(vit_embedder_weights_path): # path is given -> finetuned weights
        stat_mean = np.array([0.485, 0.456, 0.406])
        stat_std = np.array([0.229, 0.224, 0.225])
        
        finetuned_state_dict = torch.load(vit_embedder_weights_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(finetuned_state_dict, strict=False)
        print(f"Model loaded with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")
        
    else: # default weights
        stat_mean = query_mean / 255
        stat_std = query_std / 255
    
    print('Statistics used to normalize:', stat_mean, stat_std)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=stat_mean, std=stat_std)
        ]
    )
    
    # Dataset and Dataloader 
    dataset = CollectionsDataset(data=patches,
                                    labels=np.ones(len(patches)),
                                    num_classes=1,
                                    transform=transform)

    dataset_loader = DataLoader(dataset,
                                    batch_size=64, # should fit into GPU memory ! HARDCODED !
                                    shuffle=False,
                                    num_workers=4)

    # Inference
    model.eval();
    model = model.to(device)
    
    print('Inference..')
    outputs_all = []
    labels_all = []
    with torch.no_grad():
        for bi, d in tqdm( enumerate(dataset_loader) ):
            inputs = d["image"]
            labels = d["label"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)

            outputs_all.append(outputs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    features = np.concatenate(outputs_all,axis=0)
    labels = np.concatenate(labels_all,axis=0)
    print('Embeddings shape:', features.shape)


    # CLUSTERING PROCESS !
    
    ## Step 1: Query image based clustering - use it's embedding
    preprocessed_query_image = preprocess_query_image(query_image_path, transform)

    # Add batch dimension and move to device if needed
    input_tensor = preprocessed_query_image.unsqueeze(0).to(device)

    # Compute the embedding
    with torch.no_grad():
        query_embedding = model(input_tensor).cpu().numpy()#.T
    print('Query image embedding shape:', query_embedding.shape)

    ## Calculation of distance metric
    cos_sim_dists = calculate_distances(features, query_embedding, metric="cosine")
    print('Bounds as is:', cos_sim_dists.min(), cos_sim_dists.max())
    #cos_sim_dists_rescaled = cos_sim_dists # (cos_sim_dists - cos_sim_dists.min()) / (cos_sim_dists.max() - cos_sim_dists.min())
    #print('Bounds scaled:', cos_sim_dists.min(), cos_sim_dists.max())


    ## Histogram and Gaussian fitting procedure to filter annotations
    cos_sim_dists_filt = cos_sim_dists < 0.8 # avoid large background peak's effect  ## HARCODED !
    bin_centers, smoothed_hist, hist = smooth_histogram(cos_sim_dists, bins=None)
    print(len(bin_centers))
    first_peak_x = find_first_peak(bin_centers, smoothed_hist)
    popt = fit_gaussian_to_rising_edge(bin_centers, smoothed_hist, first_peak_x)
    print('Gaussian parameters:', popt)

    mask = analyze_distribution(bin_centers, hist, smoothed_hist, popt, cos_sim_dists) # 1 sigma thresholded 
    print('Number of annotation kept after filtering:', mask.sum())

    # Collecting positive and negative samples
    
    # Universal percentile-based selection strategy (works across slides)
    bottom_20_threshold = np.percentile(cos_sim_dists, 20)
    top_20_threshold = np.percentile(cos_sim_dists, 80)
    
    if popt[1] > 0.5:
        print(f"Gaussian center too far out (center = {popt[1]:.3f}). Switching to percentile fallback: bottom 20% and top 20%")
        within_threshold = cos_sim_dists <= bottom_20_threshold
        outside_threshold = cos_sim_dists >= top_20_threshold
    else:
        mask = analyze_distribution(bin_centers, hist, smoothed_hist, popt, cos_sim_dists)  # 1 sigma thresholded
        within_threshold = mask
        outside_threshold = np.logical_and( cos_sim_dists > max( 0.5, popt[1] + 5*popt[2]), ~mask ) # outside of fitted gaussian

    within_patches = patches[within_threshold]
    outside_patches = patches[outside_threshold]

    # Advanced image filtering with conventional methods - chained after each other
    is_centered_bool = [ True if is_centered(img, tolerance=0.1) else False for img in within_patches]
    is_regular_shape_bool = [ True if is_regular_shape(img) else False for img in within_patches[is_centered_bool] ]

    updated_centroids_and_patches = []
    entries_all = np.array(centroids_and_patches)[within_threshold][is_centered_bool][is_regular_shape_bool]
    cos_sim_dists_entries_all = cos_sim_dists[within_threshold][is_centered_bool][is_regular_shape_bool]
    for e, entry in enumerate(entries_all):
        global_bbox = entry['bounding_box']
        patch = entry['patch']

        refined_result = refine_bbox_with_global_coords(patch, global_bbox)

        # Update the entry with new bbox and refined patch
        entry['bounding_box'] = refined_result["updated_bbox"]
        entry['patch'] = refined_result["refined_patch"]
        entry['cos_sim_dist'] = cos_sim_dists_entries_all[e]

        updated_centroids_and_patches.append(entry)
    
    final_centroids_and_patches = robust_circularity_filter( updated_centroids_and_patches, 
                                                             lower_ellipse_axis_ratio=lower_ellipse_axis_ratio, 
                                                             upper_ellipse_axis_ratio=upper_ellipse_axis_ratio
                                                            )
    
    # Filter by size -> strict to eliminate zoom-in or zoom-out annotations
    final_centroids_and_patches = filter_by_patch_size(final_centroids_and_patches)
    print(f"After size filtering: {len(final_centroids_and_patches)} remaining.")
    
    ## Process the patches of annotated objects -> each bbox gives one patch
    final_pollen_patches, _, _ = give_back_padded_dataset(final_centroids_and_patches, max_width=max_dim, max_height=max_dim) # , num_workers=4)
    
    if len(output_images_path):
        
        pollen_folder = os.path.join(project_root_dir + output_images_path, "pollen_grains")
        negative_folder = os.path.join(project_root_dir + output_images_path, "negative_samples")
        os.makedirs(pollen_folder, exist_ok=True)
        os.makedirs(negative_folder, exist_ok=True)
        
        # Subsampling for pollen grains (max 1000)
        pollen_sample_size = min(len(final_pollen_patches), 1000)
        pollen_indices = np.random.choice(len(final_pollen_patches), pollen_sample_size, replace=False)
        sampled_pollen_patches = [final_centroids_and_patches[i] for i in pollen_indices]
        
        # Subsampling for negative samples (max 1000)
        negative_sample_size = min(len(outside_patches), pollen_sample_size) # not more than positive samples
        negative_indices = np.random.choice(len(outside_patches), negative_sample_size, replace=False)
        sampled_negative_patches = outside_patches[negative_indices]
        
        # Saving pollen grains
        print(f"Saving {pollen_sample_size} pollen grain patches.")
        for idx, entry in enumerate(sampled_pollen_patches):
            patch = entry['patch']
            bbox = entry['bounding_box']
            x_min, y_min, x_max, y_max = bbox
            width, height = x_max - x_min, y_max - y_min

            #cropped_patch = remove_padding(patch)  # Remove padding if needed
            save_name = f"{current_wsi.replace('.tif', '')}_pollen_{idx}_X{x_min}_Y{y_min}_W{width}_H{height}.png"
            save_path = os.path.join(pollen_folder, save_name)
            patch.save(save_path)
        
        # Saving negative samples
        print(f"Saving {negative_sample_size} negative patches.")
        for idx, patch in enumerate(sampled_negative_patches):
            cropped_patch = remove_padding(patch) # Remove padding if needed
            save_path = os.path.join(negative_folder, f"{current_wsi.replace('.tif', '')}_negative_{idx}.png")
            Image.fromarray(cropped_patch).save(save_path)
        
        print("All patches saved successfully in structured folders.")

    if len(output_hdf5_path):
        print('Implement this functionality: save annotations in global coordinate system to hdf5 for later use with object detectors!')
        output_hdf5_path_full = f"{project_root_dir}{output_hdf5_path}"
        os.makedirs(os.path.dirname(output_hdf5_path_full), exist_ok=True) # if not exists        
        save_global_annotations_to_hdf5(final_centroids_and_patches, output_hdf5_path_full)