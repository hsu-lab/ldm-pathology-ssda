import argparse
import os
import pandas as pd

from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

def parse_arguments():
    """
    Parse command-line arguments for processing our WSIs given a csv file. 
    """
    parser = argparse.ArgumentParser(description="Process WSIs for one cohort")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the csv file to process")
    parser.add_argument("--job_dir", type=str, required=True, help="Directory to store outputs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use for processing tasks")
    parser.add_argument('--segmenter', type=str, default='hest', 
                        choices=['hest', 'grandqc',], 
                        help='Type of tissue vs background segmenter. Options are HEST or GrandQC.')
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5, 
                    help='Confidence threshold to apply to binarize segmentation predictions. Lower this threhsold to retain more tissue. Defaults to 0.5. Try 0.4 as 2nd option.')
    parser.add_argument('--auto_skip', action='store_true', help='Auto skip processed slides')

    return parser.parse_args()

def process_slide(args):
    """
    Process a single WSI by performing segmentation, patch extraction, and feature extraction sequentially.
    """

    # Initialize the WSI
    print(f"Processing slide: {args.slide_path}")
    slide = load_wsi(slide_path=args.slide_path, 
                    name=args.slide_id,
                    lazy_init=False, 
                    custom_mpp_keys=args.custom_mpp_keys
    )

    # Step 1: Tissue Segmentation
    print("Running tissue segmentation...")
    segmentation_model = segmentation_model_factory(
        model_name=args.segmenter,
        confidence_thresh=args.seg_conf_thresh,
    )
    slide.segment_tissue(
        segmentation_model=segmentation_model,
        target_mag=segmentation_model.target_mag,
        job_dir=args.job_dir,
        device=f"cuda:{args.gpu}"
    )
    print(f"Tissue segmentation completed. Results saved to {args.job_dir}/contours_geojson and {args.job_dir}/contours")

    # Step 2: Tissue Coordinate Extraction (Patching)
    print("Extracting tissue coordinates...")
    save_coords = os.path.join(args.job_dir, 'generated_tiles')

    coords_path = slide.extract_tissue_coords(
        target_mag=args.mag,
        patch_size=args.patch_size,
        save_coords=save_coords
    )
    print(f"Tissue coordinates extracted and saved to {coords_path}.")

    # Step 3: Visualize patching
    viz_coords_path = slide.visualize_coords(
        coords_path=coords_path,
        save_patch_viz=os.path.join(save_coords, 'visualization'),
    )
    print(f"Tissue coordinates extracted and saved to {viz_coords_path}.")

if __name__ == '__main__': 

    global_args = parse_arguments() 
    # csv_path = "/workspace/hsuraid/tengyuezhang/diffusion_luad/data/cohort/early_stage_luad/nlst_slides.csv"
    # job_dir = "/workspace/hsuraid/tengyuezhang/diffusion_luad/data/early_stage_luad/nlst/"
    csv_path = global_args.csv_path 
    job_dir = global_args.job_dir 

    # Load the slide CSV
    df = pd.read_csv(csv_path)

    # Iterate through all slides
    for idx, row in df.iterrows():
        pid = str(row['pid']) if isinstance(row['pid'], str) else str(int(row['pid']))
        slide_path = row['path']
        slide_id = row['slide_id']
        mpp = row['mpp']

        # # ======== v1: rina's tile size ========
        # if 0.23 < mpp < 0.27: # 0.25mpp 
        #     obj_mag = 40
        #     mag = 10 
        #     patch_size = 512 
        # elif 0.3 < mpp < 0.35: 
        #     obj_mag = 20 
        #     mag = 20 
        #     patch_size = 1024 
        # elif 0.48 < mpp < 0.52: 
        #     obj_mag = 20 
        #     mag = 20 
        #     patch_size = 1024
        # # =======================================

        # ======== v2: PathLDM's tile size ========
        if 0.23 < mpp < 0.27: # 0.25mpp 
            obj_mag = 40
            mag = 10 
            patch_size = 256 
        elif 0.3 < mpp < 0.35: 
            obj_mag = 20 
            mag = 20 
            patch_size = 512 
        elif 0.48 < mpp < 0.52: 
            obj_mag = 20 
            mag = 20 
            patch_size = 512
        # =========================================


        out_dir = os.path.join(job_dir, pid, slide_id)

        print(f"Processing {slide_id}...")

        if global_args.auto_skip and os.path.exists(os.path.join(out_dir, 'generated_tiles', 'patches', f'{slide_id}_patches.h5')):
            print(f'{pid}/{slide_id} has been tiled already. Skipped.')
            continue 
        process_args = argparse.Namespace(
            pid=pid,
            gpu=global_args.gpu,
            slide_id=slide_id,
            slide_path=slide_path,
            job_dir=out_dir,
            patch_encoder='conch_v15',
            mag=mag, # target mag
            patch_size=patch_size,
            segmenter=global_args.segmenter,
            seg_conf_thresh=global_args.seg_conf_thresh,
            custom_mpp_keys=None,
            overlap=0,
            batch_size=32
        )

        try:
            process_slide(process_args)
        except Exception as e:
            print(f"Failed to process {slide_id}: {e}")
