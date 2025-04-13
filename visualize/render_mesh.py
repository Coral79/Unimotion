import sys
sys.path.append("/home/chuqiao/Code/Poject_2023/new_mdm/motion-diffusion-model/")
print (sys.path)
import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm
from visualize.convert import process_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    assert params.input_path.endswith('.mp4')
    if 'row' in params.input_path:
        parsed_name = os.path.basename(params.input_path).replace('.mp4', '').replace('sample', '').replace('row', '').replace('col', '').replace('rep', '')[:-2]
    else:
        parsed_name = os.path.basename(params.input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')[:-2]
    print(f"DEBUG Naama {parsed_name}")
    sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
    npy_path = os.path.join(os.path.dirname(params.input_path), 'results.npy')
    params.input_path_ = os.path.join(os.path.dirname(params.input_path), os.path.basename(params.input_path).replace('.mp4', ''), os.path.basename(params.input_path))
    # npy_path = os.path.join(os.path.dirname(os.path.dirname(params.input_path)), f'test_ft_no_overlap_rep_{rep_i}.npy')
    out_npy_path = params.input_path_.replace('.mp4', '_smpl_params.npy')
    out_local_txt_path = params.input_path_.replace('.mp4', '_frame_text.txt')
    out_global_txt_path = params.input_path_.replace('.mp4', '_global_text.txt')
    assert os.path.exists(npy_path)
    results_dir = params.input_path_.replace('.mp4', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=params.device, cuda=params.cuda)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)
    npy2obj.save_local_text(out_local_txt_path)
    npy2obj.save_global_text(out_global_txt_path)
    process_labels(out_local_txt_path)
