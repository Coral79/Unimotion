# This code is based on https://github.com/openai/guided-diffusion
"""
For motion Editing
"""

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel_Multi_t
from data_loaders.get_data import  get_dataset_loader_eval
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion_pred
import shutil
from data_loaders.tensors import collate
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import json
import pandas as pd
from model.clip_encoder import Clip_Encoder
from os.path import join as pjoin
import copy

def main():
    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196
    fps = 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    dist_util.setup_dist(args.device)
    is_txt = False
    is_csv = False
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_{}_seed{}_gscale{}'.format(name, niter, args.sample_condition, args.seed, args.guidance_param))
        if args.input_motion_path != '':
            out_path += '_' + os.path.basename(args.input_motion_path).replace('.npy', '')

    if args.input_gt_local_txt != '':
        assert os.path.exists(args.input_gt_local_txt)
        if ".txt" in args.input_gt_local_txt:
            out_path += '_' + os.path.basename(args.input_gt_local_txt).replace('.txt', '').replace(' ', '_').replace('.', '')
            is_txt = True
        elif ".csv" in args.input_gt_local_txt:
            out_path += '_' + os.path.basename(args.input_gt_local_txt).replace('.csv', '').replace(' ', '_').replace('.', '')
            is_csv = True
        else:
            raise TypeError("Incorrect text file type, use csv or txt")
        if is_txt:
            with open(args.input_gt_local_txt, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            args.num_samples = 1
        elif is_csv:
            df = pd.read_csv(args.input_gt_local_txt)
            args.num_samples = 1
            n_frames = sum(df['length'])

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = get_dataset_loader_eval(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test_ft', ###test###
                              hml_mode='train',
                              pca=args.pca,
                              size=args.num_samples)  # in train mode, both text and motion.
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel_Multi_t(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  

    if args.guidance_param == 0:
        args.unconstrained = True

    input_clip = None
    collate_args = [{'inp': torch.zeros(model.njoints + model.local_txt_dim, model.nfeats, max_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    collate_args = [dict(arg, text='_') for arg in collate_args]
    
    input_motion_clip, model_kwargs = collate(collate_args)
    input_motion_clip = input_motion_clip.to(dist_util.dev()).permute(0, 3, 2, 1).squeeze(0).squeeze(1)
    try:
        input_m =  torch.tensor(np.load(args.input_motion_path)).to(dist_util.dev())
    except ValueError:
        input_npy = np.load(args.input_motion_path, allow_pickle=True).item()
        motion_data = np.expand_dims(input_npy['motion_emb'][args.input_idx], axis=0) 
        local_clip = np.expand_dims(input_npy['local_embed'][args.input_idx], axis=0) 
        input_m = torch.tensor(motion_data).to(dist_util.dev())[0].squeeze(1)
        input_clip = torch.tensor(local_clip).to(dist_util.dev())[0].squeeze(1)
    n_frame = min(max_frames, input_m.size(0))
    collate_args[0]['lengths'] = n_frame
    _, model_kwargs = collate(collate_args)
    input_motion_clip[:n_frame, :model.njoints] = input_m[:n_frame, :] 
    input_clip = input_clip[:n_frame, :].unsqueeze(0).unsqueeze(2)
    input_mean = torch.tensor(np.load(pjoin(data.dataset.opt.data_root, f'Mean_seg_pca_{args.pca}.npy'))).to(dist_util.dev())
    input_std = torch.tensor(np.load(pjoin(data.dataset.opt.data_root, f'Std_seg_pca_{args.pca}.npy'))).to(dist_util.dev()) 

    is_m2t = any([args.input_motion_path])
    if args.input_gt_local_txt != '':
        clip_version = 'ViT-B/32'
        clip_model = Clip_Encoder(clip_version)
        clip_model.to(dist_util.dev())
        local_text = []
        for i in range(len(list(df['text']))):
            clip_enc = clip_model(df['text'][i])
            for _ in range(df['length'][i]):
                local_text.append(clip_enc) 
        local_text_enc = torch.squeeze(torch.stack(local_text), dim=1).cpu()
        if args.pca > 0:
            from sklearn.decomposition import PCA
            # Load the embeddings from the TSV file
            embeddings_np = np.loadtxt(data.dataset.opt.pca_dir, delimiter='\t')
            # Initialize PCA
            pca = PCA()
            # Fit PCA on your data
            pca.fit(embeddings_np)
            print("Calculating PCA dim", args.pca)

            frame_label_pca = torch.tensor(pca.transform(local_text_enc)[:, :args.pca])
            input_motion_clip[:frame_label_pca.shape[0], -frame_label_pca.shape[-1]:] = frame_label_pca
            input_mean = torch.tensor(np.load(pjoin(data.dataset.opt.data_root, f'Mean_seg_pca_{args.pca}.npy'))).to(dist_util.dev())
            input_std = torch.tensor(np.load(pjoin(data.dataset.opt.data_root, f'Std_seg_pca_{args.pca}.npy'))).to(dist_util.dev())

        else:
            frame_label = frame_label.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1)
            input_motion_clip[:, -frame_label.shape[1]:, :, :frame_label.shape[-1]] = frame_label
            input_mean = np.load(pjoin(data.dataset.opt.data_root, f'Mean_seg_{args.pca}.npy'))
            input_std = np.load(pjoin(data.dataset.opt.data_root, f'Std_seg_{args.pca}.npy'))

    if is_m2t or args.input_gt_local_txt != '' :
        input_motion_clip = ((input_motion_clip - input_mean) / input_std).unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1).float()

    if args.multi_t:
        model_kwargs['y']['condition_motion'] = input_motion_clip.to(dist_util.dev())

    assert max_frames == input_motion_clip.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motion_clip[:,:model.njoints, :, :]
    if args.edit_mode == 'in_between':
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motion_clip[:,:model.njoints, :, :], dtype=torch.bool,
                                                               device=input_motion_clip.device)  # True means use input motion
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            start_idx, end_idx = int(args.prefix_end), int(args.suffix_start)
            gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
            model_kwargs['y']['inpainting_mask'][i, :, :,
            start_idx: end_idx] = False  # do inpainting in those frames
    elif args.edit_mode == 'prefix':
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motion_clip[:,:model.njoints, :, :], dtype=torch.bool,
                                                               device=input_motion_clip.device)  
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            start_idx = int(args.prefix_end * length)
            gt_frames_per_sample[i] = list(range(0, length))
            model_kwargs['y']['inpainting_mask'][i, :, :,
            start_idx: ] = False 

    gt_frames_per_sample = {}
    all_motions = []
    all_motions_emb = []
    all_lengths = []
    all_text = []
    all_clips = []
    all_input_clips = []
    all_condition_clips = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        dump_multi = False
        dump_steps=None 


        sample = sample_fn(
            model,
            (args.batch_size, model.njoints + model.local_txt_dim, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None, 
            progress=True,
            dump_steps= dump_steps,
            noise=None,
            const_noise=False,
            condition=args.sample_condition,
        )
        
        if dump_steps != None:
            args.num_samples = len(dump_steps)
            sample = torch.cat(sample, dim=0)
            dump_multi = True

        sample_clip = sample[:,model.njoints:, :, :]
        sample = sample[:,:model.njoints, :, :]
        input_motion = input_motion_clip[:,:model.njoints, :, :]
        condition_clip = input_motion_clip[:,model.njoints:, :, :]
        
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1), only_motion = True).float() # ([1, 1, 240, 263])
            emb_motion =copy.deepcopy(sample).permute(0, 2, 1, 3)
            sample = recover_from_ric(sample, n_joints)  
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            input_motion = data.dataset.t2m_dataset.inv_transform(input_motion.cpu().permute(0, 2, 3, 1), only_motion = True).float() # ([1, 1, 240, 263])
            input_motion = recover_from_ric(input_motion, n_joints) 
            input_motion = input_motion.view(-1, *input_motion.shape[2:]).permute(0, 2, 3, 1) 
        
    
        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False) # ([1, 22, 3, N_frame])

        input_motion = model.rot2xyz(x=input_motion, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False) # ([1, 22, 3, N_frame])

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions_emb.append(emb_motion.cpu().numpy())
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        all_clips.append(sample_clip.cpu().numpy())
        if input_clip is not None:
            all_input_clips.append(input_clip.cpu().numpy())
        if condition_clip is not None:
            all_condition_clips.append(condition_clip.cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    if dump_multi == False:
        all_motions = np.concatenate(all_motions, axis=0) 
        all_motions = all_motions[:total_num_samples]  # [bs, njoints, 3, seqlen]
        all_motions_emb = np.concatenate(all_motions_emb, axis=0) 
        all_motions_emb = all_motions_emb[:total_num_samples]  
    else:
        all_motions = all_motions[0]
        all_motions_emb = all_motions_emb[0]
   
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_clips = np.concatenate(all_clips, axis=0) 
    all_clips = np.squeeze(all_clips, axis=2)
    if input_clip is not None:
        all_input_clips = np.concatenate(all_input_clips, axis=0) 
        all_input_clips = np.squeeze(all_input_clips, axis=2)
    if condition_clip is not None:
        all_condition_clips = np.concatenate(all_condition_clips, axis=0) 
        all_condition_clips = np.squeeze(all_condition_clips, axis=2)

    if sample_clip is not None:
        all_local_id, all_local_txt =  predict_frame_txt(all_clips, args.embeds_path, args.clip_path, args.label_id_path, method = args.txt_method, inverse_pca=args.inverse_pca)
    else:
        all_local_id = None

    if input_clip is not None:
        all_input_local_id, all_input_local_txt =  predict_frame_txt(all_input_clips, args.embeds_path, args.clip_path, args.label_id_path, method = args.txt_method, inverse_pca=args.inverse_pca)
    else:
        all_input_local_id = None
    
    if condition_clip is not None:
        all_condition_local_id, all_condition_local_txt =  predict_frame_txt(all_condition_clips, args.embeds_path, args.clip_path, args.label_id_path, method = args.txt_method, inverse_pca=args.inverse_pca)
    else:
        all_condition_local_id = None

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'motion_emb': all_motions_emb, 'local_embed': all_clips, 'local_frame_txt': all_local_txt, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.t2m_kinematic_chain


    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    if model.data_rep == 'hml_vec':
        input_motion_clip = data.dataset.t2m_dataset.inv_transform(input_motion_clip.cpu().permute(0, 2, 3, 1)).float()
        input_motion_clip = recover_from_ric(input_motion_clip, n_joints)
        input_motion_clip = input_motion_clip.view(-1, *input_motion_clip.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

    for sample_i in range(args.num_samples):
        rep_files = []
        rep_files_txt = []
        if args.show_input:
            caption = 'Input Motion' 
            length = model_kwargs['y']['lengths'][sample_i]
            motion = input_motion_clip[sample_i].transpose(2, 0, 1)[:length]
            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            if all_input_local_id is not None:
                idx = all_input_local_id[rep_i*args.batch_size + sample_i]
                frame_labels = all_input_local_txt[rep_i*args.batch_size + sample_i]
                plot_3d_motion_pred(animation_save_path, skeleton, motion, dataset=args.dataset,  
                                   title=caption, idx = idx, clip_path = args.clip_path, label_id_path = args.label_id_path, fps=fps, label_frames= frame_labels)
                rep_files_txt.append(animation_save_path)

        length = model_kwargs['y']['lengths'][sample_i]
        input_motion = input_motion_clip[sample_i].transpose(2, 0, 1)[:length]                
        if all_condition_local_id is not None:
            input_idx = all_condition_local_id[rep_i*args.batch_size + sample_i][:length]
            input_frame_labels = all_condition_local_txt[rep_i*args.batch_size + sample_i][:length]

        for rep_i in range(args.num_repetitions):
            if dump_multi :
                caption = all_text[0] + f' setp_num: {dump_steps[sample_i]}'
                length = all_lengths[0]
            else:
                caption = all_text[rep_i*args.batch_size + sample_i]
                length = all_lengths[rep_i*args.batch_size + sample_i]
            if args.sample_condition=='m+t':
                motion = all_motions[rep_i*args.batch_size + sample_i][:22].transpose(2, 0, 1)
            else:
                motion = all_motions[rep_i*args.batch_size + sample_i][:22].transpose(2, 0, 1)[:length] #cut to the corresponding length
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            if all_local_id is not None:
                if args.sample_condition=='m+t':
                    frame_labels = all_local_txt[rep_i*args.batch_size + sample_i]
                    idx = all_local_id[rep_i*args.batch_size + sample_i]
                else:
                    frame_labels = all_local_txt[rep_i*args.batch_size + sample_i][:length]
                    idx = all_local_id[rep_i*args.batch_size + sample_i][:length]
                if args.sample_condition == 't2m':
                    plot_3d_motion_pred(animation_save_path, skeleton, motion, dataset=args.dataset,  
                        title=caption, idx = input_idx, clip_path = args.clip_path, label_id_path = args.label_id_path, fps=fps, label_frames= input_frame_labels)
                elif args.sample_condition == 'm2t':
                    plot_3d_motion_pred(animation_save_path, skeleton, input_motion, dataset=args.dataset,  
                        title=caption, idx = idx, clip_path = args.clip_path, label_id_path = args.label_id_path, fps=fps, label_frames= frame_labels)
                elif args.sample_condition == 'm+t':
                    plot_3d_motion_pred(animation_save_path, skeleton, motion, dataset=args.dataset,  
                        title=caption, idx = idx, clip_path = args.clip_path, label_id_path = args.label_id_path, fps=fps, label_frames= frame_labels)
                    rep_files_txt.append(animation_save_path)
            rep_files.append(animation_save_path)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template

def predict_frame_txt(local_embed, embeds_path, clip_path, label_id_path, method = 'KNN', inverse_pca=False):

    if inverse_pca:
        embeddings_np = np.loadtxt(clip_path, delimiter='\t')
    else:
        embeddings_np = np.loadtxt(embeds_path, delimiter='\t')

    print(f"calculating Nearest Neighbour in PCA space {embeddings_np.shape[1]} dimention")
    #  Initialize and Train KNN
    if method =="KNN":
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')  # Adjust n_neighbors as needed
        knn.fit(embeddings_np)
    else:
        print("we haven't implement other ways for prediction")
        sys.exit(1)

    # Open the JSON file and load its contents into a Python object
    with open(label_id_path, 'r') as file:
        texts_dict = json.load(file)
    inverse_texts_dict = {v: k for k, v in texts_dict.items()}

    num_seq = np.shape(local_embed)[0]
    num_frame = np.shape(local_embed)[2]
    predicted_clips = np.transpose(local_embed, (0, 2, 1))

    if inverse_pca:
        ####################load the PCA###############
        pca = PCA(n_components=predicted_clips.shape[-1])
        pca.fit(embeddings_np)
        # reconstructed_data = pca.inverse_transform(local_txt)

        reconstructed_slices = []
        for i in range(predicted_clips.shape[0]):
            # Flatten the 2D slice (51, 120) into a 2D array where each row is a sample for PCA
            # Assuming that the second dimension (120) represents different samples
            data_slice = predicted_clips[i]  # (120, 51)

            # Apply inverse_transform to reconstruct the data
            reconstructed_data = pca.inverse_transform(data_slice)  # This will have shape (120, 512)

            # # Reshape the reconstructed data back to its original 2D shape and store it
            reconstructed_slices.append(reconstructed_data)

        # Stack the reconstructed slices back together to match the original shape of local_txt
        reconstructed_clips = np.stack(reconstructed_slices, axis=0)
        pred_embed = reconstructed_clips
    else:
        pred_embed = predicted_clips


    # Open the JSON file and load its contents into a Python object
    with open(label_id_path, 'r') as file:
        texts_dict = json.load(file)


    text_all = []
    idx_all = []
    for i in range(num_seq):
        # # Step 5: KNN Search
        distances, indices = knn.kneighbors(pred_embed[i]) ###(fram_num,512)

        text_seq = []
        idx_seq = []
        # Assuming you have a dictionary where keys are indices of stored_clip_vectors and values are the corresponding text
        for clip_idx, neighbors in enumerate(indices):
            # print(f"Predicted Clip {clip_idx}:")
            for neighbor_idx in neighbors:
                # print(f"- {inverse_texts_dict[neighbor_idx]}")
                text_seq.append(inverse_texts_dict[neighbor_idx])
                idx_seq.append(neighbor_idx)

        text_all.append(text_seq)
        idx_all.append(idx_seq)
    
    return idx_all, text_all


if __name__ == "__main__":
    main()
