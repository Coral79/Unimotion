from argparse import ArgumentParser
import argparse
import os
import json
from copy import deepcopy


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def parse_and_load_from_multiple_models(parser, task=''):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    model_paths = args.model_path.split(',')
    args_list = []
    for i in range(len(model_paths)):
        new_args = deepcopy(args)
        new_args.model_path = model_paths[i]
        args_list.append(load_from_model(new_args, parser, task))
    
    if task == 'inpainting' and args.inpainting_mask == '':
        inpainting_mask = ','.join([args.inpainting_mask for args in args_list])
        for args in args_list:
            args.inpainting_mask = inpainting_mask
    print(f'Using inpainting mask: {args_list[0].inpainting_mask}')
    return args_list

def load_from_model(args, parser, task=''):
    args_to_overwrite = []
    loaded_groups = ['dataset', 'model', 'diffusion']
    if task in ['multi_sample', 'multi_train']:
        loaded_groups.append('multi_person')
    elif task == 'inpainting' and args.inpainting_mask == '':
        loaded_groups.append('inpainting')
    
    for group_name in loaded_groups:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    args.model_path = args.model_path if task != 'multi_train' else args.pretrained_path
    # load args from model
    args_path = os.path.join(os.path.dirname(args.model_path), 'args.json')
    print(args_path)
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            args.__dict__[a] = model_args[a]
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine', 'sqrt'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_clip", default=1.0, type=float, help="local text loss: for the frame-level text labels.") ### the loss for the frame-level text labels
    group.add_argument("--unconstrained", action='store_true',
                       help="If True, train an unconditional diffusion model.")
    group.add_argument("--separate_t", action='store_true', default=True,
                    help="Model is trained with separate vector for diffusion time step and global text description")

    group.add_argument("--multi_t", action='store_true', default=True,
                    help="Model is trained with multiple, separate diffusion time steps for local txt and motion")

    group.add_argument("--sin_time_embed", action='store_true', default=True,
                    help="embed diffusion time steps with mlp or sinusoidal")
    group.add_argument("--efficient_t", action='store_true',
                       help="multiple diffusion time steps only contain: tx=0, ty~[0,1000]; tx=ty~[0,1000]; tx~[0,1000], ty=0")##not much difference with this option
    group.add_argument("--pos_embed", default='sin',
                       choices=['learnable', 'sin'], type=str,
                       help="Positional Encoding.")
    group.add_argument("--data_concat", default='vertical',
                       choices=['vertical', 'horizontal'], type=str,
                       help="How to concat the local txt embedding and motion data.") ##vertial: temporla alignment is used in unimotion        


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml+', choices=['humanml', 'humanml+'], type=str,
                       help="Dataset to train on.") ##humaml+ means the overlapping part with babel which has both seq-level label and frame-level label
    group.add_argument("--pca", default=51, type=int,
                       help="pca dimension.")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--clip_rep", default='clip_enc_single', choices=['clip_enc_single', 'clip_enc', 'clip_enc_concat'], type=str,
                       help="clip_enc_single - multi-label frame chose first one; " ##this proved to be the best 
                            "clip_enc - multi-label frame average; "
                            "clip_enc_concat - multi-label frame concat with ',' ; ") 


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test_ft', choices=['val', 'test_ft', 'test_ft_no_overlap'], type=str,
                       help="Which split to evaluate on during training.test_ft is the overlap part of HumanML and BABEL; test_ft_no_overlap is test_ft exclude the training ids from the BABEL's training split") 
                       ##test_ft is the overlap part of HumanML and BABEL; test_ft_no_overlap is test_ft exclude the training ids from the BABEL's training split
    group.add_argument("--train_split", default='train_ft', type=str,
                       help="Which split for training data.")
    group.add_argument("--train_babel", action='store_true',
                       help="If True, will add babel into training data.")
    group.add_argument("--train_babel_split", default='train', type=str,
                       help="Which split for training data.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--save_results", action='store_true',
                       help="If True, will save the generated results from each eavlatuion during training, which can be saved for the evaluation script later.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--eval_condition", default='t2m', choices=['t2m', 'm2t', 'm+t', 't2m2t','m2t2m'], type=str,
                       help="t2m - frame-level text to motion; "
                            "m2t - motion to frame-level text; "
                            "m+t - joint generation for motion and text; "
                            "t2m2t - first t2m, then use the resulted motion as input for m2t"
                            "m2t2m - first m2t, then use the resulted frame-level text as input for t2m") 

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=1, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--sample_condition", default='t2m', choices=['t2m', 'm2t', 'm+t','t2m2t','m2t2m', 't2m2t2m2t'], type=str,
                       help="t2m - frame-level text to motion; "
                            "m2t - motion to frame-level text; "
                            "m+t - joint generation for motion and text; ") 
    group.add_argument("--sample_split", default='test_ft_no_overlap', type=str,
                       help="Which split for generating data.")    
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.0, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--embeds_path", default='./dataset/HumanML3D/pca/reduced_clip_embeddings_51.tsv', type=str,
                       help="Specify the embedding for the nearest neighbor")
    group.add_argument("--clip_path", default='./dataset/HumanML3D/pca/clip_embeddings.tsv', type=str,
                       help="Specify the clip embeddings for the full dimension")
    group.add_argument("--label_id_path", default='./dataset/HumanML3D/pca/label_to_id.json', type=str,
                       help="Specify the embedding for the nearest neighbor")
    group.add_argument("--txt_method", default='KNN', type=str,
                       help="Specify the method to retrieve the txt from the embeds")         
    group.add_argument("--inverse_pca", default=False, type=bool,
                       help="Find the neareast neighbour in the original clip spce with 512 dim")
    group.add_argument("--input_gt_local_txt", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")

def add_sampling_sub_options(parser):
    group = parser.add_argument_group('for_visual_results')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--embeds_path", default='./dataset/HumanML3D/pca/reduced_clip_embeddings_51.tsv', type=str,
                       help="Specify the embedding for the nearest neighbor")
    group.add_argument("--clip_path", default='./dataset/HumanML3D/pca/clip_embeddings.tsv', type=str,
                       help="Specify the clip embeddings for the full dimension")
    group.add_argument("--label_id_path", default='./dataset/HumanML3D/pca/label_to_id.json', type=str,
                       help="Specify the embedding for the nearest neighbor")
    group.add_argument("--txt_method", default='KNN', type=str,
                       help="Specify the method to retrieve the txt from the embeds")         
    group.add_argument("--inverse_pca", default=False, type=bool,
                       help="Find the neareast neighbour in the original clip spce with 512 dim")
    group.add_argument("--input_gt_local_txt", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full','small'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--eval_condition", default='t2m', choices=['t2m', 'm2t', 'm+t'], type=str,
                       help="t2m - frame-level text to motion; "
                            "m2t - motion to frame-level text; "
                            "m+t - joint generation for motion and text; ") 
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    
def add_visual_options(parser):
    group = parser.add_argument_group('visual')
    group.add_argument("--show_input", action='store_true',
                       help="If true, will show the motion from which the inpainting features were taken.")
    group.add_argument("--show_uncond", action='store_true',
                       help="If true, will show the motion generated with uncond guidance param 0.")
    group.add_argument("--vis_split", default='test_ft_no_overlap', type=str,
                       help="Which split for testing data.")
    group.add_argument("--result_folder", default='None', type=str,
                       help="Which folder to visualize result.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--input_motion_path", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body', 'prefix'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=-1, type=float,
                       help="For in_between editing - Defines the end of input prefix.")
    group.add_argument("--suffix_start", default=0, type=float,
                       help="For in_between editing - Defines the start of input suffix.")
    group.add_argument("--input_idx", default=0, type=int,
                       help="number of the indx input from the result numpy saved from previous motion generation.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full','small'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--test_split", default='test_val_no_overlap', type=str,
                       help="Which split for testing data.")
    group.add_argument("--eval_input", default='wo_gt', choices=['wo_gt', 'gt_local_txt'], type=str,
                       help="wo_gt (t2m) - for the evaluation input, without any gt; "
                            "gt_local_txt (t2m) - for the evaluation input, with gt local txt (clip embedding); ")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--eval_condition", default='t2m', choices=['t2m', 'm2t', 'm+t'], type=str,
                       help="t2m - frame-level text to motion; "
                            "m2t - motion to frame-level text; "
                            "m+t - joint generation for motion and text; ")


def add_inpainting_options(parser):
    group = parser.add_argument_group('inpainting')
    group.add_argument("--inpainting_mask", default='', type=str, 
                       help="Comma separated list of masks to use. In sampling, if not specified, will load the mask from the used model/s. \
                       Every element could be one of: \
                           gt_local_text (the gt local clip embeddings), \
                           root, root_horizontal, in_between, prefix, upper_body, lower_body, \
                           or one of the joints in the humanml body format: \
                           pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3, left_foot, \
                            right_foot, neck, left_collar, right_collar, head, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,")
    group.add_argument("--no_filter_noise", action='store_false', dest='filter_noise',
                       help="When true, the noise will be filtered from the inpainted features.")
    parser.set_defaults(filter_noise=True)


def add_edit_inpainting_options(parser):
    add_inpainting_options(parser)
    group = parser.add_argument_group('edit')
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--show_input", action='store_true',
                       help="If true, will show the motion from which the inpainting features were taken.")
    group.add_argument("--show_uncond", action='store_true',
                       help="If true, will show the motion generated with uncond guidance param 0.")
    group.add_argument("--input_gt_local_txt", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml', 'humanml+']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()

def train_inpainting_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_inpainting_options(parser)
    return parser.parse_args()
    
def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_visual_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    return args

def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    add_generate_options(parser)
    add_visual_options(parser)
    return parse_and_load_from_model(parser)

def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)

def eval_visual_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_visual_options(parser)
    add_sampling_sub_options(parser)
    return parse_and_load_from_model(parser)

def edit_inpainting_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_inpainting_options(parser)
    return parse_and_load_from_multiple_models(parser, task='inpainting')