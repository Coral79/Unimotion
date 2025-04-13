"""
Utility functions for model creation and management.
"""

from model.unimotion import UnimotionBaseModel, UnimotionModel
from model.cfg_sampler import wrap_model
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, SpacedDiffusion_Multi_T, space_timesteps
from utils.parser_util import get_cond_mode
import torch

def load_model(args, data, device, ModelClass=UnimotionBaseModel):
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=ModelClass)
    model_path = args.model_path
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()  # disable random masking
    model = wrap_model(model, args)
    return model, diffusion


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('sin_pos_encoder.') for k in missing_keys])


def create_model_and_diffusion(args, data, ModelClass=UnimotionBaseModel, DiffusionClass=SpacedDiffusion):
    if args.multi_t:
        ModelClass=UnimotionModel
        DiffusionClass=SpacedDiffusion_Multi_T
    model = ModelClass(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args, DiffusionClass)
    return model, diffusion

def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    local_txt_dim = 0
    clip_dim = 512
    data_concat = args.data_concat

    if args.multi_t:
        args.separate_t = True

    if args.dataset == 'humanml+': 
        if args.pca >0:
            local_txt_dim = args.pca
        else:
            local_txt_dim = 512

    if args.dataset == 'humanml' or 'humanml+':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,       
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0 if args.pos_embed == 'learnable' else 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset, 
            'local_txt_dim': local_txt_dim, 'clip_dim': clip_dim, 'separate_t': args.separate_t,
            'multi_t': args.multi_t, 'sin_time_embed': args.sin_time_embed, 'pos_embed': args.pos_embed, 'data_concat': data_concat
            }


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_clip=args.lambda_clip,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )