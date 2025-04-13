import torch
from torch.utils.data import Dataset
import numpy as np
from utils import dist_util
import os
import json
from data_loaders.humanml_utils import get_inpainting_mask
from tqdm import tqdm

def build_models(opt):
    if opt.model == 'unimotion':
        from .unimotion import UnimotionBaseModel
        model = UnimotionBaseModel(opt)
    else:
        raise ValueError(f'Model {opt.model} not implemented')
    return model

class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class CompMotionGeneratedDataset(Dataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., eval_input='wo_gt', save_results=False):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.w_vectorizer = dataloader.dataset.w_vectorizer
        self.model = model
        self.diffusion = diffusion
        self.args = args
        self.scale = scale
        self.eval_input = eval_input
        
        assert mm_num_samples < len(dataloader.dataset)
        
        # Initialize variables
        self.generated_motion = []
        self.mm_generated_motion = []
        
        # Get mm_idxs for multi-motion samples
        real_num_batches = num_samples_limit // dataloader.batch_size + 1 if num_samples_limit else len(dataloader)
        if mm_num_samples > 0:
            mm_idxs = np.sort(np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size + 1, replace=False))
        else:
            mm_idxs = []
        
        print('real_num_batches', real_num_batches)
        print('mm_idxs', mm_idxs)

        # Generate samples
        self._generate_samples(mm_idxs, mm_num_repeats, num_samples_limit)
        
        # Save generated samples if requested
        if save_results:
            self._save_resultserated_samples()

    def _setup_inpainting(self, motion, model_kwargs):
        """Setup inpainting mask and motion"""
        TXT_MASK = np.concatenate(([False]*(263),[True] * 512))
        model_kwargs['y']['inpainted_motion'] = motion.to(dist_util.dev())
        mask = torch.tensor(TXT_MASK, dtype=torch.bool, device=dist_util.dev())
        model_kwargs['y']['inpainting_mask'] = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(
            motion.shape[0], 1, motion.shape[2], motion.shape[3])

    def _prepare_model_kwargs(self, motion, model_kwargs):
        """Prepare model kwargs for generation"""
        if self.scale != 1.:
            model_kwargs['y']['scale'] = torch.ones(motion.shape[0], device=dist_util.dev()) * self.scale
        
        if self.eval_input == 'gt_local_txt':
            self._setup_inpainting(motion, model_kwargs)
            
        if self.model.multi_t:
            model_kwargs['y']['condition_motion'] = motion.to(dist_util.dev())
        
        return model_kwargs

    def _generate_sample(self, motion, model_kwargs):
        """Generate a single sample"""
        return self.diffusion.p_sample_loop(
            self.model,
            motion.shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
            condition=self.args.eval_condition,
        )

    def _process_batch(self, sample, motion, model_kwargs, tokens):
        """Process a batch of generated samples"""
        return [{
            'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
            'caption': model_kwargs['y']['text'][bs_i],
            'tokens': tokens[bs_i],
            'cap_len': len(tokens[bs_i]),
            'name': model_kwargs['y'].get('name', [None]*len(sample))[bs_i],
            'local_text': model_kwargs['y'].get('local_text', [None]*len(sample))[bs_i],
            'local_lengths': model_kwargs['y'].get('local_lengths', [None]*len(sample))[bs_i],
            'gt_motion': motion[bs_i].squeeze().permute(1,0).cpu().numpy(),
            'mask': model_kwargs['y']['mask'][bs_i].squeeze().cpu().numpy(),
        } for bs_i in range(len(sample))]

    def _process_mm_batch(self, sample, model_kwargs):
        """Process a batch for multi-motion generation"""
        return [{
            'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
        } for bs_i in range(len(sample))]

    def _create_mm_dicts(self, model_kwargs, tokens, mm_motions, batch_size):
        """Create dictionaries for multi-motion samples"""
        return [{
            'caption': model_kwargs['y']['text'][bs_i],
            'tokens': tokens[bs_i],
            'cap_len': len(tokens[bs_i]),
            'mm_motions': mm_motions[bs_i::batch_size],
        } for bs_i in range(batch_size)]

    def _generate_samples(self, mm_idxs, mm_num_repeats, num_samples_limit):
        """Main sample generation loop"""
        self.model.eval()
        with torch.no_grad():
            dataloader_iter = tqdm(self.dataloader, desc="Generating samples")
            for i, (motion, model_kwargs) in enumerate(dataloader_iter):
                if num_samples_limit and len(self.generated_motion) >= num_samples_limit:
                    break

                # Process tokens and prepare model inputs
                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]
                model_kwargs = self._prepare_model_kwargs(motion, model_kwargs)

                # Generate samples
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []

                for t in range(repeat_times):
                    sample = self._generate_sample(motion, model_kwargs)
                    
                    if t == 0:
                        batch_dicts = self._process_batch(sample, motion, model_kwargs, tokens)
                        self.generated_motion.extend(batch_dicts)

                    if is_mm:
                        mm_motions.extend(self._process_mm_batch(sample, model_kwargs))

                if is_mm:
                    self.mm_generated_motion.extend(
                        self._create_mm_dicts(model_kwargs, tokens, mm_motions, self.dataloader.batch_size)
                    )

    def _normalize_eval_motion(self, motion, motion_dim):
        """Normalize motion for evaluation mode"""
        normed_motion = motion
        denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)[:,:motion_dim]
        return (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval

    def _process_tokens(self, tokens):
        """Process tokens into embeddings and one-hot vectors"""
        word_embeddings = []
        pos_one_hots = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            word_embeddings.append(word_emb[None, :])
            pos_one_hots.append(pos_oh[None, :])
        return (np.concatenate(word_embeddings, axis=0),
                np.concatenate(pos_one_hots, axis=0))

    def _save_resultserated_samples(self):
        """Save generated samples to disk"""
        print("Save generated samples")
        save_path = os.path.join(self.args.log_file[:-4], self.args.eval_condition, 
                               f'rep_{self.args.current_replication}')
        self.args.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Save args
        self.args.split_file = self.dataset.split_file
        with open(os.path.join(save_path, 'args.json'), 'w') as fw:
            json.dump(vars(self.args), fw, indent=4, sort_keys=True)

        # Save generated motions
        if hasattr(self.args, 'test_split'):
            self.args.npy_path = os.path.join(save_path, self.args.test_split + f'_rep_{self.args.current_replication}' + '.npy')
        elif hasattr(self.args, 'eval_split'):
            self.args.npy_path = os.path.join(save_path, self.args.eval_split + f'_rep_{self.args.current_replication}' + '.npy')
        np.save(self.args.npy_path, self.generated_motion)

        # Save individual samples
        for gen_m in self.generated_motion:
            self._save_individual_sample(gen_m, save_path)

    def _save_individual_sample(self, gen_m, save_path):
        """Save an individual generated sample"""
        # Save kwargs
        kwargs = {
            'sum_length': int(gen_m['length']),
            'caption': gen_m['caption'],
            'text': gen_m['local_text'],
            'lengths': gen_m['local_lengths'],
        }
        with open(os.path.join(save_path, f"{gen_m['name']}.json"), 'w') as f:
            json.dump(kwargs, f, indent=4)

        # Save motion
        motion = torch.from_numpy(gen_m['motion']).unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1).float()
        if self.args.eval_condition == 't2m':
            motion = motion[:,:self.model.njoints, :, :]
        elif self.args.eval_condition == 'm2t':
            motion = motion[:,self.model.njoints:, :, :]
        torch.save(motion, os.path.join(save_path, f"{gen_m['name']}.pt"))

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion_dim = self.dataloader.dataset.opt.dim_pose
        
        # Extract basic data
        motion = data['motion']
        m_length = data['length']
        caption = data['caption']
        tokens = data['tokens']
        sent_len = data['cap_len']

        # Handle evaluation mode normalization
        if self.dataset.mode == 'eval':
            motion = self._normalize_eval_motion(motion, motion_dim)

        # Process word embeddings and one-hot vectors
        word_embeddings, pos_one_hots = self._process_tokens(tokens)
        
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompInpaintingMotionDataset(CompMotionGeneratedDataset):
    """Dataset class for inpainting-based motion generation from any motion model."""
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        super().__init__(args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale)
        self.max_motion_length = max_motion_length
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            dataloader_iter = tqdm(dataloader, desc="Generating samples")
            for i, (motion, model_kwargs) in enumerate(dataloader_iter):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                model_kwargs['y']['inpainted_motion'] = motion.to(dist_util.dev())
                model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape, seg_dim = args.seg_dim)).float().to(dist_util.dev())

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion_dim = self.dataloader.dataset.opt.dim_pose
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)[:,:motion_dim]
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)