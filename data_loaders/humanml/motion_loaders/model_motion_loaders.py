from torch.utils.data import DataLoader, Dataset
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import CompMotionGeneratedDataset, CompInpaintingMotionDataset
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
import numpy as np
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.motion_dim = motion_dataset.dataset.opt.dim_pose
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        motion_dim = self.motion_dim
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            motion = motion[None, :][:,:,:motion_dim]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens



def get_motion_loader(opt_path, batch_size, ground_truth_dataset, mm_num_samples, mm_num_repeats, device):
    opt = get_opt(opt_path, device)

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    if 'v6' in opt.name:
        dataset = CompMotionGeneratedDataset(opt, ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)
    else:
        raise KeyError('Dataset not recognized!!')

    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader


# generated motion loader
def get_generated_motion_loader(args, model, diffusion, batch_size, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale, eval_input = 'wo_gt', save_results=False):
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])

    if hasattr(args, "inpainting_mask") and args.inpainting_mask != '':
        dataset = CompInpaintingMotionDataset(args, model, diffusion, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale)
    else:
        dataset = CompMotionGeneratedDataset(args, model, diffusion, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale, eval_input, save_results=save_results)

    mm_dataset = MMGeneratedDataset(opt, dataset, ground_truth_loader.dataset.w_vectorizer)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=False, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader