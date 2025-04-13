import os
import random
import numpy as np
import codecs as cs
import torch
import spacy
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt, get_opt_w_BABEL


def collate_fn(batch):
    """
    Collate function for PyTorch dataloader. 
    Sorts batch items in descending order by a specific key (x[3]) before default collating.
    """
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


import os
import random
import numpy as np
import json
import tqdm
import codecs as cs
from os.path import join as pjoin
from torch.utils.data import Dataset

""""Base Dataset Class"""
class BaseText2MotionDataset(Dataset):
    """
    A base class that unifies the logic of loading:
      - motion from disk
      - optional local label frames
      - optional PCA reduction
      - text/captions
    and building an internal data_dict.
    
    Child classes or runtime flags can control whether
    we load local text or global text or both,
    whether we return w_local/w_global, etc.
    """
    def __init__(
        self,
        opt,
        mean,
        std,
        split_file,
        w_vectorizer,
        num_frames=None,
        size=None,
        is_babel=False,
        babel_split_file=None,
        load_segmentation=False,
        return_local_global_flags=False,
        **kwargs
    ):
        super().__init__()
        
        # Basic attributes
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.mean = mean
        self.std = std
        self.split_file = split_file
        self.split = os.path.splitext(os.path.basename(split_file))[0]
        self.max_motion_length = opt.max_motion_length
        self.max_length = 20  # Some default max text length (can be changed)
        self.pointer = 0
        
        # Decide the min_motion_len rule
        if not num_frames or isinstance(num_frames, int):
            self.num_frames = num_frames
            # Based on dataset_name (t2m or something else)
            self.min_motion_len = 40 if opt.dataset_name == 't2m' else 24
            # Special case: test_ft_no_overlap => min_motion_len=0
            if 'test_ft_no_overlap' in self.split:
                self.min_motion_len = 0
        else:
            # num_frames is a tuple => [min, max]
            self.num_frames = num_frames
            self.min_motion_len = num_frames[0]
            self.max_motion_length = num_frames[1]

        # Additional flags
        self.is_babel = is_babel                # If we also want to load non-overlap part from BABEL
        self.babel_split_file = babel_split_file
        self.load_segmentation = load_segmentation
        self.return_local_global_flags = return_local_global_flags

        # Handle PCA
        self.pca = opt.pca if hasattr(opt, 'pca') else 0
        if self.pca > 0:
            from sklearn.decomposition import PCA
            embeddings_np = np.loadtxt(opt.pca_dir, delimiter='\t')
            pca_model = PCA()
            pca_model.fit(embeddings_np)
            print(f"Calculating PCA with dim={self.pca}")
            self.pca_model = pca_model
        else:
            self.pca_model = None

        # Prepare data structures
        self.data_dict = {}
        self.name_list = []
        self.length_arr = None
        
        # If any special JSON
        self.chosen_file = None
        
        # Build dataset
        self._build_data_dict(size=size)
        
        # Sort and finalize
        self._finalize_data()

        # print('name_list', self.name_list)
        # print('length', len(self.name_list))

    def _build_data_dict(self, size=None):
        """
        Reads the split file (and possibly BABEL file), loads motion & text,
        applies segmentation / PCA if needed, populates self.data_dict.
        """
        # 1) Load ID list from the main split_file
        id_list = self._read_id_list(self.split_file)
        if size is not None:
            id_list = id_list[:size]

        # 2) If using BABEL data, read from babel_split_file
        babel_list = []
        if self.is_babel and self.babel_split_file:
            babel_list = self._read_id_list(self.babel_split_file)

        # 3) Process all IDs (HumanML3D)
        for name in tqdm.tqdm(id_list, desc="Processing HumanML3D IDs"):
            self._process_single_id(name, is_babel=False)

        # 4) Process BABEL IDs if needed
        for name in tqdm.tqdm(babel_list, desc="Processing BABEL IDs"):
            self._process_single_id(name, is_babel=True)

    def _process_single_id(self, name, is_babel):
        try:
            # (1) load motion_combined
            if is_babel:
                motion_name = '0' + name
                motion_path = pjoin(self.opt.babel_motion_dir, motion_name + '.npy')
                label_path = pjoin(self.opt.babel_label_dir, motion_name + '.npy')
            else:
                motion_name = name
                motion_path = pjoin(self.opt.motion_dir, motion_name + '.npy')
                label_path = pjoin(self.opt.label_dir, motion_name + '.npy') \
                            if self.load_segmentation else None

            motion = np.load(motion_path)
            if len(motion) < self.min_motion_len or len(motion) >= 200:
                return

            # (2) optionally load segmentation labels & apply PCA
            w_local = False
            if label_path and os.path.exists(label_path):
                frame_label = np.load(label_path)
                w_local = True
            else:
                frame_label = None

            if frame_label is not None and self.pca_model is not None:
                frame_label = self.pca_model.transform(frame_label)[:, :self.pca]

            motion_combined = self._combine_motion_and_label(motion, frame_label)

            # (3) Load text lines & create sub-entries if needed
            if is_babel:
                # BABEL => dummy text, or skip the sub-slicing
                text_data = [{'caption': 'None', 'tokens': ['None']}]
                w_global = False
                new_entry = {
                    'motion': motion_combined,
                    'length': len(motion_combined),
                    'text': text_data
                }
            else:
                text_data, flag = self._load_text_data(name, motion_combined)
                w_global = True
                # Now `text_data` are the lines with (f_tag,to_tag)==0 => for the full motion
                # `flag` indicates if we actually saw any (0,0).

                # (4) If `flag` => we store the entire motion under this `name`.
                if flag:
                    new_entry = {
                        'motion': motion_combined,
                        'length': len(motion_combined),
                        'text': text_data
                    }
                else:
                    # If we never saw (0,0), we might not want to store the full motion at all.
                    # Return or skip
                    return

            # (5) Possibly store local/global flags
            if self.return_local_global_flags:
                new_entry['w_local'] = w_local
                new_entry['w_global'] = w_global

            final_name = f"babel_{motion_name}" if is_babel else name
            self.data_dict[final_name] = new_entry

        except Exception as e:
            pass


    def _combine_motion_and_label(self, motion, frame_label):
        """
        If frame_label is not None, concatenate. Otherwise just return motion.
        """
        if frame_label is None:
            return motion
        else:
            return np.concatenate((motion, frame_label), axis=1)


    def _load_text_data(self, name, motion_combined):
        """
        Loads text lines from self.opt.text_dir/<name>.txt, then replicates
        the old slicing logic:
        - If in a special split (test_ft_no_overlap, etc.), pick the single largest segment.
        - Else parse all lines.
        - For each line, if (f_tag, to_tag) == (0,0) => mark "flag", so we store the
            full motion later.
        - Otherwise slice sub-motion, store a brand-new entry in self.data_dict with
            a random prefix. Possibly replace the original name.

        Returns:
        text_data: (list) for the lines that had (f_tag,to_tag) == (0,0)
        flag: (bool) indicating whether we found at least one line with (0,0).
        """
        import random
        import string
        text_file_path = pjoin(self.opt.text_dir, name + '.txt')
        text_data = []
        flag = False  # if we ever see (f_tag,to_tag) == (0,0), set this so we store the entire motion
        if not os.path.exists(text_file_path):
            # no text => return an empty entry but still treat it as global
            return text_data, False

        with cs.open(text_file_path, 'r') as f2:
            lines = f2.readlines()

        # Decide if we pick the single largest segment or process all lines
        special_splits = ['test_ft_no_overlap', 'val_ft_no_overlap']
        if any(x in self.split for x in special_splits):
            # 1) parse all lines
            line_splits = [l.strip().split('#') for l in lines]
            captions = [x[0] for x in line_splits]
            tokens_set = [x[1].split(' ') for x in line_splits]
            f_tags = [float(x[2]) for x in line_splits]
            to_tags = [float(x[3]) for x in line_splits]

            # clean up possible NaNs
            f_tags = [0.0 if np.isnan(v) else v for v in f_tags]
            to_tags = [0.0 if np.isnan(v) else v for v in to_tags]

            # 2) compute lengths of each segment
            lengths = [int((to_ - fr_) * 20) for (fr_, to_) in zip(f_tags, to_tags)]
            lengths = [l if l > 0 else 99999 for l in lengths]
            # pick the single largest
            chosen_idx = None
            if len(lengths) > 0:
                chosen_idx = lengths.index(max(lengths))

            # 3) only keep that line
            for i, (cap, toks, fr_, to_) in enumerate(zip(captions, tokens_set, f_tags, to_tags)):
                if chosen_idx is not None and i != chosen_idx:
                    continue

                text_dict = {'caption': cap, 'tokens': toks}
                if fr_ == 0.0 and to_ == 0.0:
                    # Mark that we will store the entire motion later
                    flag = True
                    text_data.append(text_dict)
                else:
                    # try slicing sub-motion
                    try:
                        n_motion = motion_combined[int(fr_ * 20) : int(to_ * 20)]
                        if (len(n_motion) < self.min_motion_len) or (len(n_motion) >= 200):
                            continue
                        new_name = random.choice(string.ascii_uppercase) + '_' + name
                        while new_name in self.data_dict:
                            new_name = random.choice(string.ascii_uppercase) + '_' + name
                        self.data_dict[new_name] = {
                            'motion': n_motion,
                            'length': len(n_motion),
                            'text': [text_dict]
                        }

                        # If we only keep that largest sub-motion => replace the original
                        if chosen_idx is not None:
                            self.data_dict[name] = self.data_dict[new_name]
                            del self.data_dict[new_name]
                    except Exception:
                        pass

        else:
            # Non-special splits => parse all lines, each might create a sub-entry
            for line_ in lines:
                line_split = line_.strip().split('#')
                cap = line_split[0]
                toks = line_split[1].split(' ')
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                
                text_dict = {'caption': cap, 'tokens': toks}
                if f_tag == 0.0 and to_tag == 0.0:
                    flag = True
                    text_data.append(text_dict)
                else:
                    # slice sub-motion
                    try:
                        n_motion = motion_combined[int(f_tag * 20) : int(to_tag * 20)]
                        if (len(n_motion) < self.min_motion_len) or (len(n_motion) >= 200):
                            continue
                        new_name = random.choice(string.ascii_uppercase) + '_' + name
                        while new_name in self.data_dict:
                            new_name = random.choice(string.ascii_uppercase) + '_' + name

                        self.data_dict[new_name] = {
                            'motion': n_motion,
                            'length': len(n_motion),
                            'text': [text_dict]
                        }
                    except Exception:
                        pass

        return text_data, flag


    def _parse_text_line(self, line_):
        """
        Splits line by '#', returns (caption, tokens, f_tag, to_tag).
        If anything fails, return None.
        """
        line_split = line_.strip().split('#')
        if len(line_split) < 4:
            return None, None, None, None
        caption = line_split[0]
        tokens_ = line_split[1].split(' ')
        try:
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
        except:
            return None, None, None, None
        return caption, tokens_, f_tag, to_tag

    def _build_text_data(self, motion, motion_combined, caption, tokens, f_tag, to_tag):
        """
        Builds text_data entries. If (f_tag,to_tag)!=0 => sub-motion.
        If 0 => entire motion. 
        In some versions, you might store new sub-entries in data_dict,
        but we’ll just return the text data here for simplicity.
        """
        text_dict = {'caption': caption, 'tokens': tokens}
        # If f_tag == 0 & to_tag == 0 => entire motion is used
        # If not => sub-motion
        # In some classes, the code created brand-new entries in data_dict 
        # but that can also be handled by the child class if needed.
        return [text_dict]

    def _read_id_list(self, split_file):
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f:
                id_list.append(line.strip())
        return id_list

    def _finalize_data(self):
        """
        Sort by length, set up pointer, load chosen_file if needed, etc.
        """
        # Build name_list / length_list from data_dict
        tmp = []
        for k, v in self.data_dict.items():
            tmp.append((k, v['length']))
        tmp.sort(key=lambda x: x[1])  # sort by length
        self.name_list = [x[0] for x in tmp]
        self.length_arr = np.array([x[1] for x in tmp])

        # Reset pointer
        self.reset_max_len(self.max_length)

        # If there's a special JSON file for test/val
        if 'test_ft_no_overlap' in self.split:
            # or 'val_global', etc.
            try:
                with open('./dataset/HumanML3D/humanml_babel_test_set_local.json', 'r') as file:
                    self.chosen_file = json.load(file)
            except:
                self.chosen_file = None

    def reset_max_len(self, length):
        """
        Moves pointer so that only motions of length >= `length` are hidden.
        Because we sorted, pointer is the first idx where length >= desired.
        """
        if length > self.max_motion_length:
            length = self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer pointing at", self.pointer)
        self.max_length = length

    def inv_transform(self, data, only_motion=False):
        """
        Inverse normalization. If only_motion=True and pca>0, slice dims.
        """
        if only_motion and self.pca > 0:
            # assume last self.pca dims are the label
            return data * self.std[:-self.pca] + self.mean[:-self.pca]
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, idx):
        """
        Typical final step: 
          - get the motion
          - pick a random text from 'text'
          - tokenize, embed
          - random crop motion to multiple of unit_length
          - normalize
        """
        real_idx = self.pointer + idx
        name = self.name_list[real_idx]
        data_item = self.data_dict[name]

        motion = data_item['motion']
        m_length = data_item['length']
        text_list = data_item['text']

        if self.return_local_global_flags:
            w_local = data_item['w_local']
            w_global = data_item['w_global']
        else:
            w_local, w_global = None, None

        # pick a random caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        # Possibly get local txt, local_lengths from chosen_file
        local_txt, local_lengths = None, None
        if self.chosen_file:
            matches = [x for x in self.chosen_file if x.get('original_id') == name]
            if matches:
                local_txt = matches[0].get('text')
                local_lengths = matches[0].get('lengths')

        # 1) Convert tokens => embeddings
        if 'babel_' in name:
            pos_one_hots = np.zeros((22, 15))
            word_embeddings = np.zeros((22, 300))
            sent_len = 22
        else:
            pos_one_hots = []
            word_embeddings = []

            try:
                word_embeddings, pos_one_hots, sent_len, tokens = self._token_to_embeddings(tokens)
            except:
                print(name)

        # 2) Crop motion to multiple of unit_length
        motion, m_length = self._crop_motion(motion, m_length)

        # 3) Normalize
        motion = self._normalize_motion(motion, m_length)

        # 4) Pad to max_motion_length
        if m_length < self.max_motion_length:
            pad_len = self.max_motion_length - m_length
            pad_zeros = np.zeros((pad_len, motion.shape[1]))
            motion = np.concatenate([motion, pad_zeros], axis=0)

        # Build output
        result = (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
        )
        if self.load_segmentation:
            result += (name, local_txt, local_lengths)
        # If we want local/global flags, we can append them or keep them separate
        if self.return_local_global_flags:
            result += (w_local, w_global)
        return result

    def _token_to_embeddings(self, tokens):
        """
        Convert tokens => (word_embeddings, pos_one_hots), pad if needed,
        manage sent_len.  The code below is a simplified example.
        """
        # clamp or pad tokens
        max_txt_len = self.opt.max_text_len
        if len(tokens) < max_txt_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens += ['unk/OTHER'] * (max_txt_len + 2 - sent_len)
        else:
            tokens = tokens[:max_txt_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_list, emb_list = [], []
        for tok in tokens:
            word_emb, pos_oh = self.w_vectorizer[tok]
            pos_list.append(pos_oh[None, :])
            emb_list.append(word_emb[None, :])

        pos_one_hots = np.concatenate(pos_list, axis=0)
        word_embeddings = np.concatenate(emb_list, axis=0)
        return word_embeddings, pos_one_hots, sent_len, tokens

    def _crop_motion(self, motion, m_length):
        """
        Crop motion length to multiple of unit_length using random start.
        Optionally do 'double' cropping if < 10, etc.
        """
        unit_length = self.opt.unit_length
        if m_length <= 200:
            # random coin toss
            if unit_length < 10:
                coin2 = np.random.choice(['single','single','double'])
            else:
                coin2 = 'single'
            if coin2 == 'double':
                m_length = (m_length // unit_length - 1) * unit_length
            else:
                m_length = (m_length // unit_length) * unit_length
            
            if m_length < 1:
                m_length = unit_length  # fallback if something weird happens
            start_idx = random.randint(0, len(motion) - m_length)
            motion = motion[start_idx : start_idx + m_length]
        else:
            # for bigger motions (BABEL?), just clamp to max_motion_length
            start_idx = random.randint(0, len(motion) - self.max_motion_length)
            motion = motion[start_idx : start_idx + self.max_motion_length]
            m_length = self.max_motion_length
        
        return motion, m_length

    def _normalize_motion(self, motion, m_length):
        """
        Z-normalization: 
          If we have a PCA dimension, we only normalize the motion portion,
          or we do the entire combined dimension. 
          (Simplify as needed.)
        """
        if self.pca > 0:
            # Typically you'd do something like:
            # motion[...,:-pca] = (motion[...,:-pca] - mean[:-pca]) / std[:-pca]
            # but let's keep it simple:
            motion = (motion - self.mean) / self.std
        else:
            motion = (motion - self.mean) / self.std
        return motion

""""Only global text (e.g. HumanML3D or T2M) & no per-frame local text/labels"""
class GlobalOnlyText2MotionDataset(BaseText2MotionDataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, num_frames=None, size=None, **kwargs):
        # No PCA, no BABEL
        super().__init__(
            opt=opt, mean=mean, std=std,
            split_file=split_file, w_vectorizer=w_vectorizer,
            num_frames=num_frames, size=size,
            is_babel=False,
            load_segmentation=False,
            return_local_global_flags=False,
            **kwargs
        )

""""global text could be missing; per-frame local text always avaliable (e.g. Full BABEL)"""
class LocalText2MotionDataset_OptionalGlobal(BaseText2MotionDataset):
    def __init__(
        self, opt, mean, std, split_file, babel_split_file, w_vectorizer, num_frames=None, size=None, **kwargs
    ):
        # Combine HumanML3D + BABEL with segmentation + local/global flags
        super().__init__(
            opt=opt, mean=mean, std=std,
            split_file=split_file, w_vectorizer=w_vectorizer,
            num_frames=num_frames, size=size,
            is_babel=True,
            babel_split_file=babel_split_file,
            load_segmentation=True,
            return_local_global_flags=True,
            **kwargs
        )

""""global text always avaliable; per-frame local text could be missing (Full HumanML3D)"""
class GlobalText2MotionDataset_OptionalLocal(BaseText2MotionDataset):
    def __init__(
        self, opt, mean, std, split_file, w_vectorizer, num_frames=None, size=None, **kwargs
    ):
        # Segmentation, but only HumanML3D
        super().__init__(
            opt=opt, mean=mean, std=std,
            split_file=split_file, w_vectorizer=w_vectorizer,
            num_frames=num_frames, size=size,
            is_babel=False,
            load_segmentation=True,
            return_local_global_flags=True,
            **kwargs
        )

""""global text and per-frame local text both avaliable (overlapping part of HumanML3D and BABEL)"""
class GlobalAndLocalText2MotionDataset(BaseText2MotionDataset):
    def __init__(
        self, opt, mean, std, split_file, w_vectorizer, num_frames=None, size=None, **kwargs
    ):
        # Seg + HumanML3D only, no local/global flags
        super().__init__(
            opt=opt, mean=mean, std=std,
            split_file=split_file, w_vectorizer=w_vectorizer,
            num_frames=num_frames, size=size,
            is_babel=False,
            load_segmentation=True,
            return_local_global_flags=False,
            **kwargs
        )


"""text-only dataset for cases where only need text inputs without any corresponding motion data"""
class TextOnlyDataset(data.Dataset):
    """
    A dataset that only has text (no motion) for cases where we need to 
    handle text inputs without any corresponding motion data.
    """
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        try:
            self.pca = opt.pca
        except AttributeError:
            self.pca = 0

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f:
                id_list.append(line.strip())

        new_name_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f2:
                    for line_ in f2:
                        text_dict = {}
                        line_split = line_.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text': [text_dict]}
                                new_name_list.append(new_name)
                            except Exception:
                                pass

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except Exception:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data, only_motion=False):
        if only_motion and self.pca > 0:
            return data * self.std[:-self.pca] + self.mean[:-self.pca]
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data_item = self.data_dict[self.name_list[idx]]
        text_list = data_item['text']

        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        # Return dummy placeholders (no motion).
        return None, None, caption, None, np.array([0]), self.fixed_length, None


class HumanML3D(data.Dataset):
    """
    A wrapper class for the original Text2MotionDataset (for T2M) or derived classes, 
    providing an interface expected by MDM.
    """
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", **kwargs):
        self.mode = mode
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt

        if mode == 'gt':
            # For the T2M evaluator
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # For our usage
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # Translation between their norms and ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalOnlyText2MotionDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1

        assert len(self.t2m_dataset) > 1, (
            "You loaded an empty dataset. Ensure you have the full data as described in the README."
        )

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

class HumanML3D_BABEL_overlap(data.Dataset):
    """
    A wrapper class for HumanML3D and BABEL overlapping dataset with both global and local text. 
    Includes options for PCA-based dimensionality reduction on frame-wise labels.
    """
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train_ft", **kwargs):
        self.mode = mode
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.label_dir = pjoin(abs_base_path, opt.label_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'

        # Add PCA
        opt.pca = kwargs['pca']
        opt.pca_dir = pjoin(abs_base_path, opt.pca_dir)
        self.opt = opt

        if mode == 'gt':
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # For our usage
            if opt.pca > 0:
                self.mean = np.load(pjoin(opt.data_root, f'Mean_seg_pca_{opt.pca}.npy'))
                self.std = np.load(pjoin(opt.data_root, f'Std_seg_pca_{opt.pca}.npy'))
            else:
                self.mean = np.load(pjoin(opt.data_root, 'Mean_seg.npy'))
                self.std = np.load(pjoin(opt.data_root, 'Std_seg.npy'))

        if mode == 'eval':
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        elif mode == 'gt':
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalOnlyText2MotionDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1
            # print("length of dataset: ", self.t2m_dataset.__len__())
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalAndLocalText2MotionDataset(
                self.opt, self.mean, self.std, self.split_file, self.w_vectorizer
            )
            self.num_actions = 1

        assert len(self.t2m_dataset) >= 1, (
            "You loaded an empty dataset. Ensure you have the full data as described in the README."
        )

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return len(self.t2m_dataset)
    
class Babel_full(data.Dataset):
    """
    Full BABEL dataset, with possible missing global text, while local text is always available.
    """
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train_ft", **kwargs):
        self.mode = mode
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None
        opt = get_opt_w_BABEL(dataset_opt_path, device, kwargs['args'])
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.label_dir = pjoin(abs_base_path, opt.label_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.babel_data_root = pjoin(abs_base_path, opt.babel_data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'

        opt.pca = kwargs['pca']
        opt.pca_dir = pjoin(abs_base_path, opt.pca_dir)
        opt.args = kwargs['args']
        self.opt = opt

        if mode == 'gt':
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # For our usage
            if opt.pca > 0:
                print("Loading mean/std with PCA from:", pjoin(opt.babel_data_root))
                self.mean = np.load(pjoin(opt.babel_data_root, f'Mean_seg_pca_{opt.pca}.npy'))
                self.std = np.load(pjoin(opt.babel_data_root, f'Std_seg_pca_{opt.pca}.npy'))
            else:
                self.mean = np.load(pjoin(opt.babel_data_root, 'Mean_seg.npy'))
                self.std = np.load(pjoin(opt.babel_data_root, 'Std_seg.npy'))

        if mode == 'eval':
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        babel_split = kwargs['args'].train_babel_split
        self.babel_split_file = pjoin(opt.babel_data_root, f'{babel_split}.txt')

        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        elif mode == 'gt':
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalOnlyText2MotionDataset(
                self.opt, self.mean, self.std, self.split_file, self.w_vectorizer
            )
            self.num_actions = 1
        elif mode == 'eval':
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalAndLocalText2MotionDataset(
                self.opt, self.mean, self.std, self.split_file, self.w_vectorizer
            )
            self.num_actions = 1
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = LocalText2MotionDataset_OptionalGlobal(
                self.opt, self.mean, self.std, self.split_file,
                self.babel_split_file, self.w_vectorizer
            )
            self.num_actions = 1

        assert len(self.t2m_dataset) > 1, (
            "You loaded an empty dataset. Ensure you have the full data as described in the README."
        )

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

class HumanML3D_Babel_union(data.Dataset):
    """
    Merges HumanML3D and BABEL, local/global text could be missing.
    """
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train_ft", **kwargs):
        self.mode = mode
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None
        opt = get_opt_w_BABEL(dataset_opt_path, device, kwargs['args'])
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.label_dir = pjoin(abs_base_path, opt.label_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.babel_data_root = pjoin(abs_base_path, opt.babel_data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'

        opt.pca = kwargs['pca']
        opt.pca_dir = pjoin(abs_base_path, opt.pca_dir)
        opt.args = kwargs['args']
        self.opt = opt

        if mode == 'gt':
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # For our usage
            if opt.pca > 0:
                print("Loading merged mean/std with PCA from:", pjoin(opt.babel_data_root))
                self.mean = np.load(pjoin(opt.babel_data_root, f'Mean_both_seg_pca_{opt.pca}.npy'))
                self.std = np.load(pjoin(opt.babel_data_root, f'Std_both_seg_pca_{opt.pca}.npy'))
            else:
                print("Loading merged mean/std from:", pjoin(opt.babel_data_root))
                self.mean = np.load(pjoin(opt.babel_data_root, 'Mean_both_seg.npy'))
                self.std = np.load(pjoin(opt.babel_data_root, 'Std_both_seg.npy'))

        if mode == 'eval':
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        babel_split = kwargs['args'].train_babel_split
        self.babel_split_file = pjoin(opt.babel_data_root, f'{babel_split}.txt')

        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        elif mode == 'gt':
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalOnlyText2MotionDataset(
                self.opt, self.mean, self.std, self.split_file, self.w_vectorizer
            )
            self.num_actions = 1
        elif mode == 'eval':
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalAndLocalText2MotionDataset(
                self.opt, self.mean, self.std, self.split_file, self.w_vectorizer
            )
            self.num_actions = 1
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = LocalText2MotionDataset_OptionalGlobal(
                self.opt, self.mean, self.std, self.split_file, self.babel_split_file, self.w_vectorizer
            )
            self.num_actions = 1

        assert len(self.t2m_dataset) > 1, (
            "You loaded an empty dataset. Ensure you have the full data as described in the README."
        )

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

class HumanML3D_Full(data.Dataset):
    """
    Loads the full HumanML3D dataset.
    """
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train_ft", **kwargs):
        self.mode = mode
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.label_dir = pjoin(abs_base_path, opt.label_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'

        opt.pca = kwargs['pca']
        opt.pca_dir = pjoin(abs_base_path, opt.pca_dir)
        opt.args = kwargs['args']
        self.opt = opt

        if mode == 'gt':
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            if opt.pca > 0:
                self.mean = np.load(pjoin(opt.data_root, f'Mean_seg_pca_{opt.pca}.npy'))
                self.std = np.load(pjoin(opt.data_root, f'Std_seg_pca_{opt.pca}.npy'))
            else:
                self.mean = np.load(pjoin(opt.data_root, 'Mean_seg.npy'))
                self.std = np.load(pjoin(opt.data_root, 'Std_seg.npy'))

        if mode == 'eval':
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        elif mode == 'gt':
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = GlobalTextDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1
        elif mode == 'eval':
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDataset_Seg(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDataset_Seg_HML(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1

        assert len(self.t2m_dataset) > 1, (
            "You loaded an empty dataset. Ensure you have the full data as described in the README."
        )

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
