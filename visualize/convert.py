import numpy as np

def read_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

def convert_labels_to_ids(labels):
    unique_labels = list(set(labels))
    label_to_id = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    ids = [label_to_id[label] for label in labels]
    print(list(set(labels)))
    return np.array(ids), label_to_id

def process_labels(input_file):
    labels = read_labels_from_file(input_file)
    ids, label_to_id = convert_labels_to_ids(labels)
    output_file = input_file[:-15] + '_color.npy'
    save_ids_to_npy(ids, output_file)
    print("Label to ID mapping:", label_to_id)

def save_ids_to_npy(ids, output_path):
    np.save(output_path, ids)

def main(input_file, output_file):
    # Step 1: Read the text file containing action labels
    labels = read_labels_from_file(input_file)
    
    # Step 2: Convert action labels to numerical IDs
    ids, label_to_id = convert_labels_to_ids(labels)
    
    # Step 3: Save the numerical IDs as a NumPy file
    save_ids_to_npy(ids, output_file)
    
    # Optionally, print the mapping for reference
    print("Label to ID mapping:", label_to_id)

# Example usage
# input_file = '/home/chuqiao/Code/Poject_2023/motion-diffusion-model/save_results/new_seg_mt_efficient_both_loss_pca_51_humanml_trans_enc_512/samples_new_seg_mt_efficient_both_loss_pca_51_humanml_trans_enc_512_000450000_t2m_seed10_gscale0.0_a_person_waves_hand_above_head_local1/row00_col00_m/row00_col00_m_frame_text.txt'
# output_file = input_file[:-15] + '_color.npy'

# main(input_file, output_file)
