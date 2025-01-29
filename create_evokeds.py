import os
import mne
import pandas as pd
from tqdm import tqdm


def create_evokeds(folder): 
    def create_evokeds_aux(epochs_array, n):
        evokeds = []
        for i in range(1, len(epochs_array)):  
            # Slice the epochs array to get the first i epochs
            subset_epochs = epochs_array[:i+1]
            # Average the sliced epochs to create an evoked signal
            evoked = subset_epochs.average()
            evoked = mne.EpochsArray(evoked.get_data().reshape(1, 8, 100), evoked.info, verbose='CRITICAL')
            # Append the evoked signal to the list
            evokeds.append(evoked)
        
        # Return evokeds and indices adjusted with n
        evoked_indices = dict(evoked_indices=[i + n for i in range(len(evokeds))])
        return evokeds, evoked_indices

    # Read data
    train_epochs = mne.read_epochs(os.path.join(folder, 'train_epochs_epo.fif'))
    train_evoked_annotations = pd.read_csv(os.path.join(folder, 'train_evoked_annotations.csv'))
    indices = [x for x in train_evoked_annotations['epoch_indexes']]

    # Initialize variables
    evoked_list = []
    evoked_indices = []
    n = 0

    # Initialize progress bar
    with tqdm(total=len(indices), desc="Processing evokeds") as pbar:
        for i, index in enumerate(indices):
            # print(f"Processing epoch index: {i+1}/{len(indices)}")
            selected_epochs = train_epochs[eval(index)]
            evokeds, evoked_idx = create_evokeds_aux(selected_epochs, n)
            evoked_indices.append(evoked_idx)
            evoked_list.append(evokeds)
            n += len(evokeds)
            
            # Update the progress bar
            pbar.update(1)

    # Create DataFrame for evoked indices and merge with annotations
    evoked_indices = pd.DataFrame(evoked_indices)
    train_evoked_annotations = pd.concat([train_evoked_annotations, evoked_indices], axis=1)

    # Save updated annotations
    train_evoked_annotations.to_csv(os.path.join(folder, 'train_evoked_annotations.csv'), index=False)

    # Concatenate all evoked signals into a single MNE object
    evoked_signals = mne.concatenate_epochs([item for sublist in evoked_list for item in sublist], verbose='CRITICAL')

    # Save the evoked signals
    evoked_signals.save(os.path.join(folder, 'train_evokeds_epo.fif'), overwrite=True)
    
if __name__ == '__main__':
    Gib_UVA_folder1 = os.path.join('Data2', 'Gib_UVA', 'Dataset1')
    Gib_UVA_folder2 = os.path.join('Data2', 'Gib_UVA', 'Dataset2')
    Gib_UVA_folder3 = os.path.join('Data2', 'Gib_UVA', 'Dataset3')
    
    # create_evokeds(Gib_UVA_folder1)
    # create_evokeds(Gib_UVA_folder2)
    create_evokeds(Gib_UVA_folder3)
    