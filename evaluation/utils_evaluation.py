import torch
import numpy as np
import pandas as pd
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
bci_data_folder = os.path.join(parent_dir, 'Data', 'BCI Comp')

########################### CHARACTER RECOGNITION METHODS ###########################


CHARA_MAP = {
        'A':(7,1), 'B':(7,2), 'C':(7,3), 'D':(7,4), 'E':(7,5), 'F':(7,6),
        'G':(8,1), 'H':(8,2), 'I':(8,3), 'J':(8,4), 'K':(8,5), 'L':(8,6),
        'M':(9,1), 'N':(9,2), 'O':(9,3), 'P':(9,4), 'Q':(9,5), 'R':(9,6),
        'S':(10,1), 'T':(10,2), 'U':(10,3), 'V':(10,4), 'W':(10,5), 'X':(10,6),
        'Y':(11,1), 'Z':(11,2), '1':(11,3), '2':(11,4), '3':(11,5), '4':(11,6),
        '5':(12,1), '6':(12,2), '7':(12,3), '8':(12,4), '9':(12,5), '_':(12,6) 
    }

def dict_to_char(dictionary:dict) -> str:
    """given a dictionary with codes probabilities it computes the character according to the character mapping

    Args:
        dictionary (dict): Contains the probabilities/scores that the classifier attribited to each row/column
                            Example: chara_pred = {1:[], 2:[], ... , 11:[], 12:[]}

    Returns:
        str: predicted character
    """
    values = []
    row_numbers = [x[1] for _,x in CHARA_MAP.items()]
    column_numbers = [x[0] for _,x in CHARA_MAP.items()]
    
    # Extract the column with the maximum value
    columns = {k:v for k,v in dictionary.items() if k in column_numbers}
    max_column = max(columns, key=lambda x: columns[x])
    values.append(max_column)
    
    # Extract the row with the maximum value 
    rows = {k: v for k, v in dictionary.items() if k in row_numbers}
    max_row = max(rows, key=lambda x: rows[x])
    values.append(max_row)

    return str([char for char, pos in CHARA_MAP.items()
                if pos[0] == values[0] and pos[1] == values[1]][0])
    
def calculate_string_accuracy(string1:str, string2:str) -> float:
    max_length = max(len(string1), len(string2))
        
    match_count = 0
        
    for char1, char2 in zip(string1, string2):
        if char1 == char2:
            match_count += 1
        
    accuracy = round((match_count / max_length) * 100, 1)
        
    return accuracy


def predict_string(logits, codes:list, n_trials:int, n_chars:int) -> str:
    """Builds a string from the test_data
    """
    # Reshape test_data -> group all signals of a character spelling -> (n_chars, n_trials_per_char, 1, n_channels, n_time_points)
    n_row_cols = 12
    probs = torch.sigmoid(logits)

    # Reshape codes -> all codes belonging to a character spelling -> (n_chars, codes)
    codes = np.array_split(codes, n_chars, axis=0) 
    probs = np.array_split(probs, n_chars, axis=0)
    preds = []
    for char_probs, char_codes in zip(probs, codes):
        
        char_pred = {i:[] for i in range(1, n_row_cols+1)}
        
        for idx, (prob, row_col) in enumerate(zip(char_probs, char_codes)):
            if idx < n_row_cols * n_trials:
                char_pred[row_col.item()].append(prob)
        for row_col, probs in char_pred.items():
            char_pred[row_col] = torch.mean(torch.Tensor(probs))
            
        preds.append(dict_to_char(char_pred))
    return ''.join(preds)
        

def get_target_string_BCIComp(subject):
    return open(os.path.join(bci_data_folder, f"true_labels_{subject}.txt"), 'r').read()

def compute_accuracy(list1, list2):
    # Ensure both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Count the number of correct matches
    correct_matches = sum(1 for t1, t2 in zip(list1, list2) if t1 == t2)
    
    # Compute accuracy
    accuracy = round(correct_matches / len(list1) * 100, 2)
    return accuracy


def character_recognition_Gib_UVA(logits, test_annotations, speller_matrices):
    # Extract the two tensors
    # first_tensor, second_tensor = speller_matrices[0]

    # Move tensors to CPU if needed and convert to lists
    # first_values = first_tensor.cpu().tolist()
    # second_values = second_tensor.cpu().tolist()

    # Create a list of tuples
    # speller_matrices = list(zip(first_values, second_values))
    speller_matrices = speller_matrices.cpu().numpy().tolist()
    test_annotations = test_annotations.cpu().numpy()
    test_annotations = pd.DataFrame(
        test_annotations, columns=['dataset', 'subject', 'char', 'sequences', 'codes', 'labels']
    )
    test_annotations['speller_matrices'] = speller_matrices

    probs = torch.sigmoid(logits)
    chars = np.unique(test_annotations['char'])

    acc_results = []

    # Find the maximum sequences available for each character
    max_seqs_per_char = {
        char: test_annotations[test_annotations['char'] == char].shape[0] // len(set(test_annotations[test_annotations['char'] == char]['codes']))
        for char in chars
    }
    max_seqs = max(max_seqs_per_char.values())

    for t in range(1, max_seqs + 1):
        print(f"Evaluating with {t} sequences per character...")

        preds = []
        y = []

        for char in chars:
            # Limit t to the maximum number of sequences available for the current character
            # max_t = min(t, max_seqs_per_char[char])
            if max_seqs_per_char[char] >= t:

                char_annotations = test_annotations[test_annotations['char'] == char]
                char_indices = test_annotations[test_annotations['char'] == char].index.to_list()

                char_indices_tensor = torch.tensor(char_indices, dtype=torch.long)
                char_probs = probs[char_indices_tensor]

                char_codes = char_annotations['codes'].values
                target_coordenates = np.unique(char_annotations[char_annotations['labels'] == 1]['codes'])
                speller_matrix = np.unique(char_annotations['speller_matrices'].values)[0]
                
                y.append((int(target_coordenates[0]), int(target_coordenates[1])))
                
                char_pred = {i: [] for i in set(char_codes)}
                for idx, (prob, row_col) in enumerate(zip(char_probs, char_codes)):
                    if idx < len(set(char_codes)) * t:
                        char_pred[row_col.item()].append(prob)
                        # y.append((int(target_coordenates[0]), int(target_coordenates[1]))) if idx == 0 else None
                for row_col, probabilities in char_pred.items():
                    char_pred[row_col] = torch.mean(torch.Tensor(probabilities))

                row_keys = {k: v for k, v in char_pred.items() if k < speller_matrix[0]}
                column_keys = {k: v for k, v in char_pred.items() if k >= speller_matrix[0]}

                # Find the key with the highest value for rows (0-5)
                max_row_key = max(row_keys, key=row_keys.get)  # This finds the key with the highest value

                # Find the key with the highest value for columns (6-14)
                max_column_key = max(column_keys, key=column_keys.get)  # This finds the key with the highest value

                preds.append((int(max_row_key), int(max_column_key)))

        print(f"Preds -> {preds}")
        print(f"Trues -> {y}")
        acc = compute_accuracy(preds, y)
        print(f"{t} trials -> {acc}")
        acc_results.append(acc)

    print(f"Accuracy Over Trials -> {acc_results}")
    return acc_results