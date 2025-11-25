import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import random
from collections import defaultdict

class SyntheticMathDataLoader:
    """Generate synthetic mathematical tasks for the awakening phase"""
    
    def __init__(self, task_type: str = "addition", max_digits: int = 2):
        self.task_type = task_type
        self.max_digits = max_digits
        self.vocab_size = 28  # A-Z + 0-1
        self.char_to_idx = self._create_char_mapping()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
    def _create_char_mapping(self) -> Dict[str, int]:
        """Create character to index mapping"""
        mapping = {}
        # Map letters A-Z to 0-25
        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            mapping[char] = i
        # Map digits 0-1 to 26-27
        mapping['0'] = 26
        mapping['1'] = 27
        return mapping
    
    def encode_number(self, num: int, base: int = 2) -> str:
        """Encode number in specified base using available characters"""
        if num == 0:
            return '0'
        
        digits = []
        while num > 0:
            remainder = num % base
            if remainder < 26:
                digits.append(chr(ord('A') + remainder))
            else:
                digits.append(str(remainder - 26))
            num //= base
        
        return ''.join(reversed(digits))
    
    def decode_number(self, encoded: str, base: int = 2) -> int:
        """Decode number from character representation"""
        result = 0
        for char in encoded:
            if char.isalpha():
                digit = ord(char.upper()) - ord('A')
            else:
                digit = int(char) + 26
            result = result * base + digit
        return result
    
    def generate_addition_problem(self, max_value: int = 100) -> Tuple[str, str]:
        """Generate simple addition problem"""
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)
        result = a + b
        
        # Format: "A+B=C" where A, B, C are encoded numbers
        problem = f"{self.encode_number(a)}+{self.encode_number(b)}="
        answer = self.encode_number(result)
        
        return problem, answer
    
    def generate_multiplication_problem(self, max_value: int = 20) -> Tuple[str, str]:
        """Generate multiplication problem"""
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)
        result = a * b
        
        problem = f"{self.encode_number(a)}*{self.encode_number(b)}="
        answer = self.encode_number(result)
        
        return problem, answer
    
    def generate_sequence_completion(self, length: int = 10) -> Tuple[str, str]:
        """Generate sequence completion task"""
        # Simple arithmetic progression
        start = random.randint(0, 10)
        step = random.randint(1, 5)
        
        sequence = [start + i * step for i in range(length)]
        encoded_seq = [self.encode_number(x) for x in sequence]
        
        # Hide last element
        problem = ''.join(encoded_seq[:-1]) + '?'
        answer = encoded_seq[-1]
        
        return problem, answer
    
    def generate_pattern_recognition(self, pattern_length: int = 5) -> Tuple[str, str]:
        """Generate pattern recognition task"""
        # Create repeating pattern
        base_pattern = [random.randint(0, 10) for _ in range(pattern_length)]
        full_pattern = base_pattern * 3  # Repeat 3 times
        
        encoded_pattern = [self.encode_number(x) for x in full_pattern]
        
        # Show partial pattern
        show_length = len(encoded_pattern) - pattern_length
        problem = ''.join(encoded_pattern[:show_length]) + '?'
        answer = ''.join(encoded_pattern[show_length:])
        
        return problem, answer
    
    def generate_batch(self, batch_size: int = 32, task_type: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of training examples"""
        if task_type is None:
            task_type = self.task_type
            
        problems = []
        answers = []
        
        for _ in range(batch_size):
            if task_type == "addition":
                prob, ans = self.generate_addition_problem()
            elif task_type == "multiplication":
                prob, ans = self.generate_multiplication_problem()
            elif task_type == "sequence":
                prob, ans = self.generate_sequence_completion()
            elif task_type == "pattern":
                prob, ans = self.generate_pattern_recognition()
            else:
                # Random task
                task = random.choice(["addition", "multiplication", "sequence", "pattern"])
                prob, ans = self.generate_batch(1, task)[0]
                prob = prob[0]  # Unpack from batch
                ans = ans[0]
            
            problems.append(prob)
            answers.append(ans)
        
        # Convert to tensors
        input_tensors = []
        target_tensors = []
        
        for prob, ans in zip(problems, answers):
            # Combine problem and answer for training
            full_sequence = prob + ans
            
            # Convert to indices
            input_indices = [self.char_to_idx.get(char, 0) for char in full_sequence[:-1]]
            target_indices = [self.char_to_idx.get(char, 0) for char in full_sequence[1:]]
            
            input_tensors.append(torch.tensor(input_indices, dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices, dtype=torch.long))
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in input_tensors)
        
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(input_tensors, target_tensors):
            padded_input = F.pad(inp, (0, max_len - len(inp)), value=0)
            padded_target = F.pad(tgt, (0, max_len - len(tgt)), value=0)
            
            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)
        
        return torch.stack(padded_inputs), torch.stack(padded_targets)
    
    def decode_predictions(self, predictions: torch.Tensor) -> List[str]:
        """Decode model predictions back to strings"""
        predicted_indices = torch.argmax(predictions, dim=-1)
        decoded_strings = []
        
        for seq in predicted_indices:
            chars = []
            for idx in seq:
                char = self.idx_to_char.get(idx.item(), 'A')
                chars.append(char)
            decoded_strings.append(''.join(chars))
        
        return decoded_strings
    
    def evaluate_accuracy(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Evaluate prediction accuracy"""
        total_chars = 0
        correct_chars = 0
        total_sequences = 0
        correct_sequences = 0
        
        for pred, target in zip(predictions, targets):
            total_sequences += 1
            if pred == target:
                correct_sequences += 1
            
            # Character-level accuracy
            min_len = min(len(pred), len(target))
            total_chars += min_len
            correct_chars += sum(1 for i in range(min_len) if pred[i] == target[i])
        
        return {
            'character_accuracy': correct_chars / total_chars if total_chars > 0 else 0.0,
            'sequence_accuracy': correct_sequences / total_sequences if total_sequences > 0 else 0.0
        }

class ConceptualDataAnalyzer:
    """Analyze conceptual patterns in generated data"""
    
    def __init__(self):
        self.concept_patterns = defaultdict(list)
        self.pattern_frequencies = defaultdict(int)
        
    def analyze_sequence(self, sequence: str):
        """Analyze conceptual patterns in a sequence"""
        # Look for repeating patterns of different lengths
        for pattern_length in range(2, min(len(sequence) // 2, 10)):
            for i in range(len(sequence) - pattern_length + 1):
                pattern = sequence[i:i+pattern_length]
                self.pattern_frequencies[pattern] += 1
                
        # Identify most common patterns
        common_patterns = sorted(self.pattern_frequencies.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_patterns': len(self.pattern_frequencies),
            'common_patterns': common_patterns,
            'pattern_diversity': len(set(self.pattern_frequencies.values()))
        }
    
    def analyze_conceptual_groups(self, vq_codes: np.ndarray) -> Dict:
        """Analyze conceptual groupings in VQ codes"""
        if len(vq_codes) == 0:
            return {}
            
        # Calculate pairwise distances between VQ codes
        distances = []
        for i in range(len(vq_codes)):
            for j in range(i+1, len(vq_codes)):
                dist = np.linalg.norm(vq_codes[i] - vq_codes[j])
                distances.append(dist)
        
        # Calculate clustering metrics
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Identify code clusters (simplified)
        clusters = []
        threshold = mean_distance - 0.5 * std_distance
        
        for i in range(len(vq_codes)):
            cluster = [i]
            for j in range(len(vq_codes)):
                if i != j:
                    dist = np.linalg.norm(vq_codes[i] - vq_codes[j])
                    if dist < threshold:
                        cluster.append(j)
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return {
            'num_clusters': len(clusters),
            'mean_inter_distance': mean_distance,
            'std_inter_distance': std_distance,
            'cluster_sizes': [len(cluster) for cluster in clusters]
        }
    
    def generate_concept_report(self, sequences: List[str], vq_codes: np.ndarray) -> Dict:
        """Generate comprehensive concept analysis report"""
        sequence_analysis = self.analyze_sequence(''.join(sequences))
        grouping_analysis = self.analyze_conceptual_groups(vq_codes)
        
        return {
            'sequence_patterns': sequence_analysis,
            'conceptual_groupings': grouping_analysis,
            'emergent_complexity': sequence_analysis['pattern_diversity'] * len(grouping_analysis.get('cluster_sizes', [1])),
            'conceptual_stability': 1.0 / (1.0 + grouping_analysis.get('std_inter_distance', 1.0))
        }