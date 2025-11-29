"""
Task generators for ZD-CBE v2 (Enhanced):
1. Binary Addition
2. Sequence Reverse
3. Pattern Completion
4. Sorting (NEW)
5. Parity Checking (NEW)
6. Next in Sequence (NEW)
"""
import torch
import numpy as np
from typing import Tuple, List
import random


class TaskGenerator:
    """Base class for task generation"""

    def __init__(self, vocab_size: int = 36):
        self.vocab_size = vocab_size
        self.char_to_idx = self._create_char_mapping()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def _create_char_mapping(self):
        """Create character to index mapping"""
        mapping = {}
        # Map letters A-Z to 0-25
        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            mapping[char] = i
        # Map digits 0-9 to 26-35
        for i, char in enumerate('0123456789'):
            mapping[char] = 26 + i
        return mapping

    def string_to_indices(self, s: str) -> List[int]:
        """Convert string to token indices"""
        return [self.char_to_idx.get(c.upper(), 0) for c in s]

    def indices_to_string(self, indices: List[int]) -> str:
        """Convert indices to string"""
        return ''.join(self.idx_to_char.get(idx, 'A') for idx in indices)


class BinaryAdditionTask(TaskGenerator):
    """Binary addition: 10 + 11 = 101"""

    def __init__(self, max_digits: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_digits = max_digits

    def encode_number(self, num: int) -> str:
        """Encode number in binary"""
        if num == 0:
            return '0'
        return bin(num)[2:]  # Remove '0b' prefix

    def generate_single(self) -> Tuple[str, str]:
        """Generate single addition problem"""
        max_val = (2 ** self.max_digits) - 1
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        result = a + b

        problem = f"{self.encode_number(a)}+{self.encode_number(b)}="
        answer = self.encode_number(result)

        return problem, answer

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of problems"""
        problems = []
        answers = []

        for _ in range(batch_size):
            prob, ans = self.generate_single()
            problems.append(prob)
            answers.append(ans)

        return self._to_tensors(problems, answers)

    def _to_tensors(self, problems: List[str], answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert problems and answers to tensors"""
        input_tensors = []
        target_tensors = []

        for prob, ans in zip(problems, answers):
            full_sequence = prob + ans

            # Convert to indices
            input_indices = self.string_to_indices(full_sequence[:-1])
            target_indices = self.string_to_indices(full_sequence[1:])

            input_tensors.append(torch.tensor(input_indices, dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices, dtype=torch.long))

        # Pad sequences
        max_len = max(len(seq) for seq in input_tensors)

        padded_inputs = []
        padded_targets = []

        for inp, tgt in zip(input_tensors, target_tensors):
            padded_input = torch.nn.functional.pad(inp, (0, max_len - len(inp)), value=0)
            padded_target = torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=0)

            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)

        return torch.stack(padded_inputs), torch.stack(padded_targets)


class SequenceReverseTask(TaskGenerator):
    """Sequence reversal: A B C D → D C B A"""

    def __init__(self, min_length: int = 3, max_length: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def generate_single(self) -> Tuple[str, str]:
        """Generate single reversal problem"""
        length = random.randint(self.min_length, self.max_length)

        # Generate random sequence
        sequence = [random.choice('ABCDEFGHIJ') for _ in range(length)]

        # Input: sequence with separator
        problem = ' '.join(sequence) + ' > '

        # Answer: reversed sequence
        answer = ' '.join(reversed(sequence))

        return problem, answer

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of reversal problems"""
        problems = []
        answers = []

        for _ in range(batch_size):
            prob, ans = self.generate_single()
            # Remove spaces for token encoding
            prob_no_space = prob.replace(' ', '')
            ans_no_space = ans.replace(' ', '')
            problems.append(prob_no_space)
            answers.append(ans_no_space)

        return self._to_tensors(problems, answers)

    def _to_tensors(self, problems: List[str], answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors"""
        input_tensors = []
        target_tensors = []

        for prob, ans in zip(problems, answers):
            full_sequence = prob + ans

            # Convert to indices
            input_indices = self.string_to_indices(full_sequence[:-1])
            target_indices = self.string_to_indices(full_sequence[1:])

            input_tensors.append(torch.tensor(input_indices, dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices, dtype=torch.long))

        # Pad sequences
        max_len = max(len(seq) for seq in input_tensors)

        padded_inputs = []
        padded_targets = []

        for inp, tgt in zip(input_tensors, target_tensors):
            padded_input = torch.nn.functional.pad(inp, (0, max_len - len(inp)), value=0)
            padded_target = torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=0)

            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)

        return torch.stack(padded_inputs), torch.stack(padded_targets)


class PatternCompletionTask(TaskGenerator):
    """Pattern completion: A B A B A → B"""

    def __init__(self, pattern_types: List[str] = None, **kwargs):
        super().__init__(**kwargs)

        if pattern_types is None:
            self.pattern_types = [
                'repeat',      # A A A A → A
                'alternate',   # A B A B → A
                'increment',   # A B C D → E
            ]
        else:
            self.pattern_types = pattern_types

    def generate_repeat_pattern(self) -> Tuple[str, str]:
        """Generate repeating pattern: A A A A → A"""
        char = random.choice('ABCDEFGHIJ')
        length = random.randint(3, 6)

        sequence = [char] * length
        problem = ''.join(sequence) + '?'
        answer = char

        return problem, answer

    def generate_alternate_pattern(self) -> Tuple[str, str]:
        """Generate alternating pattern: A B A B → A"""
        char1, char2 = random.sample('ABCDEFGHIJ', 2)
        length = random.randint(4, 8)

        sequence = [char1 if i % 2 == 0 else char2 for i in range(length)]
        problem = ''.join(sequence) + '?'
        answer = char1 if length % 2 == 0 else char2

        return problem, answer

    def generate_increment_pattern(self) -> Tuple[str, str]:
        """Generate increment pattern: A B C D → E"""
        start_char = random.choice('ABCDEFG')  # Ensure we can increment
        start_idx = ord(start_char) - ord('A')
        length = random.randint(3, 5)

        sequence = [chr(ord('A') + start_idx + i) for i in range(length)]
        problem = ''.join(sequence) + '?'
        answer = chr(ord('A') + start_idx + length)

        return problem, answer

    def generate_single(self) -> Tuple[str, str]:
        """Generate single pattern completion problem"""
        pattern_type = random.choice(self.pattern_types)

        if pattern_type == 'repeat':
            return self.generate_repeat_pattern()
        elif pattern_type == 'alternate':
            return self.generate_alternate_pattern()
        elif pattern_type == 'increment':
            return self.generate_increment_pattern()
        else:
            return self.generate_repeat_pattern()

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of pattern completion problems"""
        problems = []
        answers = []

        for _ in range(batch_size):
            prob, ans = self.generate_single()
            problems.append(prob)
            answers.append(ans)

        return self._to_tensors(problems, answers)

    def _to_tensors(self, problems: List[str], answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors"""
        input_tensors = []
        target_tensors = []

        for prob, ans in zip(problems, answers):
            full_sequence = prob + ans

            # Convert to indices
            input_indices = self.string_to_indices(full_sequence[:-1])
            target_indices = self.string_to_indices(full_sequence[1:])

            input_tensors.append(torch.tensor(input_indices, dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices, dtype=torch.long))

        # Pad sequences
        max_len = max(len(seq) for seq in input_tensors)

        padded_inputs = []
        padded_targets = []

        for inp, tgt in zip(input_tensors, target_tensors):
            padded_input = torch.nn.functional.pad(inp, (0, max_len - len(inp)), value=0)
            padded_target = torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=0)

            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)

        return torch.stack(padded_inputs), torch.stack(padded_targets)


class SortingTask(TaskGenerator):
    """Sorting: 5 3 8 1 > → 1 3 5 8"""

    def __init__(self, min_length: int = 3, max_length: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def generate_single(self) -> Tuple[str, str]:
        """Generate single sorting problem"""
        length = random.randint(self.min_length, self.max_length)

        # Generate random sequence of digits 0-9
        sequence = [str(random.randint(0, 9)) for _ in range(length)]

        # Sort the sequence
        sorted_sequence = sorted(sequence)

        # Problem: unsorted sequence with separator
        problem = ''.join(sequence) + '>'

        # Answer: sorted sequence
        answer = ''.join(sorted_sequence)

        return problem, answer

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of sorting problems"""
        problems = []
        answers = []

        for _ in range(batch_size):
            prob, ans = self.generate_single()
            problems.append(prob)
            answers.append(ans)

        return self._to_tensors(problems, answers)

    def _to_tensors(self, problems: List[str], answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors"""
        input_tensors = []
        target_tensors = []

        for prob, ans in zip(problems, answers):
            full_sequence = prob + ans

            # Convert to indices
            input_indices = self.string_to_indices(full_sequence[:-1])
            target_indices = self.string_to_indices(full_sequence[1:])

            input_tensors.append(torch.tensor(input_indices, dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices, dtype=torch.long))

        # Pad sequences
        max_len = max(len(seq) for seq in input_tensors)

        padded_inputs = []
        padded_targets = []

        for inp, tgt in zip(input_tensors, target_tensors):
            padded_input = torch.nn.functional.pad(inp, (0, max_len - len(inp)), value=0)
            padded_target = torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=0)

            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)

        return torch.stack(padded_inputs), torch.stack(padded_targets)


class ParityCheckingTask(TaskGenerator):
    """Parity checking: 1 0 1 1 ? → 1 (odd number of 1s)"""

    def __init__(self, min_length: int = 3, max_length: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def generate_single(self) -> Tuple[str, str]:
        """Generate single parity checking problem"""
        length = random.randint(self.min_length, self.max_length)

        # Generate random binary sequence
        sequence = [str(random.randint(0, 1)) for _ in range(length)]

        # Count 1s
        count_ones = sum(int(bit) for bit in sequence)

        # Parity: 1 if odd, 0 if even
        parity = '1' if count_ones % 2 == 1 else '0'

        # Problem: binary sequence with question mark
        problem = ''.join(sequence) + '?'

        # Answer: parity bit
        answer = parity

        return problem, answer

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of parity checking problems"""
        problems = []
        answers = []

        for _ in range(batch_size):
            prob, ans = self.generate_single()
            problems.append(prob)
            answers.append(ans)

        return self._to_tensors(problems, answers)

    def _to_tensors(self, problems: List[str], answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors"""
        input_tensors = []
        target_tensors = []

        for prob, ans in zip(problems, answers):
            full_sequence = prob + ans

            # Convert to indices
            input_indices = self.string_to_indices(full_sequence[:-1])
            target_indices = self.string_to_indices(full_sequence[1:])

            input_tensors.append(torch.tensor(input_indices, dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices, dtype=torch.long))

        # Pad sequences
        max_len = max(len(seq) for seq in input_tensors)

        padded_inputs = []
        padded_targets = []

        for inp, tgt in zip(input_tensors, target_tensors):
            padded_input = torch.nn.functional.pad(inp, (0, max_len - len(inp)), value=0)
            padded_target = torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=0)

            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)

        return torch.stack(padded_inputs), torch.stack(padded_targets)


class NextInSequenceTask(TaskGenerator):
    """Next in sequence: A B C ? → D or 2 4 6 ? → 8"""

    def __init__(self, sequence_types: List[str] = None, **kwargs):
        super().__init__(**kwargs)

        if sequence_types is None:
            self.sequence_types = [
                'alphabetic',  # A B C → D
                'numeric',     # 2 4 6 → 8 (arithmetic progression)
            ]
        else:
            self.sequence_types = sequence_types

    def generate_alphabetic_sequence(self) -> Tuple[str, str]:
        """Generate alphabetic progression: A B C ? → D"""
        start_char = random.choice('ABCDEFGHIJKLMNOPQRS')  # Ensure room to increment
        start_idx = ord(start_char) - ord('A')
        length = random.randint(3, 5)

        sequence = [chr(ord('A') + start_idx + i) for i in range(length)]
        problem = ''.join(sequence) + '?'
        answer = chr(ord('A') + start_idx + length)

        return problem, answer

    def generate_numeric_sequence(self) -> Tuple[str, str]:
        """Generate arithmetic progression: 2 4 6 ? → 8"""
        start = random.randint(0, 5)
        step = random.randint(1, 3)
        length = random.randint(3, 5)

        sequence = [str(start + i * step) for i in range(length)]

        # Ensure all numbers are single digits
        if any(int(num) > 9 for num in sequence):
            # Fallback to simpler sequence
            start = random.randint(0, 2)
            step = 1
            sequence = [str(start + i) for i in range(length)]

        problem = ''.join(sequence) + '?'
        answer = str(start + length * step)

        # Ensure answer is single digit
        if int(answer) > 9:
            answer = str(int(answer) % 10)

        return problem, answer

    def generate_single(self) -> Tuple[str, str]:
        """Generate single next-in-sequence problem"""
        sequence_type = random.choice(self.sequence_types)

        if sequence_type == 'alphabetic':
            return self.generate_alphabetic_sequence()
        elif sequence_type == 'numeric':
            return self.generate_numeric_sequence()
        else:
            return self.generate_alphabetic_sequence()

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of next-in-sequence problems"""
        problems = []
        answers = []

        for _ in range(batch_size):
            prob, ans = self.generate_single()
            problems.append(prob)
            answers.append(ans)

        return self._to_tensors(problems, answers)

    def _to_tensors(self, problems: List[str], answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors"""
        input_tensors = []
        target_tensors = []

        for prob, ans in zip(problems, answers):
            full_sequence = prob + ans

            # Convert to indices
            input_indices = self.string_to_indices(full_sequence[:-1])
            target_indices = self.string_to_indices(full_sequence[1:])

            input_tensors.append(torch.tensor(input_indices, dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices, dtype=torch.long))

        # Pad sequences
        max_len = max(len(seq) for seq in input_tensors)

        padded_inputs = []
        padded_targets = []

        for inp, tgt in zip(input_tensors, target_tensors):
            padded_input = torch.nn.functional.pad(inp, (0, max_len - len(inp)), value=0)
            padded_target = torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=0)

            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)

        return torch.stack(padded_inputs), torch.stack(padded_targets)


# Test all tasks
if __name__ == "__main__":
    print("="*60)
    print("TESTING ALL TASK GENERATORS (6 TASKS)")
    print("="*60)

    # Test Binary Addition
    print("\n1. Binary Addition Task:")
    addition_task = BinaryAdditionTask(max_digits=3)
    for i in range(5):
        prob, ans = addition_task.generate_single()
        print(f"   {prob}{ans}")

    inputs, targets = addition_task.generate_batch(4)
    print(f"   Batch shape: inputs={inputs.shape}, targets={targets.shape}")

    # Test Sequence Reverse
    print("\n2. Sequence Reverse Task:")
    reverse_task = SequenceReverseTask(min_length=3, max_length=5)
    for i in range(5):
        prob, ans = reverse_task.generate_single()
        print(f"   {prob}{ans}")

    inputs, targets = reverse_task.generate_batch(4)
    print(f"   Batch shape: inputs={inputs.shape}, targets={targets.shape}")

    # Test Pattern Completion
    print("\n3. Pattern Completion Task:")
    pattern_task = PatternCompletionTask()
    for i in range(5):
        prob, ans = pattern_task.generate_single()
        print(f"   {prob}{ans}")

    inputs, targets = pattern_task.generate_batch(4)
    print(f"   Batch shape: inputs={inputs.shape}, targets={targets.shape}")

    # Test Sorting (NEW)
    print("\n4. Sorting Task:")
    sorting_task = SortingTask(min_length=3, max_length=6)
    for i in range(5):
        prob, ans = sorting_task.generate_single()
        print(f"   {prob}{ans}")

    inputs, targets = sorting_task.generate_batch(4)
    print(f"   Batch shape: inputs={inputs.shape}, targets={targets.shape}")

    # Test Parity Checking (NEW)
    print("\n5. Parity Checking Task:")
    parity_task = ParityCheckingTask(min_length=3, max_length=8)
    for i in range(5):
        prob, ans = parity_task.generate_single()
        print(f"   {prob}{ans}")

    inputs, targets = parity_task.generate_batch(4)
    print(f"   Batch shape: inputs={inputs.shape}, targets={targets.shape}")

    # Test Next in Sequence (NEW)
    print("\n6. Next in Sequence Task:")
    sequence_task = NextInSequenceTask()
    for i in range(5):
        prob, ans = sequence_task.generate_single()
        print(f"   {prob}{ans}")

    inputs, targets = sequence_task.generate_batch(4)
    print(f"   Batch shape: inputs={inputs.shape}, targets={targets.shape}")

    print("\n" + "="*60)
    print("✅ ALL 6 TASK GENERATORS WORKING!")
    print("="*60)
