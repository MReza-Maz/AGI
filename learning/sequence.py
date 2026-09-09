"""Sequence dataset helpers for next-token prediction."""


class SequenceDataset:
    """Create overlapping input/target windows from an encoded token stream."""
    def __init__(self, token_ids, sequence_length):
        if sequence_length < 1:
            raise ValueError("sequence length must be positive")
        self.token_ids = [int(token_id) for token_id in token_ids]
        self.sequence_length = int(sequence_length)
        if len(self.token_ids) < self.sequence_length + 1:
            raise ValueError("token stream is too short for the requested sequence length")

    def __len__(self):
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("sequence index out of range")
        start = index
        end = start + self.sequence_length
        return self.token_ids[start:end], self.token_ids[start + 1:end + 1]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
