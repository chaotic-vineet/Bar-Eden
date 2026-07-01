from collections import Counter
from collections import defaultdict
import regex as re
import json

SPLIT_PATTERN = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

def pair_freqs(seq:list, counts:dict, weight: int, pairs_in_seq: dict) -> None:
    for pair in zip(seq, seq[1:]):
        counts[pair] = counts.get(pair, 0) + weight
        pairs_in_seq[pair].add(seq)

def merge_pair(seq: tuple, pair: tuple, replacement: int) -> list:
    merged = []
    
    i=0
    while i < len(seq):
        if i < len(seq)-1 and (seq[i], seq[i+1]) == pair:
            merged.append(replacement)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    
    return merged

class ByteBPE:
    def __init__(self):
        self.vocab  = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self._encode_cache = {}

    def train(self, text: str, vocab_size: int) -> None:       
        unique_chunks = Counter(
            tuple(chunk.encode('utf-8')) for chunk in re.findall(SPLIT_PATTERN, text)
        )

        full_pair_freqs = {}
        pair_to_chunks = defaultdict(set)
         
        for chunk, weight in unique_chunks.items():
            pair_freqs(seq=chunk, counts=full_pair_freqs, weight=weight, pairs_in_seq=pair_to_chunks)

        while len(self.vocab) < vocab_size:
            most_freq_pair = max(full_pair_freqs, key=full_pair_freqs.get)
            new_token_id   = len(self.vocab)

            self.merges[most_freq_pair] = new_token_id
            self.vocab[new_token_id]    = self.vocab[most_freq_pair[0]] + self.vocab[most_freq_pair[1]]

            affected_chunks = list(pair_to_chunks[most_freq_pair])

            for chunk in affected_chunks:
                weight = unique_chunks[chunk]
                
                for pair in zip(chunk, chunk[1:]):
                    full_pair_freqs[pair] -= weight
                
                    if full_pair_freqs[pair] == 0:
                        del full_pair_freqs[pair]
                
                    pair_to_chunks[pair].discard(chunk)
                
                merged = tuple(merge_pair(seq=chunk, pair=most_freq_pair, replacement=new_token_id))
                
                for pair in zip(merged, merged[1:]):
                    full_pair_freqs[pair] = full_pair_freqs.get(pair, 0) + weight
                    pair_to_chunks[pair].add(merged)

                del unique_chunks[chunk]
                
                unique_chunks[merged] = unique_chunks.get(merged, 0) + weight
            

    def encode(self, text: str) -> list:
        chunks = [list(chunk.encode('utf-8')) for chunk in re.findall(SPLIT_PATTERN, text)]
        encoded_chunks = []
        
        for chunk in chunks:
            key = tuple(chunk)
            cached = self._encode_cache.get(key)
            if cached is None:
                while True:
                    applicable_pairs = set()
                
                    for pair in zip(chunk, chunk[1:]):
                        if pair in self.merges:
                            applicable_pairs.add(pair)
                    
                    if not applicable_pairs:
                        break
                    
                    applicable_merge = min(applicable_pairs, key=lambda merge: self.merges[merge])

                    chunk = merge_pair(seq=chunk, pair=applicable_merge, replacement=self.merges[applicable_merge])
                cached = tuple(chunk)
                self._encode_cache[key] = cached
            encoded_chunks.append(cached)
        
        return [token for encoded_chunk in encoded_chunks for token in encoded_chunk]



    def decode(self, tokens:list) -> str:
        bytes_text = b"".join(self.vocab[token] for token in tokens)
        text = bytes_text.decode('utf-8', errors='replace')

        return text

    # def save(self, path):
    #     merges = {f"{p0},{p1}": new_id for (p0, p1), new_id in self.merges.items()}

    #     with open(path, 'w') as f:
    #         json.dump(merges, f)

    # def load(self, path):
    #     with open(path, 'r') as f:
    #         merges = json.load(f)

    #     self.vocab = {i: bytes([i]) for i in range(256)}

    #     for merge, rule in merges.items():
    #         key = merge
    #         p0, p1 = key.split(',')
    #         pair = (int(p0), int(p1))

    #         self.vocab[rule] = self.vocab[pair[0]] + self.vocab[pair[1]]
    #         self.merges[pair] = rule