import json

class ByteBPE:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def train(self, text, vocab_size):
        bytes_text = list(text.encode('utf-8'))

        while len(self.vocab) < vocab_size:
            pairs = list(zip(bytes_text, bytes_text[1:]))
            counts = {}

            for pair in pairs:
                counts[pair] = counts.get(pair, 0) + 1
            max_pair = max(counts, key=counts.get)
            new_id = len(self.vocab)

            self.merges[max_pair] = new_id
            self.vocab[new_id] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]

            new_bytes = []
            i = 0
            while i < len(bytes_text):
                if i < len(bytes_text) - 1 and bytes_text[i] == max_pair[0] and bytes_text[i+1] == max_pair[1]:
                    new_bytes.append(new_id)
                    i += 2
                else:
                    new_bytes.append(bytes_text[i])
                    i += 1
            bytes_text = new_bytes

    def encode(self, text):
        bytes_text = list(text.encode('utf-8'))

        for merge, new_id in self.merges.items():
            new_bytes = []
            i = 0

            while i < len(bytes_text):
                if i < len(bytes_text) - 1 and bytes_text[i] == merge[0] and bytes_text[i+1] == merge[1]:
                    new_bytes.append(new_id)
                    i += 2
                else:
                    new_bytes.append(bytes_text[i])
                    i += 1
            bytes_text = new_bytes

        return bytes_text


    def decode(self, ids):
        bytes_text = b"".join(self.vocab[id] for id in ids)
        return bytes_text.decode('utf-8')

    def save(self, path):
        merges = {f"{p0},{p1}": new_id for (p0, p1), new_id in self.merges.items()}

        with open(path, 'w') as f:
            json.dump(merges, f)

    def load(self, path):
        with open(path, 'r') as f:
            merges = json.load(f)

        self.vocab = {i: bytes([i]) for i in range(256)}

        for merge, rule in merges.items():
            key = merge
            p0, p1 = key.split(',')
            pair = (int(p0), int(p1))

            self.vocab[rule] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = rule