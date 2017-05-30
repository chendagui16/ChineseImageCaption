# @ Author: Dagui Chen
# @ Email: goblin_chen@163.com
# @ Date: 2017-05-08
# ========================================
import heapq
import math


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, logprob, score):
        """Initializes the Caption.

        Args:
          sentence: List of word ids in the caption.
          logprob: Log-probability of the caption.
          score: Score of the caption.
        """
        self.sentence = sentence
        self.logprob = logprob
        self.score = score

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For python3 compatibility
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
            sort: Whether to return the elements in descending sorted order.

        Returns:
            A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):
    def __init__(self, model, ctable, caption_len, beam_size=3, length_normalization_factor=0.0):
        self.ctable = ctable
        self.model = model  # the Caption model which must have methods: get_image_output and inference_step
        self.beam_size = beam_size
        self.caption_len = caption_len + 2  # add start word and stop word
        # bigger length_normalization_fator indicates that longer captions will be favored
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, img_feature):
        image_embedding = self.model.get_image_output(img_feature[None, ...])
        initial_beam = Caption(sentence=[self.ctable.start_idx],
                               logprob=0.0, score=0.0)
        part_captions = TopN(self.beam_size)
        part_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size)
        for _ in range(self.caption_len - 1):
            part_captions_list = part_captions.extract()
            part_captions.reset()
            sentence_feed = [c.sentence for c in part_captions_list]
            for i, part_caption in enumerate(part_captions_list):
                softmax = self.model.inference_step(image_embedding, sentence_feed[i]).squeeze()
                words_and_probs = list(enumerate(softmax))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0)
                    sentence = part_caption.sentence + [w]
                    logprob = part_caption.logprob + math.log(p)
                    score = logprob
                    if w == self.ctable.start_idx:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence)**self.length_normalization_factor
                        beam = Caption(sentence, logprob, score)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, logprob, score)
                        part_captions.push(beam)
            if part_captions.size() == 0:
                break
            if not complete_captions.size():
                complete_captions = part_captions
        return [c.sentence for c in complete_captions.extract(sort=True)]
