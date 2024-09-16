import re
import string
import timeit
import pickle
import traceback
from pprint import pprint

# This is a cpp module. Compile janitor_util.cpp with:
# c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) janitor_util.cpp -o janitor_util$(python3-config --extension-suffix) -undefined dynamic_lookup
try:
    import janitor_util

    JANITOR_CPP = True
except Exception:
    print("WARNING: C++ module could not be loaded. Janitor running in python mode")
    traceback.print_exc()
    JANITOR_CPP = False


# Implementation from nltk source
# https://www.nltk.org/_modules/nltk/util.html
def form_ngrams(sequence, n):
    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def word_ngrams(s, n):
    """Splits a string into ngram words"""
    tokens = s.split()  # not a generator :(
    ngram_seqs = form_ngrams(iter(tokens), n)
    return (" ".join(ngram) for ngram in ngram_seqs)


# Does character sequences only - combined faster function to play around with later
# def word_ngrams_indices_combined(sequence, n):
#     current_word = ""
#     history = []
#     gap = False;
#     start = 0
#     end = 0
#     for character in sequence:
#         if character == " ":
#             if not gap:
#                 gap = True
#                 history.append(current_word)
#                 end += len(current_word) - 1
#                 current_word = ""
#                 if len(history) == n:
#                     yield (tuple(history), start, end)
#                     del history[0]
#                     start = end + 1
#                     end = start
#         else:
#             gap = False
#             current_word += character


# https://stackoverflow.com/questions/13734451/string-split-with-indices-in-python
def split_indices(s):
    """Splits a string on whitespaces and records the indices of each in the original string.
    @:return generator((word, (start_idx, end_idx)), ...)
    """
    return ((m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r"\S+", s))


def word_ngrams_indices(s, n):
    """Splits a string into pairs of (ngram words, their start/end indices)"""
    tokens_with_indices = split_indices(s)

    # Generator of ngrams of (word, idx_pairs)
    # (
    #   [(word, (start,end)), (word, (start, end))...],
    #   [(word, (start, end)), ...],
    #   ...
    # )
    ngram_seqs_with_indices = form_ngrams(tokens_with_indices, n)

    # Generator of pairs of word and index ngrams
    # (
    #   ([word, word, ...], [(start,end), (start,end), ...]),
    #   ...
    # )
    ngram_indices_pairs = (
        zip(*ngram_with_indices) for ngram_with_indices in ngram_seqs_with_indices
    )

    # Generator of ( (word_ngram, (start, end)), (word_ngram, start, end)), ...)
    return (
        (" ".join(ngram_seq), (indices[0][0], indices[-1][1]))
        for ngram_seq, indices in ngram_indices_pairs
    )


class Janitor:

    # FIXME delete_chars: Should anything else go here? Special chars?
    def __init__(
        self,
        ngram_n=13,
        window_to_remove=200,
        too_dirty_cutoff=10,
        minimum_slice_length=200,
        delete_chars=string.punctuation,
    ):
        self.ngram_n = ngram_n
        self.window_to_remove = window_to_remove
        self.too_dirty_cutoff = too_dirty_cutoff
        self.minimum_slice_length = minimum_slice_length
        self.delete_chars = delete_chars

        self.dirt_ngrams = set()

        # If in python, we'll translate uppercase to lowercase and delete naughty characters.
        # This is fast by python standards
        # https://stackoverflow.com/questions/638893/what-is-the-most-efficient-way-in-python-to-convert-a-string-to-all-lowercase-st
        self.translation_table = str.maketrans(
            string.ascii_lowercase + string.ascii_uppercase,  # These characters
            string.ascii_lowercase * 2,  # Become these characters
            self.delete_chars,  # These are deleted
        )

    ##############
    # I/O for saving contamination ngrams
    ##############

    def save_contamination_ngrams(self, filename):
        with open(filename, "wb") as fp:
            pickle.dump(filename, fp)

    def load_contamination_ngrams(self, filename):
        with open(filename, "rb") as fp:
            self.dirt_ngrams = pickle.load(fp)

    ##############
    # Call these :)
    ##############

    def register_contaminant(self, dirt_string):
        """Register a string as contamination to be removed, e.g. a test set
        This breaks the dirt_string into ngrams to store for future cleaning"""
        if JANITOR_CPP:
            return self.register_contaminant_cpp(dirt_string)
        else:
            print("WARNING: Janitor running in python mode")
            return self.register_contaminant_python(dirt_string)

    def clean(self, dirty_string):
        """Clean a string (e.g. a training set) by removing all ngrams previously
        registered as contaminants. Returns a list of clean chunks, or empty if
        the string was too dirty"""
        if JANITOR_CPP:
            return self.clean_cpp(dirty_string)
        else:
            print("WARNING: Janitor running in python mode")
            return self.clean_python(dirty_string)

    def _split_chunks(self, dirty_string, dirty_parts):
        clean_chunks = []
        splice_idx = 0
        end = -1
        for i, (ngram, start, end) in enumerate(dirty_parts):
            if i >= self.too_dirty_cutoff:
                return []
            start = max(0, start - self.window_to_remove)
            end = min(len(dirty_string), end + self.window_to_remove)

            if start - splice_idx > self.minimum_slice_length:
                clean_chunks.append(dirty_string[splice_idx:start])
            splice_idx = end

        if end < len(dirty_string) - self.minimum_slice_length:
            clean_chunks.append(dirty_string[end + 1 :])

        return clean_chunks

    ##############
    # Fast C++
    ##############

    def register_contaminant_cpp(self, dirt_string):
        self.dirt_ngrams.update(
            janitor_util.clean_ngram(dirt_string, self.delete_chars, self.ngram_n)
        )

    def clean_cpp(self, dirty_string):
        contamination_indices = janitor_util.clean_ngram_with_indices(
            dirty_string, self.delete_chars, self.ngram_n
        )
        return self._split_chunks(dirty_string, contamination_indices)

    ##############
    # Slow python
    ##############

    def normalize_string(self, s):
        return s.translate(self.translation_table)

    def register_contaminant_python(self, dirt_string):
        self.dirt_ngrams.update(
            word_ngrams(self.normalize_string(dirt_string), self.ngram_n)
        )

    def clean_python(self, dirty_string):
        contamination_indices = (
            (None, *idx_pair)
            for dirty_ngram, idx_pair in word_ngrams_indices(dirty_string, self.ngram_n)
            if self.normalize_string(dirty_ngram) in self.dirt_ngrams
        )
        return self._split_chunks(dirty_string, contamination_indices)


##################################################################
# Tests
#################################################################

# def print_cpp():
#     source = """   ,, I'm a very !dirty,, ,,  dirty boy. Clean me daddy. \n\nhe he he hehe heh.  lastword  """ * 2

#     for i in range(1, 10, 2):
#         pprint(janitor_util.clean_ngram(source, string.punctuation, i))
#         for ngram, start, end in \
#                 janitor_util.clean_ngram_with_indices(source, string.punctuation, i):
#             print(ngram, "\t", start, end, source[start:end].replace("\n", "\\n"))


# def test_cpp():
#     source = """   ,, I'm a very !dirty,, ,,  dirty boy. Clean me daddy. \n\nhe he he hehe heh.  lastword  """ * 2
#     contaminant = "dirty boy. Clean he he"

#     jan_python = Janitor()
#     jan_cpp = Janitor()

#     jan_python.register_contaminant_python(contaminant)
#     jan_cpp.register_contaminant(contaminant)

#     assert jan_python.dirt_ngrams == jan_cpp.dirt_ngrams, (jan_python.dirt_ngrams, jan_cpp.dirt_ngrams)

#     assert jan_python.clean_python(source) == jan_cpp.clean(source), \
#         (jan_python.clean_python(source), jan_cpp.clean(source))

#     print("Passed test, python==cpp")


# def benchmark():
#     # Download and put in data folder: enwik8 (100 MB) from https://cs.fit.edu/~mmahoney/compression/textdata.html
#     setup = \
#         """
#         with open("data/enwik8", "r") as f:
#             data = f.read()
#         jan = Janitor(too_dirty_cutoff=1000)
#         jan.register_contaminant('''
#         theories is that there is a connection between &quot;geekdom&quot; and autism.
#         This is hinted, for instance, by a ''Wired Magazine'' article in 2001 entitled &quot;
#         The [[Geek]] Syndrome&quot;, which is a point argued by many in the autism rights
#         movement{{ref|Wired}}.  This article, many professionals assert, is just one example of
#         the media's application of mental disease labels to what is actually variant normal behavior
#         &amp;mdash;they argue that shyness, lack of athletic ability or social skills, and intellectual
#         interests, even when they seem unusual to others, are not in themselves signs of autism or
#         Asperger's syndrome. Others assert that it is actually the medical profession which is applying
#         mental disease labels to children who in the past would have simply been accepted as a little
#         different or even labeled 'gifted'. See [[clinomorphism]] for further discussion of this issue.
#         Due to the recent publicity surrounding autism and autis
#         ultan Al Nahyan]] granted [[Petroleum]] concessions, and oil was first found in 1958.  At first,
#         oil money had a marginal impact.  A few lowrise concete buildings were erected, and the first
#         paved road was completed in 1961, but Sheikh Shakbut, uncertain whether the new oil royalties
#         would last, took a cautious approach, preferring to save the revenue rather than investing it in
#         development.  His brother, [[Zayed bin Sultan Al Nahayan]], saw that oil wealth had the potential
#         to transform Abu Dhabi.  The ruling Al Nahayan family decided that Sheikh Zayed should replace his
#         brother as Ruler and carry out his vision of developing the country.  On [[August 6]], [[1966]],
#         with the assistance of the British, Sheikh Zayed became the new ruler.  See generally, Al-Fahim, M,
#         ''From Rags to Riches: A Story of Abu Dhabi'', Chapter Six (London Centre of Arab Studies, 1995),
#         ISBN 1 900404 00 1. With the announcement by Britain in 1968 that it would withdraw from the
#         Gulf area by 1971, Sheikh Zayed became the main driving force behind the formation of the
#         [[United Arab Emirates]]. After the Emirates gained independence in 1971,
#         ''')
#         """

#     n = 1
#     print(f"Timing {n} run on 100 MB")
#     print("Register contaminant")
#     # print("\tPython", timeit.timeit("jan.register_contaminant_python(data)", setup=setup, globals=globals(), number=n))
#     print("\tCpp", timeit.timeit("jan.register_contaminant(data)", setup=setup, globals=globals(), number=n))

#     print("Clean")
#     # print("\tPython", timeit.timeit("jan.clean_python(data)", setup=setup, globals=globals(), number=n))
#     print("\tCpp", timeit.timeit("jan.clean(data)", setup=setup, globals=globals(), number=n))


# def test_janitor_general():
#     source = """   ,, I'm a very !dirty,, ,,  dirty boy. Clean me daddy. \n\nhe he he hehe heh.  lastword  """ * 2
#     contaminant = "dirty boy. Clean he he"

#     jan = Janitor(ngram_n=3)
#     jan.register_contaminant(contaminant)
#     cleaned = " ".join(jan.clean(source))
#     for contam in jan.dirt_ngrams:
#         assert contam not in cleaned, contam

#     filename = "data/saved_contam"
#     jan.save_contamination_ngrams(filename)

#     jan = Janitor(ngram_n=3)
#     jan.load_contamination_ngrams(filename)
#     cleaned = " ".join(jan.clean(source))
#     for contam in jan.dirt_ngrams:
#         assert contam not in cleaned, contam


# if __name__ == "__main__":
#     test()
#     # print_cpp()
#     # test_cpp()
#     # benchmark()
