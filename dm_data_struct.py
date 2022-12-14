"""
This script defines the overall exercise for ATIAM structure course

 - Use this as a baseline script
 - You are authorized to define other files for functions
 - Write a (small) report document (PDF) explaining your approach
 - All your files should be packed in a zip file named
     [ATIAM][FpA2020]FirstName_LastName.zip

@author: esling
"""
# Basic set of imports (here you can see if everything passes)
import pickle, logging, pretty_midi, warnings
from unittest.mock import mock_open
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, cpu_count
from needleman import needleman_simple, needleman_affine
from music21 import converter
from itertools import repeat, islice
from multiprocessing import Pool

logging.basicConfig(filename="data_struct.log", level=logging.INFO, filemode="w")

"""
 
PART 1 - Exploring a track collections (text dictionnaries) and playing with MIDI

In this part, we will start easy by looking at a collection of tracks.
The set of classical music pieces is provided in the _atiam-fpa.pkl_ file, which
is already loaded at this point of the script and contain two structures
    - composers         = Array of all composers in the database
    - composers_tracks  = Hashtable of tracks for a given composer
    
Some examples of the content of these structures
# Names of composers
composers[23] => 'Abela, Placido'
composers[1210]  => 'Beethoven, Ludwig van'
# List of tracks for each composer
composers_tracks['Abela, Placido'] => ['Ave Maria(Meditation on Prelude No. 1 by J.S.Bach)']
composers_tracks['Beethoven, Ludwig van'] => ['Drinking Song', 'Sonatine No. 3 for Mandolin and Piano', ...]
# Retrieve the first track of 
composers_tracks['Beethoven, Ludwig van'][0] => 'Drinking Song'

"""

"""

Q-1.1 Re-implement one of the array sorting algorithm seen in class
        either bubble sort or quicksort
        (+1 point bonus for quicksort)

"""


def my_sort(array: np.array, f=lambda x: x, reversed: bool = False) -> np.array:
    l = len(array)
    if l <= 1:
        return array
    else:
        a = f(array)[l // 2]
        sup = array[f(array) > a]
        eq = array[f(array) == a]
        inf = array[f(array) < a]
        if reversed:
            return np.concatenate(
                (my_sort(sup, f, reversed), eq, my_sort(inf, f, reversed)), axis=0
            )
        return np.concatenate((my_sort(inf, f), eq, my_sort(sup, f)), axis=0)


def q_1_1():
    print("Question 1.1")
    logging.info("Question 1.1 - Quicksort with sorting function as argument")
    L = np.random.randint(-100, 100, 20)
    M = np.random.randint(-100, 100, (20, 2))
    with np.printoptions(linewidth=np.inf):
        logging.info(f"Non sorted array     : {L}")
        logging.info(f"Sorted array         : {my_sort(L)}")
        logging.info(f"Sorted array (by abs): {my_sort(L, lambda x: abs(x))}")
        logging.info(f"Reverse sorted array : {my_sort(L, reversed=True)}")
    first_axis = lambda x: x[:, 0]
    second_axis = lambda x: x[:, 1]
    logging.info(f"Non sorted matrix: \n{M}")
    logging.info(f"Sorted matrix by first axis: \n{my_sort(M, first_axis)}")
    logging.info(f"Sorted matrix by second axis: \n{my_sort(M, second_axis)}")


"""

Q-1.2 Use your own algorithm to sort the collection of composers by decreasing number of tracks

"""
################
# YOUR CODE HERE
################


def q_1_2(composers_tracks: dict):
    print("Question 1.2")
    logging.info("Question 1.2 - Sorting composers by decreasing number of tracks")
    composers = np.array(list(composers_tracks.keys()))
    len_tracks = [len(composers_tracks[x]) for x in composers]
    to_sort = np.array(list(zip(composers, len_tracks)))
    sorting_f = lambda x: x[:, 1].astype(int)
    sorted_composers = my_sort(to_sort, sorting_f, True)
    with np.printoptions(linewidth=np.inf, edgeitems=10):
        logging.info(sorted_composers)


"""

Q-1.3 Extend your sorting procedure, to sort all tracks from all composers alphabetically 

"""
################
# YOUR CODE HERE
################


def q_1_3(composers_tracks: dict):
    print("Question 1.3")
    logging.info("Question 1.3 - Sorting all tracks of the database alphabetically")
    tracks = np.concatenate(list(composers_tracks.values()))
    with np.printoptions(edgeitems=20):
        logging.info(my_sort(tracks))


"""

MIDI part - In addition to the pickle file, you can find some example MIDI
files in the atiam-fpa/ folder.

Here we are going to import and plot the different MIDI files. We recommend
to use the pretty_midi library 
pip install pretty_midi
But you can rely on any method (even code your own if you want)

"""

"""

Q-1.4 Import and plot some MIDI files

Based on the provided MIDI files (random subset of Beethoven tracks), try
to import, plot and compare different files

"""

################
# YOUR CODE HERE
################
def find_pitch_boundaries(mat: np.array):
    """Input: 128xN array of MIDI information
    Ouput: Boundaries of non-empty content in mat

    This function returns the boundaries of non-empty content in an array for display purposes"""
    n_pitch, n = mat.shape
    min_pitch, max_pitch = 0, n_pitch - 1
    inf, sup = 0, 0
    while inf * sup == 0:
        if inf + mat[min_pitch].sum() == 0:
            min_pitch += 1
        else:
            inf = mat[min_pitch].sum()
        if sup + mat[max_pitch].sum() == 0:
            max_pitch -= 1
        else:
            sup = mat[max_pitch].sum()
    return min_pitch, max_pitch + 1


def note_number_to_name(note_number: np.array) -> list:
    """Input: 1xN array of note numbers between 0 and 127
    Output: 1xN array of note names between C-1 and G9"""
    # Note names within one octave
    semis = [
        "$C$",
        "$C_\#$",
        "$D$",
        "$D_\#$",
        "$E$",
        "$F$",
        "$F_\#$",
        "$G$",
        "$G_\#$",
        "$A$",
        "$A_\#$",
        "$B$",
    ]
    # Get the semitone and the octave, and concatenate to create the name
    return [f"{semis[note % 12]}{note//12 - 1}" for note in np.round(note_number)]


def plot_midi(midi_data: pretty_midi.PrettyMIDI, ax: plt.Axes) -> None:
    """Plot MIDI notes of a file, trimmed to useful content"""
    bpm = midi_data.estimate_tempo()
    fs = bpm / 60
    mat = midi_data.get_piano_roll(fs)
    min_pitch, max_pitch = find_pitch_boundaries(mat)
    y_ticks = np.arange(0, max_pitch - min_pitch, (max_pitch - min_pitch) // 4)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(note_number_to_name(y_ticks + min_pitch))
    return ax.imshow(mat[min_pitch:max_pitch], "inferno", aspect="auto", origin="lower")


def q_1_4(files, show: bool = True):
    print("Question 1.4")
    logging.info("Question 1.4 - Plot all the MIDI files available")
    text = ["No Display", "Display ON "]
    print(text[show * 1])
    if show:
        n_files = len(files)
        n_rows = round(np.sqrt(n_files))
        n_cols = n_files // n_rows + (n_files % n_rows > 0) * 1
        fig, axes = plt.subplots(n_rows, n_cols)
        for id, midi_data in enumerate(files):
            im = plot_midi(midi_data, axes[id // n_cols, id % n_cols])
        plt.show()


"""

Q-1.5 Compute the number of notes in a MIDI and sort the collection

First write a function counting the number of notes played in a given MIDI
file. Then, sort the set of MIDI files based on the number of notes.

"""
################
# YOUR CODE HERE
################


def count_notes(midi_file: pretty_midi.PrettyMIDI):
    try:
        return (midi_file.get_piano_roll(1) > 0).sum()
    except:
        return -1


def q_1_5(files, midi_files):
    print("Question 1.5")
    logging.info("Question 1.5")

    n_notes = [count_notes(x) for x in midi_files]
    to_sort = np.array(list(zip(files, n_notes)))
    sorting_f = lambda x: x[:, 1].astype(int)
    sorted_files = my_sort(to_sort, sorting_f, True)
    logging.info(sorted_files)


"""
 
PART 2 - Symbolic alignments and simple text dictionnaries

In this part, we will use our knowledge on computer structures to solve a very 
well-known problem of string alignement. Hence, this part is split between
  1 - Implement a string alignment 
  2 - Try to apply this to a collection of classical music pieces names
  3 - Develop your own more adapted procedure to have a matching inside large set
  
The set of classical music pieces is provided in the atiam-fpa.pkl file, which
is already loaded at this point of the script and contain two structures
    - composers         = Array of all composers in the database
    - composers_tracks  = Hashtable of tracks for a given composer
    
Some examples of the content of these structures

composers[23] => 'Abela, Placido'
composers[1210]  => 'Beethoven, Ludwig van'

composers_tracks['Abela, Placido'] => ['Ave Maria(Meditation on Prelude No. 1 by J.S.Bach)']
composers_tracks['Beethoven, Ludwig van'] => ['"Ode to Joy"  (Arrang.)', '10 National Airs with Variations, Op.107 ', ...]

composers_tracks['Beethoven, Ludwig van'][0] => '"Ode to Joy"  (Arrang.)'

"""

# Question 1 - Reimplementing the simple NW alignment

"""

Q-2.1 Here perform your Needleman-Wunsch (NW) implementation.
    - You can find the definition of the basic NW here
    https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
    - In this first version, we will be implementing the _basic_ gap costs
    - Remember to rely on a user-defined matrix for symbols distance

"""


def substitution_indice(char: str, alpha: str):
    # ord('A') = 65, ord('Z') = 90
    if char in alpha:
        return ord(char) - 65
    return -1


def reconstruct(traceback_matrix, arg1, arg2):
    i, j = -1, -1
    res1, res2 = [], []
    traceback = 0
    while traceback != -1:
        traceback = traceback_matrix[i, j]
        if traceback == 0:
            # Match
            res1.append(arg1[i])
            res2.append(arg2[j])
            i -= 1
            j -= 1
        elif traceback == 1:
            # Insertion
            res1.append(arg1[i])
            res2.append("-")
            i -= 1
        elif traceback == 2:
            # Deletion
            res1.append("-")
            res2.append(arg2[j])
            j -= 1
    res1.reverse()
    res2.reverse()
    return res1, res2


def my_needleman_simple(
    arg1: str,
    arg2: str,
    substitution_matrix: np.array,
    alpha: str,
    gap: int = -5,
):
    arg1 = arg1.upper()
    arg2 = arg2.upper()
    n1 = len(arg1)
    n2 = len(arg2)
    score_matrix = np.zeros((n1 + 1, n2 + 1))
    # 0 stands for match
    traceback_matrix = -np.ones((n1 + 1, n2 + 1))
    score_matrix[1:, 0] = gap * np.arange(1, n1 + 1)
    # 1 stands for insertion
    traceback_matrix[1:, 0] = 1
    score_matrix[0, 1:] = gap * np.arange(1, n2 + 1)
    # 2 stands for deletion
    traceback_matrix[0, 1:] = 2
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            ind1, ind2 = substitution_indice(arg1[i - 1], alpha), substitution_indice(
                arg2[j - 1], alpha
            )
            score_matrix[i, j], traceback_matrix[i, j] = max(
                (
                    (score_matrix[i - 1, j - 1] + substitution_matrix[ind1, ind2], 0),
                    (score_matrix[i - 1, j] + gap, 1),
                    (score_matrix[i, j - 1] + gap, 2),
                )
            )
    return (score_matrix[-1, -1], traceback_matrix)


def my_needleman_opti(
    arg1: str,
    arg2: str,
    substitution_matrix: np.array,
    alpha: str,
    gap: int = -5,
):
    n1 = len(arg1)
    n2 = len(arg2)
    score_matrix = np.zeros((n1 + 1, n2 + 1))
    score_matrix[1:, 0] = gap * np.arange(1, n1 + 1)
    score_matrix[0, 1:] = gap * np.arange(1, n2 + 1)
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            ind1 = ord(arg1[i - 1]) - 65 if arg1[i - 1] in alpha else -1
            ind2 = ord(arg2[j - 1]) - 65 if arg2[j - 1] in alpha else -1
            score_matrix[i, j] = max(
                score_matrix[i - 1, j - 1] + substitution_matrix[ind1, ind2],
                score_matrix[i - 1, j] + gap,
                score_matrix[i, j - 1] + gap,
            )
    return score_matrix[-1, -1]


################
# YOUR CODE HERE
################
from time import perf_counter


def q_2_1(matrix):
    print("Question 2.1")
    logging.info("Question 2.1")
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    arg1 = "CEELECANTH"
    arg2 = "PELICAN"
    score, traceback_matrix = my_needleman_simple(arg1, arg2, matrix, alpha, gap=-2)
    str1, str2 = reconstruct(traceback_matrix, arg1, arg2)
    logging.info(f"Needleman-Wunsch:\n{''.join(str1)}\n{''.join(str2)}\nScore: {score}")
    # # Reference code for testing
    aligned = needleman_simple(
        "CEELECANTH", "PELICAN", matrix="atiam-fpa_alpha.dist", gap=-2
    )
    logging.info(
        f"Results for basic gap costs (linear):\n{aligned[0]}\n{aligned[1]}\nScore: {aligned[2]}"
    )


"""

Q-2.2 Apply the NW algorithm between all tracks of each composer
    * For each track of a composer, compare to all remaining tracks of the same composer
    * Establish a cut criterion (what is the relevant similarity level ?) to only print relevant matches
    * Propose a set of matching tracks and save it through Pickle
    
"""

################
# YOUR CODE HERE
################
def compare_names(items: tuple, matrix, alpha: str, lim_len=10, lim_score=15):
    matches = []
    # Sets do not store duplicates
    _, tracks = items
    tracks = set(tracks)
    for name1 in tracks:
        tracks = tracks - {name1}
        name1 = name1.upper()
        for name2 in tracks:
            if abs(len(name1) - len(name2)) > lim_len:
                score = -1
            else:
                score = my_needleman_opti(name1, name2.upper(), matrix, alpha)
            if score >= lim_score:
                matches.append((name1, name2))
    return matches


def q_2_2(composers_tracks: dict, matrix: np.array):
    print("Question 2.2")
    logging.info(
        "Question 2.2 - Seek similarities between track names of each composers - Truncated problem"
    )
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lim_score = 80
    lim_len = 8
    n = len(composers_tracks)
    matches = []
    for i, item in enumerate(composers_tracks.items()):
        # Sets do not store duplicates
        m = len(set(item[1]))
        print(f"Progression: {i}/{n} composers. {m} names to compare{' '*10}", end="\r")
        matches.append(compare_names(item, matrix, alpha, lim_len, lim_score))
    logging.info(matches)


def q_2_2_multi_proc(composers_tracks: dict, matrix: np.array):
    print(f"Question 2.2 - Takes around 3 minutes{' '*10}")
    logging.info(
        "Question 2.2 - Seek similarities between track names of each composers - Multiprocessing version"
    )
    lim_score = 80
    lim_len = 8
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    args = zip(
        composers_tracks.items(),
        repeat(matrix),
        repeat(alpha),
        repeat(lim_len),
        repeat(lim_score),
    )
    with Pool(cpu_count()) as pool:
        matches = pool.starmap(compare_names, args)
    with open("name_matches.txt", "w", encoding="utf-8") as file:
        file.write(f"{matches}")


"""

Q-2.3 Extend your previous code so that it can compare
    * A given track to all tracks of all composers (full database)
    * You should see that the time taken is untractable (computational explosion)
    * Propose a method to avoid such a huge amount of computation
    * Establish a cut criterion (what is relevant similarity)
    * Propose a set of matching tracks and save it through Pickle
    
"""


def compare_names_all_dataset(
    split_names: np.array,
    all_names: np.array,
    matrix,
    alpha,
    lim_score: int,
    lim_len: int,
):
    matches = []
    # Sets do not store duplicates
    for name1 in set(split_names):
        compare_set = all_names - set(name1)
        name1 = name1.upper()
        for name2 in compare_set:
            name2 = name2.upper()
            if abs(len(name1) - len(name2)) > lim_len:
                score = -1
            else:
                score = my_needleman_opti(name1, name2.upper(), matrix, alpha)
            if score >= lim_score:
                matches.append((name1, name2))
    return matches


################
# YOUR CODE HERE
################

# The following function is the brute-force approach, but a smarter way
# would be to first sort the names in small groups of similar names, then
# only take one name per group to compare all the names indirectly.
# For timing reasons, this is not yet implemented
def q_2_3(composers_tracks: dict, matrix: np.array):
    print("Question 2.3 - Takes way too much time, don't run it")
    logging.info("Question 2.3 - Needleman accross all names")

    n_cpu = cpu_count()

    names = np.concatenate(list(composers_tracks.values()))
    lim_score = 80
    lim_len = 8
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = names.shape[0]
    r = n % n_cpu
    split_names = np.split(names[:-r], n_cpu)
    temp = np.empty(split_names[0].shape[0] + r, dtype="<U256")
    temp[:-r] = split_names[0]
    temp[-r:] = names[-r:]
    split_names[0] = temp
    args = zip(
        split_names,
        repeat(set(names)),
        repeat(matrix),
        repeat(alpha),
        repeat(lim_len),
        repeat(lim_score),
    )
    with Pool(n_cpu) as pool:
        matches = pool.starmap(compare_names_all_dataset, args)
    with open("name_matches_all_dataset.txt", "w", encoding="utf-8") as file:
        file.write(f"{matches}")


"""

PART 3 - Extending the alignment algorithm and musical matching

You might have seen from the previous results that
        - Purely string matching on classical music names is not the best approach
        - This mostly comes from the fact that the importance of symbols is not the same
        - For instance 
            "Symphony for orchestra in D minor"
            "Symphony for orchestra in E minor"
          Looks extremely close but the key is the most important symbol
  
Another flaw in our approach is that the NW algorithm treats all gaps
equivalently. Hence, it can put lots of small gaps everywhere.
Regarding alignement, it would be better to have long coherent gaps rather
than small ones. This is handled by a mecanism known as _affine gap penalty_
which separates the costs of either _opening_ or _extending_ a gap. This
is known as the Gotoh algorithm, which can be found here :
    - http://helios.mi.parisdescartes.fr/~lomn/Cours/BI/Material2019/gap-penalty-gotoh.pdf

"""
"""

Q-3.1 Extending to a true musical name matching
    * Start by exploring the collection for well-known composers, what do you see ?
    * Propose a new name matching algorithm adapted to classical music piece names
        - Can be based on a rule-based system
        - Can be a pre-processing for symbol finding and then an adapted weight matrix
        - Can be a local-alignement procedure
        (These are only given as indicative ideas ...)
    * Implement this new comparison procedure adapted to classical music piece names
    * Re-run your previous results (Q-2.2 and Q-2.3) with this procedure
    
"""

################
# YOUR CODE HERE
################
def q_3_1(composers_tracks):
    print("Question 3.1")
    logging.info("Question 3.1")
    logging.info(f"{composers_tracks['Mozart, Wolfgang Amadeus']=}")
    # Many pieces names are duplicates and many only for one digit
"""

Q-3.2 Extending the NW algorithm 
    * Add the affine gap penalty to your original NW algorithm
    * You can use the Gotoh algorithm reference
    * Verify your code by using the provided compiled version
    
"""

################
# YOUR CODE HERE
################


def q_3_2():
    print("Question 3.2")
    logging.info("Question 3.2")
    aligned = needleman_affine("CEELECANTH", "PELICAN", matrix='atiam-fpa_alpha.dist', gap_open=-5, gap_extend=-2)
    print('Results for affine gap costs')
    print(aligned[0])
    print(aligned[1])
    print('Score : ' + str(aligned[2]))


"""
 
PART 4 - Alignments between MIDI files and error-detection

Interestingly the problem of string alignment can be extended to the more global 
problem of aligning any series of symbolic information (vectors). Therefore,
we can see that the natural extension of this problem is to align any sequence
of symbolic information.

This definition matches very neatly to the alignement of two musical scores 
that can then be used as symbolic similarity between music, or score following.
However, this requires several key enhancements to the previous approach. 
Furthermore, MIDI files gathered on the web are usually of poor quality and 
require to be checked. Hence, here you will
    1 - Learn how to read and watch MIDI files
    2 - Explore their properties to perform some quality checking
    3 - Extend alignment to symbolic score alignement
    
To fasten the pace of your musical analysis, we will rely on the excellent 
Music21 library, which provides all sorts of musicological analysis and 
properties over symbolic scores. You will need to really perform this part
to go and read the documentation of this library online

"""

# Question 4 - Importing and plotting MIDI files (using Music21)

import math


def get_start_time(el, measure_offset, quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(
            math.ceil(
                ((measure_offset[el.measureNumber] or 0) + el.offset) * quantization
            )
        )
    # Else, no time defined for this element and the function returns None


def get_end_time(el, measure_offset, quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(
            math.ceil(
                (
                    (measure_offset[el.measureNumber] or 0)
                    + el.offset
                    + el.duration.quarterLength
                )
                * quantization
            )
        )
    # Else, no time defined for this element and the function returns None


def get_pianoroll_part(part, quantization):
    # Get the measure offsets
    measure_offset = {None: 0}
    for el in part.recurse(classFilter=("Measure")):
        measure_offset[el.measureNumber] = el.offset
    # Get the duration of the part
    duration_max = 0
    for el in part.recurse(classFilter=("Note", "Rest")):
        t_end = get_end_time(el, measure_offset, quantization)
        if t_end > duration_max:
            duration_max = t_end
    # Get the pitch and offset+duration
    piano_roll_part = np.zeros((128, math.ceil(duration_max)))
    for this_note in part.recurse(classFilter=("Note")):
        note_start = get_start_time(this_note, measure_offset, quantization)
        note_end = get_end_time(this_note, measure_offset, quantization)
        piano_roll_part[this_note.midi, note_start:note_end] = 1
    return piano_roll_part


# Here we provide a MIDI import function
def importMIDI(f):
    piece = converter.parse(f)
    all_parts = {}
    for part in piece.parts:
        print(part)
        try:
            track_name = part[0].bestName()
        except AttributeError:
            track_name = "None"
        cur_part = get_pianoroll_part(part, 16)
        if cur_part.shape[1] > 0:
            all_parts[track_name] = cur_part
    print("Returning")
    return piece, all_parts


################
# YOUR CODE HERE
################

"""

Q-4.1 Exploring MIDI properties

The Music21 library propose a lot of properties directly on the piece element,
but we also provide separately a dictionary containing for each part a matrix
representation (pianoroll) of the corresponding notes (without dynamics).
    - By relying on Music21 documentation (http://web.mit.edu/music21/doc/)
        * Explore various musicology properties proposed by the library
        * Check which could be used to assess the quality of MIDI files

"""


def example_4_1(file):
    piece, all_parts = importMIDI(file)
    # # Here a few properties that can be plotted ...
    piece.plot("scatter", "quarterLength", "pitch")
    piece.plot("scatterweighted", "pitch", "quarterLength")
    piece.plot("histogram", "pitchClass")
    # Here is the list of all MIDI parts (with a pianoroll matrix)
    for key, val in sorted(all_parts.items()):
        print("Instrument: %s has content: %s " % (key, val))


################
# YOUR CODE HERE
################
def q_4_1(file: str):
    print("Question 4.1")
    logging.info("Question 4.1 - Exploring Music21")

    midi_data = converter.parse(file)

"""

Q-4.2 Automatic evaluation of a MIDI file quality

One of the most pervasive problem with MIDI scores is that a large part of the
files that you can find on the internet are of rather low quality.
Based on your exploration in the previous questions and your own intuition,
    - Propose an automatic procedure that could evaluate the quality of a MIDI file.
    - Test how this could be used on a whole set of files

"""

################
# YOUR CODE HERE
################
def q_4_2():
    print("Question 4.2")
    logging.info("Question 4.2")


"""

Q-4.3 Extending your alignment algorithm to MIDI scores

As explained earlier, our alignment algorithm can work with any set of symbols,
which of course include even complex scores. The whole trick here is to see
that the "distance matrix" previously used could simply be replaced by a
"distance function", which can represent the similarity between any elements
    - Propose a fit distance measures between two slices of pianorolls
    - Modify your previous algorithm so that it can use your distance
    - Modify the algorithm so that it can work with MIDI files
    - Apply your algorithm to sets of MIDI files

"""

################
# YOUR CODE HERE
################
def q_4_3():
    print("Question 4.3")
    logging.info("Question 4.3")


def main():
    # Loading database
    midi_database = pickle.load(open("atiam-fpa.pkl", "rb"))
    composers_tracks = midi_database["composers_tracks"]
    # Extracting file names
    files = ["atiam-fpa/" + f for f in listdir("atiam-fpa")]
    # Extracting MIDI informations with pretty_midi
    pretty_midi_files = []
    for file in files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pretty_midi_files.append(pretty_midi.PrettyMIDI(file))
        except:
            print(f"Bad file {file}")
    # Extracting alphabet substitution matrix
    with open("atiam-fpa_alpha.dist", "r") as file:
        lines = file.readlines()
    basic_alphabet_matrix = np.array(
        [l.rstrip("\n").split("  ")[1:-1] for l in lines[1:]]
    ).astype("int")
    # Adding an extra dimension for spaces and special chars
    alphabet_matrix = -3 * np.ones(
        (basic_alphabet_matrix.shape[0] + 1, basic_alphabet_matrix.shape[1] + 1)
    )
    alphabet_matrix[-1, -1] = 5
    alphabet_matrix[:-1, :-1] = basic_alphabet_matrix
    # Extracting dna matrix
    with open("atiam-fpa_dna.dist", "r") as file:
        lines = file.readlines()
    lines = [l for l in lines if l[0] != "#"]
    dna_matrix = np.array([l.rstrip("\n").split()[1:-1] for l in lines[1:]]).astype(
        "int"
    )
    print("######### Part 1 #########")
    q_1_1()
    q_1_2(composers_tracks)
    q_1_3(composers_tracks)
    q_1_4(pretty_midi_files, True)
    q_1_5(files, pretty_midi_files)

    print("######### Part 2 #########")
    q_2_1(alphabet_matrix)
    # Only a fraction is given for the single process version
    q_2_2(dict(islice(composers_tracks.items(), 1, 10)), alphabet_matrix)
    q_2_2_multi_proc(composers_tracks, alphabet_matrix)
    # Same reason
    q_2_3(dict(islice(composers_tracks.items(), 1, 10)), alphabet_matrix)

    print("######### Part 3 #########")
    q_3_1(composers_tracks)
    q_3_2()

    print("######### Part 4 #########")
    q_4_1(files[0])
    q_4_2()
    q_4_3()


if __name__ == "__main__":
    main()
