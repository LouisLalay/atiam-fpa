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
from ast import ListComp
import pickle
from xmlrpc.client import Boolean

# Here collect the whole set of tracks
def init():
    midi_database = pickle.load(open("atiam-fpa.pkl", "rb"))
    composers = midi_database['composers']
    composers_tracks = midi_database['composers_tracks']
    return composers, composers_tracks

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
import logging
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(filename = 'data_struct.log', level = logging.INFO, filemode = 'w')

'''

Q-1.1 Re-implement one of the array sorting algorithm seen in class
        either bubble sort or quicksort
        (+1 point bonus for quicksort)

'''
def my_sort(array: np.array, f = lambda x : x) -> np.array:
    l = len(array)
    if l<=1:
        return array
    else:
        a = f(array)[l//2]
        sup = array[f(array)>a]
        eq = array[f(array)==a]
        inf = array[f(array)<a]
        return np.concatenate((my_sort(inf, f), eq, my_sort(sup, f)), axis=0)
def q_1_1():
    L = np.random.randint(-100, 100, 20)
    M = np.random.randint(-100, 100, (20,2))
    with np.printoptions(linewidth=np.inf):
        logging.info(f"Non sorted array     : {L}")
        logging.info(f"Sorted array         : {my_sort(L)}")
        logging.info(f"Sorted array (by abs): {my_sort(L, lambda x: abs(x))}")
    first_axis = lambda x: x[:,0]
    second_axis = lambda x: x[:,1]
    logging.info(f"Non sorted matrix: \n{M}")
    logging.info(f"Sorted matrix by first axis: \n{my_sort(M, first_axis)}")
    logging.info(f"Sorted matrix by second axis: \n{my_sort(M, second_axis)}")
'''

Q-1.2 Use your own algorithm to sort the collection of composers by decreasing number of tracks

'''
################
# YOUR CODE HERE
################

def q_1_2(composers: list, composers_tracks: dict):
    logging.info(f"{type(composers)=}")
    logging.info(f"{type(composers_tracks)=}")
    logging.info(f"{len(composers_tracks.keys())=}")
    logging.info(f"{len(composers)=}")
    keys = list(composers_tracks.keys())
    val = list(composers_tracks.values())
    val = [len(x) for x in val]
    list_composers = np.array(list(zip(keys,val)))
    sorting_f = lambda x: x[:,1]
    sorted_composers = my_sort(list_composers, sorting_f)
    logging.info(f"Sorted composers by number of tracks: {sorted_composers}")
    with np.set_printoptions(edge = 50):
        logging.info(f"Sorted number of tracks: {sorting_f(sorted_composers)}")

'''

Q-1.3 Extend your sorting procedure, to sort all tracks from all composers alphabetically 

'''
################
# YOUR CODE HERE
################

"""

MIDI part - In addition to the pickle file, you can find some example MIDI
files in the atiam-fpa/ folder.

Here we are going to import and plot the different MIDI files. We recommend
to use the pretty_midi library 
pip install pretty_midi
But you can rely on any method (even code your own if you want)

"""

'''

Q-1.4 Import and plot some MIDI files

Based on the provided MIDI files (random subset of Beethoven tracks), try
to import, plot and compare different files

'''

################
# YOUR CODE HERE
################
def find_pitch_boundaries(mat: np.array):
    '''Input: 128xN array of MIDI information
    Ouput: Boundaries of non-empty content in mat
    
    This function returns the boundaries of non-empty content in an array for display purposes'''
    n_pitch, n = mat.shape
    min_pitch, max_pitch = 0, n_pitch-1
    inf, sup = 0, 0
    while inf*sup==0:
        if inf+mat[min_pitch].sum()==0:
            min_pitch+=1
        else:
            inf = mat[min_pitch].sum()
        if sup + mat[max_pitch].sum()==0:
            max_pitch-=1
        else:
            sup = mat[max_pitch].sum()
    return min_pitch, max_pitch+1
def note_number_to_name(note_number: np.array) -> np.array:
    '''Input: 1xN array of note numbers between 0 and 127
    Output: 1xN array of note names between C-1 and G9'''
    # Note names within one octave
    semis = ['$C$', '$C_\#$', '$D$', '$D_\#$', '$E$', '$F$', '$F_\#$', '$G$', '$G_\#$', '$A$', '$A_\#$', '$B$']
    # Get the semitone and the octave, and concatenate to create the name
    return [f"{semis[note % 12]}{note//12 - 1}" for note in np.round(note_number)]
def plot_midi(midi_data: pretty_midi.PrettyMIDI, ax: plt.Axes) -> None:
    '''Plot MIDI notes of a file, trimmed to useful content'''
    bpm = midi_data.estimate_tempo()
    fs = bpm/60
    mat = midi_data.get_piano_roll(fs)
    min_pitch, max_pitch = find_pitch_boundaries(mat)
    y_ticks = np.arange(0, max_pitch-min_pitch, (max_pitch-min_pitch)//4)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(note_number_to_name(y_ticks+min_pitch))
    return ax.imshow(mat[min_pitch : max_pitch], 'inferno', aspect="auto", origin='lower')
def q_1_4(affichage: Boolean = True):
    folder = "atiam-fpa/"
    files = [f"beethoven_{i}.mid" for i in range(25)]
    fig, axes = plt.subplots(5)
    if affichage:
        for id, file in enumerate(files[::5]):
            midi_data = pretty_midi.PrettyMIDI(folder + file)
            im = plot_midi(midi_data, axes[id])
            fig.colorbar(im)
        plt.show()
'''

Q-1.5 Compute the number of notes in a MIDI and sort the collection

First write a function counting the number of notes played in a given MIDI
file. Then, sort the set of MIDI files based on the number of notes.

'''
################
# YOUR CODE HERE
################

def count_notes(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    return (midi_data.get_piano_roll(1)>0).sum()
def q_1_5():
    ...
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

'''

Q-2.1 Here perform your Needleman-Wunsch (NW) implementation.
    - You can find the definition of the basic NW here
    https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
    - In this first version, we will be implementing the _basic_ gap costs
    - Remember to rely on a user-defined matrix for symbols distance

'''
def my_needleman_simple(str1, str2, matrix='atiam-fpa_alpha.dist', gap_open=-5, gap_extend=-5):
    score = 0
    ################
    # YOUR CODE HERE
    ################
    return ('', '', score)

# # Reference code for testing
# from needleman import needleman_simple
# aligned = needleman_simple("CEELECANTH", "PELICAN", matrix='atiam-fpa_alpha.dist', gap=-2)
# print('Results for basic gap costs (linear)')
# print(aligned[0])
# print(aligned[1])
# print('Score : ' + str(aligned[2]))

# Question 2 - Applying this to a collection of musical scores

################
# YOUR CODE HERE
################

'''

Q-2.2 Apply the NW algorithm between all tracks of each composer
    * For each track of a composer, compare to all remaining tracks of the same composer
    * Establish a cut criterion (what is the relevant similarity level ?) to only print relevant matches
    * Propose a set of matching tracks and save it through Pickle
    
'''

################
# YOUR CODE HERE
################

'''

Q-2.3 Extend your previous code so that it can compare
    * A given track to all tracks of all composers (full database)
    * You should see that the time taken is untractable (computational explosion)
    * Propose a method to avoid such a huge amount of computation
    * Establish a cut criterion (what is relevant similarity)
    * Propose a set of matching tracks and save it through Pickle
    
'''

################
# YOUR CODE HERE
################

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
'''

Q-3.1 Extending to a true musical name matching
    * Start by exploring the collection for well-known composers, what do you see ?
    * Propose a new name matching algorithm adapted to classical music piece names
        - Can be based on a rule-based system
        - Can be a pre-processing for symbol finding and then an adapted weight matrix
        - Can be a local-alignement procedure
        (These are only given as indicative ideas ...)
    * Implement this new comparison procedure adapted to classical music piece names
    * Re-run your previous results (Q-2.2 and Q-2.3) with this procedure
    
'''

################
# YOUR CODE HERE
################    
    
'''

Q-3.2 Extending the NW algorithm 
    * Add the affine gap penalty to your original NW algorithm
    * You can use the Gotoh algorithm reference
    * Verify your code by using the provided compiled version
    
'''

################
# YOUR CODE HERE
################

# from needleman import needleman_affine
# aligned = needleman_affine("CEELECANTH", "PELICAN", matrix='atiam-fpa_alpha.dist', gap_open=-5, gap_extend=-2)
# print('Results for affine gap costs')
# print(aligned[0])
# print(aligned[1])
# print('Score : ' + str(aligned[2]))


#%% 
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

#%% Question 4 - Importing and plotting MIDI files (using Music21)

import math
import numpy as np
from music21 import converter

def get_start_time(el,measure_offset,quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil(((measure_offset[el.measureNumber] or 0) + el.offset)*quantization))
    # Else, no time defined for this element and the functino return None

def get_end_time(el,measure_offset,quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil(((measure_offset[el.measureNumber] or 0) + el.offset + el.duration.quarterLength)*quantization))
    # Else, no time defined for this element and the functino return None
    
def get_pianoroll_part(part,quantization):
    # Get the measure offsets
    measure_offset = {None:0}
    for el in part.recurse(classFilter=('Measure')):
        measure_offset[el.measureNumber] = el.offset
    # Get the duration of the part
    duration_max = 0
    for el in part.recurse(classFilter=('Note','Rest')):
        t_end = get_end_time(el,measure_offset,quantization)
        if(t_end>duration_max):
            duration_max=t_end
    # Get the pitch and offset+duration
    piano_roll_part = np.zeros((128,math.ceil(duration_max)))
    for this_note in part.recurse(classFilter=('Note')):
        note_start = get_start_time(this_note,measure_offset,quantization)
        note_end = get_end_time(this_note,measure_offset,quantization)
        piano_roll_part[this_note.midi,note_start:note_end] = 1
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
            track_name = 'None'
        cur_part = get_pianoroll_part(part, 16)
        if (cur_part.shape[1] > 0):
            all_parts[track_name] = cur_part
    print('Returning')
    return piece, all_parts



################
# YOUR CODE HERE
################

'''

Q-4.1 Exploring MIDI properties

The Music21 library propose a lot of properties directly on the piece element,
but we also provide separately a dictionary containing for each part a matrix
representation (pianoroll) of the corresponding notes (without dynamics).
    - By relying on Music21 documentation (http://web.mit.edu/music21/doc/)
        * Explore various musicology properties proposed by the library
        * Check which could be used to assess the quality of MIDI files

'''

# # Here a few properties that can be plotted ...
# piece.plot('scatter', 'quarterLength', 'pitch')
# piece.plot('scatterweighted', 'pitch', 'quarterLength')
# piece.plot('histogram', 'pitchClass')
# # Here is the list of all MIDI parts (with a pianoroll matrix)
# for key, val in sorted(all_parts.items()):
#     print('Instrument: %s has content: %s ' % (key, val))

################
# YOUR CODE HERE
################

'''

Q-4.2 Automatic evaluation of a MIDI file quality

One of the most pervasive problem with MIDI scores is that a large part of the
files that you can find on the internet are of rather low quality.
Based on your exploration in the previous questions and your own intuition,
    - Propose an automatic procedure that could evaluate the quality of a MIDI file.
    - Test how this could be used on a whole set of files

'''

################
# YOUR CODE HERE
################

'''

Q-4.3 Extending your alignment algorithm to MIDI scores

As explained earlier, our alignment algorithm can work with any set of symbols,
which of course include even complex scores. The whole trick here is to see
that the "distance matrix" previously used could simply be replaced by a
"distance function", which can represent the similarity between any elements
    - Propose a fit distance measures between two slices of pianorolls
    - Modify your previous algorithm so that it can use your distance
    - Modify the algorithm so that it can work with MIDI files
    - Apply your algorithm to sets of MIDI files

'''

################
# YOUR CODE HERE
################
def main():
    composers, composers_tracks = init()
    composers, composers_tracks = init()
    print("Question 1.1")
    logging.info("Question 1.1")
    q_1_1()
    print("Question 1.2")
    logging.info("Question 1.2")
    q_1_2(composers, composers_tracks)
    print("Question 1.4")
    q_1_4(affichage=False)
    print("Question 1.5")
    q_1_5()
if __name__ == '__main__':
    main()