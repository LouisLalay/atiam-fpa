"""
This script provides the solution to the needleman part of the ATIAM structure course project

 - This implements two variants of the Needleman-Wunsch algorithm
     * Basic version with simple gaps
     * Affine version (Goto algorithm) with affine gaps
 
@author: esling
"""


def QueryS(char1, char2, Smatrix, Smatrixlength):
    s1 = None
    s2 = None
    char1 = char1.upper()
    char2 = char2.upper()
    for p in range(0, Smatrixlength - 1):
        if Smatrix[0][p] == char1:
            s1 = p + 1
        if Smatrix[0][p] == char2:
            s2 = p + 1

    if s1 == None or s2 == None:
        if (char1 and char2) in (" ", "_"):
            result = int(max(Smatrix[1][1:]))
        else:
            if "*" in Smatrix[0][:]:
                for p in range(0, Smatrixlength - 1):
                    if Smatrix[0][p] == "*":
                        s1 = p

                if char1.upper() == char2.upper():
                    result = int(Smatrix[s1][s1])
                else:
                    result = int(min(Smatrix[:][s1]))
            else:
                result = int(min(Smatrix[1][1:]))
    else:
        result = int(Smatrix[s1][s2])
    return result


def needleman_simple(str1, str2, matrix="atiam-fpa_alpha.dist", gap=-2):
    l_str1 = len(str1)
    l_str2 = len(str2)
    nw_matrix = [[0 for col in range(l_str2 + 1)] for row in range(l_str1 + 1)]
    with open(matrix) as (f):
        Smatrix = list((line.split() for line in f if not line.startswith("#")))
    Smatrixlength = len(Smatrix[0])
    for i in range(1, l_str1 + 1):
        nw_matrix[i][0] = i * gap

    for j in range(1, l_str2 + 1):
        nw_matrix[0][j] = j * gap

    for i in range(1, l_str1 + 1):
        for j in range(1, l_str2 + 1):
            Match = nw_matrix[(i - 1)][(j - 1)] + QueryS(
                str1[(i - 1)], str2[(j - 1)], Smatrix, Smatrixlength
            )
            Delete = nw_matrix[(i - 1)][j] + gap
            Insert = nw_matrix[i][(j - 1)] + gap
            nw_matrix[i][j] = max(Match, Delete, Insert)

    align_str1 = ""
    align_str2 = ""
    i = l_str1
    j = l_str2
    score = int(nw_matrix[i][j])
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and nw_matrix[i][j]
            == nw_matrix[(i - 1)][(j - 1)]
            + QueryS(str1[(i - 1)], str2[(j - 1)], Smatrix, Smatrixlength)
        ):
            align_str1 = str1[(i - 1)] + align_str1
            align_str2 = str2[(j - 1)] + align_str2
            i -= 1
            j -= 1
        elif i > 0 and nw_matrix[i][j] == nw_matrix[(i - 1)][j] + gap:
            align_str1 = str1[(i - 1)] + align_str1
            align_str2 = "-" + align_str2
            i -= 1
        elif j > 0 and nw_matrix[i][j] == nw_matrix[i][(j - 1)] + gap:
            align_str1 = "-" + align_str1
            align_str2 = str2[(j - 1)] + align_str2
            j -= 1

    return (align_str1, align_str2, score)


def needleman_affine(
    str1, str2, matrix="atiam-fpa_alpha.dist", gap_open=-10, gap_extend=-2
):
    l_str1 = len(str1)
    l_str2 = len(str2)
    NINF = float("-inf")
    nw_matrix = [[0 for col in range(l_str2 + 1)] for row in range(l_str1 + 1)]
    with open(matrix) as (f):
        Smatrix = list((line.split() for line in f if not line.startswith("#")))
    Smatrixlength = len(Smatrix[0])
    for i in range(1, l_str1 + 1):
        nw_matrix[i][0] = gap_open + i * gap_extend

    for j in range(1, l_str2 + 1):
        nw_matrix[0][j] = gap_open + j * gap_extend

    for i in range(1, l_str1 + 1):
        for j in range(1, l_str2 + 1):
            Match = nw_matrix[(i - 1)][(j - 1)] + QueryS(
                str1[(i - 1)], str2[(j - 1)], Smatrix, Smatrixlength
            )
            Insert = NINF
            Delete = NINF
            for k in range(0, i):
                temp = nw_matrix[(i - k - 1)][j] + gap_open + k * gap_extend
                if temp > Delete:
                    Delete = temp

            for l in range(0, j):
                temp = nw_matrix[i][(j - l - 1)] + gap_open + l * gap_extend
                if temp > Insert:
                    Insert = temp

            nw_matrix[i][j] = max(Match, Delete, Insert)

    align_str1 = ""
    align_str2 = ""
    i = l_str1
    j = l_str2
    score = int(nw_matrix[i][j])
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and nw_matrix[i][j]
            == nw_matrix[(i - 1)][(j - 1)]
            + QueryS(str1[(i - 1)], str2[(j - 1)], Smatrix, Smatrixlength)
        ):
            align_str1 = str1[(i - 1)] + align_str1
            align_str2 = str2[(j - 1)] + align_str2
            i -= 1
            j -= 1
        else:
            for k in range(0, i):
                if (
                    nw_matrix[i][j]
                    == nw_matrix[(i - k - 1)][j] + gap_open + k * gap_extend
                ):
                    align_str1 = str1[i - k - 1 : i] + align_str1
                    align_str2 = (k + 1) * "-" + align_str2
                    i = i - k - 1

            for l in range(0, j):
                if (
                    nw_matrix[i][j]
                    == nw_matrix[i][(j - l - 1)] + gap_open + l * gap_extend
                ):
                    align_str1 = (l + 1) * "-" + align_str1
                    align_str2 = str2[j - l - 1 : j] + align_str2
                    j = j - l - 1

    return (align_str1, align_str2, score)

if __name__=="__main__":
    aligned = needleman_simple("CEELECANTH", "PELICAN", matrix="atiam-fpa_alpha.dist")
    print(aligned)
