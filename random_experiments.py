# experiment 1) obtain 2 FV of same file.

# output: an exact same face vector is produced
import filecmp

print(filecmp.cmp('temp/vector.json', 'temp/vector_1.json'))

# does augmented images produce the same face vectors?
