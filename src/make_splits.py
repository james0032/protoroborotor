#Starting from robokop/rotorobo.txt, create a training set, a test set, and a
# validation set.  The fractions are inputs to the functtion.

import random

def make_splits(train_frac = 0.8, test_frac = 0.1, input_file="robokop/rotorobo.txt",
                train_file="robokop/robo_train.txt", test_file="robokop/robo_test.txt",
                val_file="robokop/robo_val.txt"):
    with open(input_file, 'r') as reader:
        with open(train_file, 'w') as train_writer:
            with open(test_file, 'w') as test_writer:
                with open(val_file, 'w') as val_writer:
                    for line in reader:
                        r = random.random()
                        if r < train_frac:
                            train_writer.write(line)
                        elif r < train_frac + test_frac:
                            test_writer.write(line)
                        else:
                            val_writer.write(line)

if __name__ == "__main__":
    make_splits()
    print("Files created.")