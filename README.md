# rsvd
A simple implementation of randomized SVD according to https://arxiv.org/abs/0909.4061 (algorithms 4.4 and 5.1).

Includes implementation for:
*Matrices (rsvd.py)
*General Tensors (avoiding all reshapes and transposes, rsvd_ten.py)
*Symtensors (avoiding all reshapes and transposes, rsvd_symten.py)

# Running an example
To run an example calculation with any of the three implementations, you can use the three run*.py scripts provided. 

`python run.py pic2.jpg 10`

Where the first argument provides an input rgb image and the second argument is the number of retained singular values.
