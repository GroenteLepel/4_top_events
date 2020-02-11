PICKLEJAR = "data/picklejar/"
N_BINS = 8  # number of bins per particle in the stored event map
LEN_VECTOR = 5  # length of the four-vector plus the charge element
N_PARTICLES = 5  # amount of particles in the datafile, j, b, g, m, e etc
N_PARAMS = N_BINS  # amount of parameters besides the four-vector

SIZE_1D = N_PARAMS + N_PARTICLES * LEN_VECTOR * N_BINS
SIZE_2D = (N_PARTICLES + 1) * LEN_VECTOR * N_BINS
