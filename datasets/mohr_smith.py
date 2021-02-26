# This codes loads a labelled subset of the Mohr-Smith 2017 dataset containing photometric data of
# Be stars and other types of OB stars
# More info at:
# VizieR Online Data Catalog: New OB star candidates in Carina Arm (Mohr-Smith+, 2015):
#   url: https://ui.adsabs.harvard.edu/abs/2017yCat..74503855M/abstract
#  Reddening-free Q indices to identify Be star candidates (arXiv):
#  url: https://arxiv.org/abs/2009.06017

coefficients = {'umag': 4.39,
                'gmag':  3.30,
                'rmag':  2.31,
                'imag':  1.71,
                'Hamag': 2.14,  # valor interpolado
                'Jmag':  0.72,
                'Hmag':  0.46,
                'Kmag':  0.306,
                'W1mag': 0.18,
                'W2mag': 0.16}

system =        {'umag': 'VPHAS',
                'gmag': 'VPHAS',
                'rmag': 'VPHAS',
                'imag': 'VPHAS',
                'Hamag':'VPHAS',
                'Jmag': '2MASS',
                'Hmag': '2MASS',
                'W1mag':'WISE',
                'W2mag':'WISE',
                }

filename =  "Mohr-Smith_2017.csv"

