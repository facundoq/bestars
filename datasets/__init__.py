
from . import liu,mohr_smith,mcswain,hou


def common_columns(datasets:[str]):
    def lequal(l1,l2):
        l1=list(sorted((l1)))
        return len(l1) == len(l2) and sorted(l1)==sorted(l2)

    if lequal(datasets,["liu2017","mohr"]):
        return [ 'umag', 'gmag', 'rmag',
                 'imag', 'Hamag', 'Jmag', 'Hmag', 'Kmag',
                 ]
    else:
        raise ValueError(f"No entries  for dataset combination: {datasets}")
