from math import factorial
from random import choices
from typing import List, Tuple
import numpy as np
import random
import time

def intersect(a, b):
    return len(list(set(a) & set(b))) - 1 