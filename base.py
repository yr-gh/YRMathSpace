# -*- coding: utf-8 -*-
"""
Odds and ends.

@author: Rui Yang
"""

import sys
import logging

# Configure log module
logging.basicConfig(
    format = '[%(asctime)s | %(process)d-%(thread)d | %(levelname)s] %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    level = logging.INFO, stream = sys.stdout)
log = logging.getLogger(__name__)
