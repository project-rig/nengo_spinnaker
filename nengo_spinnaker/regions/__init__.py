"""A system for defining data which is placed into regions of memory.

A region consists of a block of data (which may be empty) that is to be stored
in the SDRAM or the DTCM of a SpiNNaker core.  The data may be "partitioned" so
that only the data that is required by the core is written out.

Regions are able to report their size, represent data and write portions of
their data out to files as necessary.
"""

from .list import ListRegion
from .matrix import MatrixPartitioning, MatrixRegion
from .keyspaces import KeyspacesRegion, KeyField, MaskField
from .profiler import Profiler
from .region import Region
from .recording import (RecordingRegion, WordRecordingRegion,
                        SpikeRecordingRegion, VoltageRecordingRegion,
                        EncoderRecordingRegion)
from . import utils
