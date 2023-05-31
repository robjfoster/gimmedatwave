import os
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

HEADER_SIZE = 6
HEADER_DTYPE = np.uint32


class DigitizerFamily(Enum):
    X742 = 1
    X740 = 2
    X730 = 3
    X725 = 4


_DIGITIZER_FAMILY_HEADER_LENGTH_MAP = {
    DigitizerFamily.X742: HEADER_SIZE,
    DigitizerFamily.X740: HEADER_SIZE,
    DigitizerFamily.X730: HEADER_SIZE,
    DigitizerFamily.X725: HEADER_SIZE
}

_DIGITIZER_FAMILY_HEADER_DTYPE_MAP = {
    DigitizerFamily.X742: HEADER_DTYPE,
    DigitizerFamily.X740: HEADER_DTYPE,
    DigitizerFamily.X730: HEADER_DTYPE,
    DigitizerFamily.X725: HEADER_DTYPE
}

_DIGITIZER_FAMILY_RECORD_LENGTH_MAP = {
    DigitizerFamily.X742: 1024,
    DigitizerFamily.X740: 1024,
    DigitizerFamily.X730: 1024,
    DigitizerFamily.X725: 1024
}

_DIGITIZER_FAMILY_RECORD_DTYPE_MAP = {
    DigitizerFamily.X742: np.float32,
    DigitizerFamily.X740: np.uint16,
    DigitizerFamily.X730: np.uint16,
    DigitizerFamily.X725: np.uint16
}


@dataclass
class CAENHeader:
    event_size: int
    board_id: int
    pattern: int
    channel_mask: int
    event_counter: int
    trigger_time_tag: int

    def display(self):
        print(f"Event size: {self.event_size}")
        print(f"Board ID: {self.board_id}")
        print(f"Pattern: {self.pattern}")
        print(f"Channel mask: {self.channel_mask}")
        print(f"Event counter: {self.event_counter}")
        print(f"Trigger time tag: {self.trigger_time_tag}")


@dataclass
class CAENEvent:
    header: CAENHeader
    record: np.ndarray
    id: int = 0

    def display(self):
        plt.plot(self.record)
        plt.show()


class Parser():

    def __init__(self, file, digitizer_family, record_length=None, record_dtype=None):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found")
        self.file = file
        self.digitizer_family = digitizer_family
        self.record_length = record_length or _DIGITIZER_FAMILY_RECORD_LENGTH_MAP[
            digitizer_family]
        self.record_dtype = record_dtype or _DIGITIZER_FAMILY_RECORD_DTYPE_MAP[
            digitizer_family]
        self.header_length = _DIGITIZER_FAMILY_HEADER_LENGTH_MAP[digitizer_family]
        self.header_dtype = _DIGITIZER_FAMILY_HEADER_DTYPE_MAP[digitizer_family]
        self.dtype = self._create_dtype(
            self.header_length, self.header_dtype, self.record_length, self.record_dtype)
        self.n_entries = self._get_entries()
        self.cur_idx = 0

    def _create_dtype(self, header_size, header_dtype, record_size, record_dtype):
        return np.dtype([('header', header_dtype, header_size), ('record', record_dtype, record_size)])

    def _get_entries(self):
        return int(os.path.getsize(self.file) / self.dtype.itemsize)

    def get_event(self, index):
        unpacked = np.fromfile(self.file, dtype=self.dtype, count=1,
                               offset=index * self.dtype.itemsize)
        try:
            return CAENEvent(CAENHeader(*unpacked['header'][0]), unpacked['record'][0], index)
        except IndexError:
            raise IndexError(f"Index {index} beyond end of file")

    def get_all_events(self, start=0):
        unpacked = np.fromfile(
            self.file, dtype=self.dtype, count=-1, offset=start)
        events = []
        for i, event in enumerate(unpacked):
            events.append(CAENEvent(CAENHeader(
                *event['header']), event['record']), i)
        return events

    def read_dat(self, start=0, stop=None, step=1):
        index = start
        if stop is not None and stop > self.n_entries:
            raise IndexError(
                f"Stop index {stop} beyond end of file ({self.n_entries})")
        end = stop or self.n_entries-1
        with open(self.file, 'rb') as f:
            while index < end:
                unpacked = np.fromfile(f, dtype=self.dtype, count=1)
                yield CAENEvent(CAENHeader(*unpacked['header'][0]), unpacked['record'][0], index)
                index += step

    def read_next(self):
        if self.cur_idx >= self.n_entries:
            raise IndexError(f"Index {self.cur_idx} beyond end of file")
        event = self.get_event(self.cur_idx)
        self.cur_idx += 1
        return event


if __name__ == "__main__":
    parser = Parser("wave_20.dat", DigitizerFamily.X742)
    events = []
    for event in parser.read_dat(start=5, step=100):
        events.append(event)
    breakpoint()
