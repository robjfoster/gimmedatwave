import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Generator, List

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

# The record length is now calculated automatically.
_DIGITIZER_FAMILY_RECORD_LENGTH_MAP = {
    DigitizerFamily.X742: 1024,
    DigitizerFamily.X740: 1024,
    DigitizerFamily.X730: 1024,
    DigitizerFamily.X725: 1030
}

_DIGITIZER_FAMILY_RECORD_DTYPE_MAP = {
    DigitizerFamily.X742: np.float32,
    DigitizerFamily.X740: np.uint16,
    DigitizerFamily.X730: np.uint16,
    DigitizerFamily.X725: np.uint16
}

_DIGITIZER_FAMILY_SAMPLE_RATE_MAP = {
    DigitizerFamily.X742: 5000,
    DigitizerFamily.X740: 62.5,
    DigitizerFamily.X730: 500,
    DigitizerFamily.X725: 250
}


@dataclass
class CAENHeader:
    event_size: int
    board_id: int
    pattern: int
    channel_mask: int
    event_counter: int
    trigger_time_tag: int

    def display(self) -> None:
        """
        Prints the header data to the console.
        """
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
    sample_times: np.ndarray
    id: int = 0

    def display(self) -> None:
        """
        Plots the event's record data.
        """
        plt.plot(self.sample_times, self.record)
        plt.show()


class Parser():
    """
    The parser for CAEN WaveDump files. Create a parser object and then use one of the methods
    to read the data.
    Usage: 
    import gimmedatwave as gdw
    parser = gdw.Parser("wave_1.dat", gdw.DigitizerFamily.X742)
    event = parser.get_event(0)
    event.display()
    """

    def __init__(self,
                 file: str,
                 digitizer_family: DigitizerFamily,
                 record_length: Optional[int] = None,
                 record_dtype: Optional[np.dtype] = None,
                 sample_rate: Optional[float] = None
                 ) -> None:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found")
        self.file = file
        self.digitizer_family = digitizer_family
        self.record_dtype = record_dtype or _DIGITIZER_FAMILY_RECORD_DTYPE_MAP[
            digitizer_family]
        self.header_length = _DIGITIZER_FAMILY_HEADER_LENGTH_MAP[digitizer_family]
        self.header_dtype = _DIGITIZER_FAMILY_HEADER_DTYPE_MAP[digitizer_family]
        self.record_length = record_length or self._calc_record_length()
        self.sample_rate = sample_rate or _DIGITIZER_FAMILY_SAMPLE_RATE_MAP[digitizer_family]
        self.sample_times = np.arange(
            self.record_length) / self.sample_rate * 1e6
        self.dtype = self._create_dtype(
            self.header_length, self.header_dtype, self.record_length, self.record_dtype)
        self.n_entries = self._get_entries()
        self.cur_idx = 0

    def _calc_record_length(self) -> int:
        # Read the header from the first event in the file
        header = np.fromfile(
            self.file, dtype=self.header_dtype, count=self.header_length)
        record_length = int((header[0] - (self.header_length *
                                          self.header_dtype().itemsize)) / (self.record_dtype().itemsize))
        return record_length

    def _create_dtype(self,
                      header_size: int,
                      header_dtype: np.dtype,
                      record_size: int,
                      record_dtype: np.dtype
                      ) -> np.dtype:
        return np.dtype([('header', header_dtype, header_size), ('record', record_dtype, record_size)])

    def _get_entries(self) -> int:
        return int(os.path.getsize(self.file) / self.dtype.itemsize)

    def get_event(self, index: int) -> CAENEvent:
        """Get a single event from the binary file.
        The file pointer is incremented by the size of the event as defined by the DigitizerFamily.
        Usage: get_event(0) gets the first event in the file.

        Args:
            index (int): The event number to get.

        Raises:
            IndexError: Raised if the index is beyond the end of the file.

        Returns:
            CAENEvent: The event at the specified index.
        """
        unpacked = np.fromfile(self.file, dtype=self.dtype, count=1,
                               offset=index * self.dtype.itemsize)
        try:
            return CAENEvent(CAENHeader(*unpacked['header'][0]), unpacked['record'][0], self.sample_times, index)
        except IndexError:
            raise IndexError(f"Index {index} beyond end of file")

    def get_all_events(self, start: int = 0) -> List[CAENEvent]:
        """Gets all events in a file and returns as a list of CAENEvents.
        Not recommended for large files since it will load the entire file into memory.
        Usage: get_all_events(start=49) gets all events starting at the 50th event.

        Args:
            start (int, optional): The event to start reading from. Defaults to 0.

        Returns:
            list[CAENEvent]: The list of CAENEvents.
        """
        unpacked = np.fromfile(
            self.file, dtype=self.dtype, count=-1, offset=start)
        events = []
        for i, event in enumerate(unpacked):
            events.append(CAENEvent(CAENHeader(
                *event['header']), event['record'], self.sample_times, i))
        return events

    def read_dat(self, start: int = 0, stop: Optional[int] = None, step: int = 1) -> Generator[CAENEvent, None, None]:
        """Generator that yields CAENEvents from a binary file.
        Usage: for event in read_dat(start=5, step=100): event.display()

        Args:
            start (int, optional): The event number to begin reading from. Defaults to 0.
            stop (Optional[int], optional): The event number to stop reading at. Defaults to None.
            step (int, optional): How many events to step each iteration. Defaults to 1.

        Raises:
            IndexError: Raised if the index is beyond the end of the file.

        Yields:
            CAENEvent: The CAENEvent at the current index.
        """
        index = start
        if stop is not None and stop > self.n_entries:
            raise IndexError(
                f"Stop index {stop} beyond end of file ({self.n_entries})")
        end = stop or self.n_entries-1
        with open(self.file, 'rb') as f:
            while index < end:
                unpacked = np.fromfile(f, dtype=self.dtype, count=1)
                yield CAENEvent(CAENHeader(*unpacked['header'][0]), unpacked['record'][0], self.sample_times, index)
                index += step

    def read_next(self) -> CAENEvent:
        """Read the next event from the current position in the file.

        Raises:
            IndexError: Raised if the index is beyond the end of the file.

        Returns:
            CAENEvent: The CAENEvent at the next position in the file.
        """
        if self.cur_idx >= self.n_entries:
            raise IndexError(f"Index {self.cur_idx} beyond end of file")
        event = self.get_event(self.cur_idx)
        self.cur_idx += 1
        return event
