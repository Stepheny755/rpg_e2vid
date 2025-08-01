import pandas as pd
import zipfile
from os.path import splitext
import numpy as np
import tqdm
from .timers import Timer
import h5py


class FixedSizeEventReader:
    """
    Reads events from a '.txt', '.zip', or '.hdf5' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0, 
                 x_min=0, y_min=0, x_max=None, y_max=None):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip', '.hdf5', '.h5'])
        
        self.is_hdf5_file = (file_extension in ['.hdf5', '.h5'])
        self.num_events = num_events
        self.start_index = start_index
        self.path_to_event_file = path_to_event_file
        
        # Coordinate bounds for filtering (optional)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        
        if self.is_hdf5_file:
            # For HDF5, just open the file and get dataset info
            self._init_hdf5()
            self.current_event_index = start_index
            self._hdf5_buffer = []  # Buffer for filtered events
            self._buffer_start_idx = 0  # Starting index of current buffer in file
        else:
            # For text/zip files, use pandas chunked reading (original implementation)
            self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                        names=['t', 'x', 'y', 'pol'],
                                        dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                        engine='c',
                                        skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def _init_hdf5(self):
        """Initialize HDF5 file reading without loading all data"""
        print("Initializing HDF5 reader...")
        self._hdf5_file = h5py.File(self.path_to_event_file, 'r')
        self._hdf5_dataset = self._hdf5_file['CD/events']
        self._total_events = len(self._hdf5_dataset)
        print(f"HDF5 file contains {self._total_events} total events")
        
        # Determine chunk size for efficient reading (balance memory vs I/O)
        # Read larger chunks to minimize I/O overhead, but not too large to consume memory
        self._chunk_size = max(self.num_events * 2, 50000)

    def _read_hdf5_chunk(self, start_idx, chunk_size):
        """Read a chunk of events from HDF5 file and apply filtering"""
        end_idx = min(start_idx + chunk_size, self._total_events)
        if start_idx >= self._total_events:
            return np.array([])
        
        # Read chunk from HDF5
        raw_chunk = self._hdf5_dataset[start_idx:end_idx]
        
        # Apply filtering and format conversion
        filtered_events = []
        for e in raw_chunk:
            x, y, p, ts = e
            
            # Apply coordinate filtering if specified
            if self.x_max is not None and self.y_max is not None:
                if x < self.x_min or y < self.y_min or x >= self.x_max or y >= self.y_max:
                    continue
            
            # Store as [timestamp_in_seconds, x_adjusted, y_adjusted, polarity]
            filtered_events.append([
                ts,
                x - self.x_min,  # Adjust x coordinate
                y - self.y_min,  # Adjust y coordinate
                p
            ])
        
        if filtered_events:
            events_array = np.array(filtered_events, dtype=np.float64)
            # Convert x, y, pol columns to int16 to match pandas reader behavior
            events_array[:, 1] = events_array[:, 1].astype(np.int16)  # x
            events_array[:, 2] = events_array[:, 2].astype(np.int16)  # y
            events_array[:, 3] = events_array[:, 3].astype(np.int16)  # pol
            return events_array
        else:
            return np.array([])

    def _ensure_buffer_has_events(self, required_events):
        """Ensure buffer has at least required_events available"""
        while len(self._hdf5_buffer) < required_events:
            # Calculate where to read from in the file
            file_idx = self._buffer_start_idx + len(self._hdf5_buffer)
            
            if file_idx >= self._total_events:
                # No more events in file
                break
            
            # Read next chunk
            chunk = self._read_hdf5_chunk(file_idx, self._chunk_size)
            
            if len(chunk) == 0:
                # No events after filtering in this chunk, try next chunk
                self._buffer_start_idx = file_idx + self._chunk_size
                continue
            
            # Add to buffer
            if len(self._hdf5_buffer) == 0:
                self._hdf5_buffer = chunk.tolist()
            else:
                self._hdf5_buffer.extend(chunk.tolist())

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_hdf5_file:
            return self._next_hdf5_window()
        else:
            return self._next_text_window()

    def _next_hdf5_window(self):
        """Get next fixed-size event window from HDF5 data"""
        # Skip to start_index if this is the first call
        if hasattr(self, '_first_call'):
            pass
        else:
            self._first_call = False
            # Fast-forward to start_index by updating buffer position
            if self.start_index > 0:
                self._buffer_start_idx = self.start_index
        
        # Ensure we have enough events in buffer
        self._ensure_buffer_has_events(self.num_events)
        
        # Check if we have enough events for a full window
        if len(self._hdf5_buffer) < self.num_events:
            # Clean up and raise StopIteration
            self._hdf5_file.close()
            raise StopIteration
        
        # Extract the event window
        event_window = np.array(self._hdf5_buffer[:self.num_events], dtype=np.float64)
        
        # Remove used events from buffer
        self._hdf5_buffer = self._hdf5_buffer[self.num_events:]
        self._buffer_start_idx += self.num_events
        
        # Ensure proper data types
        event_window[:, 1] = event_window[:, 1].astype(np.int16)  # x
        event_window[:, 2] = event_window[:, 2].astype(np.int16)  # y
        event_window[:, 3] = event_window[:, 3].astype(np.int16)  # pol
        
        return event_window

    def _next_text_window(self):
        """Get next fixed-size event window from text/zip file (original implementation)"""
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
        return event_window

    def __del__(self):
        """Clean up HDF5 file handle"""
        if self.is_hdf5_file and hasattr(self, '_hdf5_file'):
            try:
                self._hdf5_file.close()
            except:
                pass


class FixedDurationEventReader:
    """
    Reads events from a '.txt', '.zip', or '.hdf5' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader for txt/zip files.
              The reason is that the latter can use Pandas' very efficient chunk-based reading scheme implemented in C.
              However, HDF5 files are read efficiently using h5py.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0, 
                 x_min=0, y_min=0, x_max=None, y_max=None):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip', '.hdf5', '.h5'])
        
        self.is_zip_file = (file_extension == '.zip')
        self.is_hdf5_file = (file_extension in ['.hdf5', '.h5'])
        self.path_to_event_file = path_to_event_file
        
        # Coordinate bounds for filtering (optional)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        
        self.duration_s = duration_ms / 1000.0
        self.last_stamp = None
        self.start_index = start_index
        
        if self.is_hdf5_file:
            # For HDF5, we'll load all events at once and then iterate through them
            self._load_hdf5_events()
            self.current_event_index = start_index
        else:
            # Initialize file readers for txt/zip files
            if self.is_zip_file:  # '.zip'
                self.zip_file = zipfile.ZipFile(path_to_event_file)
                files_in_archive = self.zip_file.namelist()
                assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
                self.event_file = self.zip_file.open(files_in_archive[0], 'r')
            else:
                self.event_file = open(path_to_event_file, 'r')

            # ignore header + the first start_index lines
            for i in range(1 + start_index):
                self.event_file.readline()

    def _load_hdf5_events(self):
        """Load all events from HDF5 file into memory"""
        print("Loading HDF5 events...")
        with h5py.File(self.path_to_event_file, 'r') as f:
            dataset = f['CD/events']
            
            # Pre-allocate lists for better performance
            self.events = []
            
            for e in dataset:
                x, y, p, ts = e
                
                # Apply coordinate filtering if specified
                if self.x_max is not None and self.y_max is not None:
                    if x < self.x_min or y < self.y_min or x >= self.x_max or y >= self.y_max:
                        continue
                
                # Store as [timestamp_in_seconds, x, y, polarity]
                # Adjust coordinates by offset and convert timestamp to seconds
                self.events.append([
                    ts,
                    x - self.x_min,
                    y - self.y_min,
                    p
                ])
            
            # Convert to numpy array for efficient indexing
            self.events = np.array(self.events)
            print(f"Loaded {len(self.events)} events from HDF5 file")

    def __iter__(self):
        return self

    def __del__(self):
        if hasattr(self, 'zip_file') and self.is_zip_file:
            self.zip_file.close()
        
        if hasattr(self, 'event_file') and not self.is_hdf5_file:
            self.event_file.close()

    def __next__(self):
        if self.is_hdf5_file:
            return self._next_hdf5_window()
        else:
            return self._next_text_window()

    def _next_hdf5_window(self):
        """Get next event window from HDF5 data"""
        if self.current_event_index >= len(self.events):
            raise StopIteration
        
        event_list = []
        start_index = self.current_event_index
        
        # Get the timestamp of the first event in this window
        if self.last_stamp is None:
            self.last_stamp = self.events[start_index][0]  # timestamp is first element
        
        # Collect events within the duration window
        while self.current_event_index < len(self.events):
            event = self.events[self.current_event_index]
            t = event[0]  # timestamp
            
            if t > self.last_stamp + self.duration_s:
                self.last_stamp = t
                break
            
            # Reorder to [t, x, y, pol] format to match original class
            event_list.append([t, event[1], event[2], event[3]])
            self.current_event_index += 1
        
        if not event_list:
            raise StopIteration
        
        return np.array(event_list)

    def _next_text_window(self):
        """Get next event window from text/zip file (original implementation)"""
        event_list = []
        for line in self.event_file:
            if self.is_zip_file:
                line = line.decode("utf-8")
            t, x, y, pol = line.split(' ')
            t, x, y, pol = float(t), int(x), int(y), int(pol)
            event_list.append([t, x, y, pol])
            if self.last_stamp is None:
                self.last_stamp = t
            if t > self.last_stamp + self.duration_s:
                self.last_stamp = t
                event_window = np.array(event_list)
                return event_window

        raise StopIteration