import pandas as pd
import zipfile
from os.path import splitext
import numpy as np
import tqdm
from .timers import Timer
import h5py
import pandas as pd


class FixedSizeEventReader:
    """
    Reads events from a '.txt', '.zip', or '.hdf5' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(
        self,
        path_to_event_file,
        num_events=10000,
        start_index=0,
        x_min=0,
        y_min=0,
        x_max=None,
        y_max=None,
    ):
        print("Will use fixed size event windows with {} events".format(num_events))
        print("Output frame rate: variable")

        file_extension = splitext(path_to_event_file)[1]
        assert file_extension in [".txt", ".zip", ".hdf5", ".h5"]

        self.is_hdf5_file = file_extension in [".hdf5", ".h5"]
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
            self.iterator = pd.read_csv(
                path_to_event_file,
                delim_whitespace=True,
                header=None,
                names=["t", "x", "y", "pol"],
                dtype={"t": np.float64, "x": np.int16, "y": np.int16, "pol": np.int16},
                engine="c",
                skiprows=start_index + 1,
                chunksize=num_events,
                nrows=None,
                memory_map=True,
            )

    def _init_hdf5(self):
        """Initialize HDF5 file reading without loading all data"""
        print("Initializing HDF5 reader...")
        self._hdf5_file = h5py.File(self.path_to_event_file, "r")
        self._hdf5_dataset = self._hdf5_file["CD/events"]
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
                if (
                    x < self.x_min
                    or y < self.y_min
                    or x >= self.x_max
                    or y >= self.y_max
                ):
                    continue

            # Store as [timestamp_in_seconds, x_adjusted, y_adjusted, polarity]
            filtered_events.append(
                [
                    ts,
                    x - self.x_min,  # Adjust x coordinate
                    y - self.y_min,  # Adjust y coordinate
                    p,
                ]
            )

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
        if hasattr(self, "_first_call"):
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
        event_window = np.array(self._hdf5_buffer[: self.num_events], dtype=np.float64)

        # Remove used events from buffer
        self._hdf5_buffer = self._hdf5_buffer[self.num_events :]
        self._buffer_start_idx += self.num_events

        # Ensure proper data types
        event_window[:, 1] = event_window[:, 1].astype(np.int16)  # x
        event_window[:, 2] = event_window[:, 2].astype(np.int16)  # y
        event_window[:, 3] = event_window[:, 3].astype(np.int16)  # pol

        return event_window

    def _next_text_window(self):
        """Get next fixed-size event window from text/zip file (original implementation)"""
        with Timer("Reading event window from file"):
            event_window = self.iterator.__next__().values
        return event_window

    def __del__(self):
        """Clean up HDF5 file handle"""
        if self.is_hdf5_file and hasattr(self, "_hdf5_file"):
            try:
                self._hdf5_file.close()
            except:
                pass


class FixedDurationEventReader:
    """
    Reads events from a '.txt', '.zip', or '.hdf5' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: HDF5 files are now read efficiently using pandas DataFrame approach.
              Text/zip files maintain the original line-by-line reading for memory efficiency.
    """

    def __init__(
        self,
        path_to_event_file,
        duration_ms=50.0,
        start_index=0,
        x_min=0,
        y_min=0,
        x_max=None,
        y_max=None,
    ):
        print(
            "Will use fixed duration event windows of size {:.2f} ms".format(
                duration_ms
            )
        )
        print("Output frame rate: {:.1f} Hz".format(1000.0 / duration_ms))

        file_extension = splitext(path_to_event_file)[1]
        assert file_extension in [".txt", ".zip", ".hdf5", ".h5"]

        self.is_zip_file = file_extension == ".zip"
        self.is_hdf5_file = file_extension in [".hdf5", ".h5"]
        self.path_to_event_file = path_to_event_file

        # Coordinate bounds for filtering (optional)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.duration_s = duration_ms / 1000.0
        self.duration_us = duration_ms * 1000.0  # For HDF5 microsecond timestamps
        self.last_stamp = None
        self.start_index = start_index

        if self.is_hdf5_file:
            # Use efficient pandas-based loading for HDF5
            self._load_hdf5_events_efficient2()
            self.current_index = 0
            self.timestamps_in_seconds = None
        else:
            # Initialize file readers for txt/zip files
            if self.is_zip_file:  # '.zip'
                self.zip_file = zipfile.ZipFile(path_to_event_file)
                files_in_archive = self.zip_file.namelist()
                assert (
                    len(files_in_archive) == 1
                )  # make sure there is only one text file in the archive
                self.event_file = self.zip_file.open(files_in_archive[0], "r")
            else:
                self.event_file = open(path_to_event_file, "r")

            # ignore header + the first start_index lines
            for i in range(1 + start_index):
                self.event_file.readline()

    def _load_hdf5_events_efficient(self):
        """Load HDF5 events efficiently with random subsampling and optional ROI filtering"""
        with h5py.File(self.path_to_event_file, "r") as f:
            if "CD/events" not in f:
                raise ValueError("Dataset 'CD/events' not found in HDF5 file")

            events_dataset = f["CD/events"]
            total = len(events_dataset)
            print(f"Total events in file: {total}")

            events = events_dataset[5 * total // 6 :]
            print("Finished reading raw event data into structured array")

        # Convert to DataFrame directly (no field-by-field split)
        df = pd.DataFrame.from_records(events)

        # Rename to standard column names
        df.rename(columns={"t": "timestamp", "p": "polarity"}, inplace=True)

        # Coordinate filtering (if requested)
        if self.x_max is not None and self.y_max is not None:
            mask = (
                (df["x"] >= self.x_min)
                & (df["x"] < self.x_max)
                & (df["y"] >= self.y_min)
                & (df["y"] < self.y_max)
            )
            df = df[mask].reset_index(drop=True)
            print(f"After coordinate filtering: {len(df)} events")

        # Offset coordinates
        df["x"] -= self.x_min
        df["y"] -= self.y_min

        # Apply start index
        if self.start_index > 0:
            df = df.iloc[self.start_index :].reset_index(drop=True)
            print(f"After start_index filtering: {len(df)} events")

        # Sort by time for consistent iteration
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.df = df
        print(f"Loaded {len(self.df)} events into memory.")
        if len(df) > 0:
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    def _load_hdf5_events_efficient2(self):
        """Load HDF5 events efficiently with random subsampling and optional ROI filtering"""
        df = pd.read_hdf(self.path_to_event_file, key="CD/events")

        # Rename to standard column names
        df.rename(columns={"t": "timestamp", "p": "polarity"}, inplace=True)

        # Coordinate filtering (if requested)
        if self.x_max is not None and self.y_max is not None:
            mask = (
                (df["x"] >= self.x_min)
                & (df["x"] < self.x_max)
                & (df["y"] >= self.y_min)
                & (df["y"] < self.y_max)
            )
            df = df[mask].reset_index(drop=True)
            print(f"After coordinate filtering: {len(df)} events")

        # Offset coordinates
        df["x"] -= self.x_min
        df["y"] -= self.y_min

        # Apply start index
        if self.start_index > 0:
            df = df.iloc[self.start_index :].reset_index(drop=True)
            print(f"After start_index filtering: {len(df)} events")

        # Sort by time for consistent iteration
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.df = df
        print(f"Loaded {len(self.df)} events into memory.")
        if len(df) > 0:
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    def _parse_events_structure(self, all_events):
        """Parse the structure of events data (structured vs regular array)."""
        if hasattr(all_events.dtype, "names") and all_events.dtype.names:
            # Structured array with named fields
            print(f"Event fields: {all_events.dtype.names}")
            return {field: all_events[field] for field in all_events.dtype.names}
        else:
            # Regular array - assume columns are [x, y, p, t] or [t, x, y, p]
            print("Events are in regular array format")
            if all_events.shape[1] >= 4:
                # Try to detect timestamp column (usually much larger values)
                col_means = np.mean(all_events, axis=0)
                timestamp_col = np.argmax(
                    col_means
                )  # Timestamp usually has largest values

                if timestamp_col == 0:  # [t, x, y, p]
                    return {
                        "t": all_events[:, 0],
                        "x": all_events[:, 1],
                        "y": all_events[:, 2],
                        "p": all_events[:, 3],
                    }
                else:  # [x, y, p, t]
                    return {
                        "x": all_events[:, 0],
                        "y": all_events[:, 1],
                        "p": all_events[:, 2],
                        "t": all_events[:, 3],
                    }
            else:
                raise ValueError(f"Unexpected event array shape: {all_events.shape}")

    def _create_standardized_dataframe(self, df_dict):
        """Create DataFrame with standardized column names."""
        field_mapping = {
            "t": "timestamp",
            "time": "timestamp",
            "ts": "timestamp",
            "x": "x",
            "y": "y",
            "p": "polarity",
            "pol": "polarity",
            "polarity": "polarity",
        }

        standardized_dict = {}
        for field, data in df_dict.items():
            standard_name = field_mapping.get(field, field)
            standardized_dict[standard_name] = data

        df = pd.DataFrame(standardized_dict)

        # Validate required columns
        if "timestamp" not in df.columns:
            raise ValueError("No timestamp column found in event data")
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError("No x,y coordinate columns found in event data")

        return df

    def __iter__(self):
        return self

    def __del__(self):
        if hasattr(self, "zip_file") and self.is_zip_file:
            self.zip_file.close()

        if hasattr(self, "event_file") and not self.is_hdf5_file:
            self.event_file.close()

    def __next__(self):
        if self.is_hdf5_file:
            return self._next_hdf5_window_efficient()
        else:
            return self._next_text_window()

    def _next_hdf5_window_efficient(self):
        """Get next event window from HDF5 data using efficient pandas operations"""
        if len(self.df) == 0:
            raise StopIteration

        # Initialize last_stamp on first call - HDF5 timestamps are in microseconds
        if self.last_stamp is None:
            # Convert first timestamp from microseconds to seconds
            self.last_stamp = self.df["timestamp"].iloc[0] / 1e6
            # Pre-convert all timestamps to seconds for efficiency
            self.timestamps_in_seconds = self.df["timestamp"].values / 1e6
            self.current_index = 0

        if self.current_index >= len(self.df):
            raise StopIteration

        # Collect events within the duration window
        event_list = []
        window_end = self.last_stamp + self.duration_s

        while self.current_index < len(self.df):
            event_timestamp = self.timestamps_in_seconds[self.current_index]

            if event_timestamp > window_end:
                # This event starts the next window
                self.last_stamp = event_timestamp
                break

            # Add event to current window
            row = self.df.iloc[self.current_index]
            x_coord = row["x"]
            y_coord = row["y"]
            polarity = row.get("polarity", 1)  # Default to 1 if no polarity

            event_list.append([event_timestamp, x_coord, y_coord, polarity])
            self.current_index += 1

        if not event_list:
            raise StopIteration

        return np.array(event_list)

    def _next_text_window(self):
        """Get next event window from text/zip file (original implementation)"""
        event_list = []
        for line in self.event_file:
            if self.is_zip_file:
                line = line.decode("utf-8")
            t, x, y, pol = line.split(" ")
            t, x, y, pol = float(t), int(x), int(y), int(pol)

            # Apply coordinate filtering
            if self.x_max is not None and self.y_max is not None:
                if (
                    x < self.x_min
                    or y < self.y_min
                    or x >= self.x_max
                    or y >= self.y_max
                ):
                    continue

            # Adjust coordinates by offset
            x_adj = x - self.x_min
            y_adj = y - self.y_min

            event_list.append([t, x_adj, y_adj, pol])

            if self.last_stamp is None:
                self.last_stamp = t
            if t > self.last_stamp + self.duration_s:
                self.last_stamp = t
                event_window = np.array(event_list)
                return event_window

        raise StopIteration
