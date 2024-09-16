import os
import zstandard
import json
import jsonlines
import io
import datetime
import mmap
import tqdm
from pathlib import Path


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime,)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


# Modified version of lm_dataformat Archive for single file.
class Archive:
    def __init__(self, file_path, compression_level=3):
        self.file_path = file_path
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.fh = open(self.file_path, "wb")
        self.cctx = zstandard.ZstdCompressor(level=compression_level)
        self.compressor = self.cctx.stream_writer(self.fh)

    def add_data(self, data, meta={}):
        self.compressor.write(
            json.dumps({"text": data, "meta": meta}, default=json_serial).encode(
                "UTF-8"
            )
            + b"\n"
        )

    def commit(self):
        self.compressor.flush(zstandard.FLUSH_FRAME)
        self.fh.flush()
        self.fh.close()


# Modified version of lm_dataformat Reader with self.fh set, allowing peeking for tqdm.
class Reader:
    def __init__(self):
        pass

    def read(self, file, get_meta=False, autojoin_paragraphs=True, para_joiner="\n\n"):
        with open(file, "rb") as fh:
            self.fh = fh
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            for ob in rdr:
                # naive jsonl where each object is just the string itself, with no meta. For legacy compatibility.
                if isinstance(ob, str):
                    assert not get_meta
                    yield ob
                    continue

                text = ob["text"]

                if autojoin_paragraphs and isinstance(text, list):
                    text = para_joiner.join(text)

                if get_meta:
                    yield text, (ob["meta"] if "meta" in ob else {})
                else:
                    yield text


class TextArchive:
    def __init__(self, file_path, mode="rb+"):
        self.file_path = file_path
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if not os.path.exists(file_path):
            Path(file_path).touch()

        self.fh = open(self.file_path, mode)

    def add_data(self, data):
        self.fh.write(data.encode("UTF-8") + b"\n")

    def commit(self):
        self.fh.flush()
        self.fh.close()


class TextReader:
    def __init__(self, file_path):
        self.file_path = file_path

    # Optimized mmap read with infrequent tqdm updates to maintain speed
    # Tested up to 250MB/s.
    def read_tqdm(self, update_frequency=10000):
        current_file_position = 0
        line_counter = 0
        with open(self.file_path, "r") as fh, tqdm.tqdm(
            total=os.path.getsize(self.file_path),
            dynamic_ncols=True,
            unit="byte",
            unit_scale=1,
        ) as progress:
            with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                for line in iter(mmap_obj.readline, b""):
                    line = line.decode("utf-8")
                    line_counter += 1
                    if line_counter == update_frequency:
                        new_file_pos = mmap_obj.tell()
                        bytes_read = new_file_pos - current_file_position
                        current_file_position = new_file_pos
                        progress.update(bytes_read)
                        line_counter = 0
                    yield line[:-1]

    def read_and_tell(self):
        current_file_position = 0
        with open(self.file_path, "r", encoding="utf8") as fh:
            with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                for line in iter(mmap_obj.readline, b""):
                    line = line.decode("utf-8")
                    new_file_pos = mmap_obj.tell()
                    raw_bytes_read = new_file_pos - current_file_position
                    current_file_position = new_file_pos
                    yield line[:-1], raw_bytes_read

    def read(self):
        with open(self.file_path, "r", encoding="utf8") as fh:
            with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                for line in iter(mmap_obj.readline, b""):
                    line = line.decode("utf-8")
                    yield line[:-1]

    def read_slow(self):
        with open(self.file_path, "r", encoding="utf8") as fh:
            while True:
                line = fh.readline()
                if line == -1 or line == "":
                    break
                else:
                    yield line[:-1]


# Optimized for speed. Decompresses the archive in shell before
# using the mmap'd TextReader.
class ZStdTextReader:
    def __init__(self, file):
        self.file = file

    def read_tqdm(self):
        decompressed_file = self.file[:-4]
        print("Decompressing file, please wait...")
        os.system(f"zstd -d {self.file}")  # linux decompress is faster
        reader = TextReader(decompressed_file)
        yield from reader.read_tqdm()
        os.remove(decompressed_file)
