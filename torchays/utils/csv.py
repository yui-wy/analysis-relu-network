from typing import Any, List


class CSV(object):
    def __init__(self, save_path: str) -> None:
        self.save_path = save_path
        self.csv_buf: List[str] = list()
        self.with_header = False

    def save(self):
        csv_buf = "\r\n".join(self.csv_buf)
        with open(self.save_path, 'w') as w:
            w.write(csv_buf)
            w.close()

    def set_header(self, tag, header: List[Any]):
        header_buf = self._row(tag, header)
        if len(self.csv_buf) == 0:
            self.csv_buf.append(header_buf)
        elif not self.with_header:
            self.csv_buf.insert(0, header_buf)
        else:
            self.csv_buf[0] = header_buf
        self.with_header = True
        return self

    def add_row(self, tag, content: List[Any]):
        self.csv_buf.append(self._row(tag, content))
        return self

    def _row(self, tag, content: List[Any]) -> str:
        content_buf = ",".join(list(map(str, content)))
        return f"{tag},{content_buf}"
