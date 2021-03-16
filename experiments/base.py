import os
from pathlib import Path
from datetime import datetime
import abc
import datasets
import sys
from pathlib import Path
current=Path(__file__)

plots_folder = current.parent / "../plots/"
class Experiment(abc.ABC):

    def __init__(self):
        self.plot_folderpath = plots_folder / self.id()
        self.plot_folderpath.mkdir(exist_ok=True, parents=True)
        with open(self.plot_folderpath / "description.txt", "w") as f:
            f.write(self.description())
        self.venv = Path(".")

    def id(self):
        return self.__class__.__name__

    def __call__(self, force=False, venv=".", *args, **kwargs):
        stars = "*" * 15
        strf_format = "%Y/%m/%d %H:%M:%S"
        dt_started = datetime.now()
        dt_started_string = dt_started.strftime(strf_format)
        if not self.has_finished() or force:
            self.mark_as_unfinished()
            print(f"[{dt_started_string}] {stars} Running experiment {self.id()}  {stars}")
            self.run()

            # time elapsed and finished
            dt_finished = datetime.now()
            dt_finished_string = dt_finished.strftime(strf_format)
            elapsed = dt_finished - dt_started
            print(f"[{dt_finished_string}] {stars} Finished experiment {self.id()}  ({elapsed} elapsed) {stars}")
            self.mark_as_finished()
        else:
            print(f"[{dt_started_string}] {stars}Experiment {self.id()} already finished, skipping. {stars}")

    def has_finished(self):
        return self.finished_filepath().exists()

    def finished_filepath(self):
        return self.plot_folderpath / "finished"

    def mark_as_finished(self):
        self.finished_filepath().touch(exist_ok=True)

    def mark_as_unfinished(self):
        f = self.finished_filepath()
        if f.exists():
            f.unlink()

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass


    def print_date(self, message):
        strf_format = "%Y/%m/%d %H:%M:%S"
        dt = datetime.now()
        dt_string = dt.strftime(strf_format)
        message = f"[{dt_string}] *** {message}"
        print(message)
