import csv
import os
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, arg, save_path):
        self.log_path = save_path + "/log.txt"
        self.lines = []
        if os.path.exists(self.log_path) is False:
            with open(self.log_path, "a+", encoding="utf8") as f:
                for key, val in sorted(arg.__dict__.items()):
                    val = str(val)
                    log_string = key
                    log_string += "." * (80 - len(key) - len(val))
                    log_string += (val + '\n')
                    f.write(log_string)
                f.write("args end\n")

    def will_write(self, line, is_print=True):
        if is_print:
            print(line)
        self.lines.append(line)

    def flush(self):
        with open(self.log_path, "a", encoding="utf8") as f:
            for line in self.lines:
                f.write(line + "\n")
        self.lines = []

    def write(self, line):
        self.will_write(line)
        self.flush()
