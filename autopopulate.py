#!/usr/bin/env python

import os
import subprocess


for root, subdir, filename in os.walk("classes"):
    if filename:
        with open(os.path.join(root, filename[0]), "r") as f:
            lines = [i.strip("\n") for i in f.readlines()]
            lines_seen = []

            for line in lines:
                if not lines_seen or line not in lines_seen:
                    lines_seen.append(line)

         #   files = ""
          #  for i in lines_seen:
           #     files += " %s" % i

#            print("python main.py -c %s %s" % (root.split("/", 1)[1:][0], files))
            #os.system("python main.py -c %s %s" % (root.split("/", 1)[1:][0], files))
            for i in lines_seen:
                os.system("python main.py -c %s %s" % (root.split("/", 1)[1:][0], i))
