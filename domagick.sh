#!/bin/bash

magick boxer-starter.jpg \
-monitor \
-unsharp 5 \
-posterize 10 \
-unsharp 5 \
-colors 6 \
-unsharp 5 \
-colors 6 \
-modulate 100 \
-noise 5 \
-median 5 \
-normalize \
-morphology open diamond:5 \
identify -format "%k"\
img.jpg
