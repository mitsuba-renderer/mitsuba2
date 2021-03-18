#!/bin/bash
set -e

ninja
# mode="scalar_rgb"
# mode="scalar_spectral"
mode="cuda_rgb"

# for d in 0 1 2 3 6; do
# # for d in 1 2; do
#     ./mitsuba -m $mode ../resources/data/scenes/cbox/cbox-rgb.xml -Dspp=1024 -Dintegrator=path -Dmax_depth=$d -o cbox-rgb-d$d-path.exr
#     ./mitsuba -m $mode ../resources/data/scenes/cbox/cbox-rgb.xml -Dspp=4096 -Dintegrator=ptracer -Dmax_depth=$d -o cbox-rgb-d$d-ptracer.exr
# done

d=2
./mitsuba -m $mode ../resources/data/scenes/cbox/cbox-rgb.xml -Dspp=512 -Dintegrator=path -Dmax_depth=$d -o cbox-rgb-d$d-path.exr
./mitsuba -m $mode ../resources/data/scenes/cbox/cbox-rgb.xml -Dspp=1024 -Dintegrator=ptracer -Dmax_depth=$d -o cbox-rgb-d$d-ptracer.exr
