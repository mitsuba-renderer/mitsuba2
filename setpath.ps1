
# ***************************************************************
# * This script adds Mitsuba to the current path on Windows.
# * It assumes that Mitsuba is either compiled within the 
# * source tree or within a subdirectory named 'build'.
# ***************************************************************

$env:MITSUBA_DIR=Get-Location
$env:PATH=$env:PATH + ";" + $env:MITSUBA_DIR + "\dist;" + $env:MITSUBA_DIR + "\build\dist"
$env:PYTHONPATH=$env:PYTHONPATH + ";" +$env:MITSUBA_DIR + "\dist\python;" + $env:MITSUBA_DIR + "\build\dist\python"
