# jpeg2000-decoder
Decodes JPEG 2000 images using OpenJPEG 2000.

# IN PROGRESS

# Overview

This project generates both a library and an executable.
The executable decodes JPEG 2000 images by calling the library.
The library, "jpeg2000-decoder" does image fetching and decoding.

This is intended primarily for Second Life / Open Simulator content.

# Executable

Usage: jpeg2000-decoder -i INFILE -o OUTFILE

## Options

* **-i INFILE** Input file, JPEG 2000. May be a URL or a file.
* **--input INFILE** 

* **-o OUTFILE** Output file. Only **.png** is currently supported.
* **--output OUTFILE**

* **--maxsize PIXELS** Maximum dimension of output image. Image will be fetched and reduced accordingly.

* **--user-agent USERAGENT** HTTP user agent to use when making requests.

* **-v** Verbose 
* **--verbose**

