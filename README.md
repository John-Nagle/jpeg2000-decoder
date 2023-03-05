# jpeg2000-decoder
Decodes JPEG 2000 images in a sandbox, for safety.

# IN PROGRESS

# Overview

OpenJPEG is a JPEG 2000 decoder written in C.
Because of a long history of buffer overflows and security vulnerabilities,
it can't be trusted in the same address space as Rust code. So, here it is
being compiled to WASM and run in a sandbox.

This project generates both a library and an example executable.
The example decodes JPEG 2000 images.

This is intended primarily for Second Life / Open Simulator content.

# Example

Usage: jpeg2000-decoder -i INFILE -o OUTFILE

## Options

* **-i INFILE** Input file, JPEG 2000. May be a URL or a file.
* **--input INFILE** 

* **-o OUTFILE** Output file. Only **.png** is currently supported.
* **--output OUTFILE**

* **--maxsize PIXELS** Maximum dimension of output image. Image will be fetched and reduced accordingly.

* **-v** Verbose 
* **--verbose**
