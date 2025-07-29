#!/bin/bash

# sudo apt install imagemagick ghostscript
# sudo mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.disabled  # Disable security policy for PDF

for f in *.pdf; do convert -density 600 -antialias "$f" "${f%.*}.png"; done
