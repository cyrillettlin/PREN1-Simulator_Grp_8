import cv2

images = []
for i in range(1, 7):
    path = f"Images/real_puzzle/puzzle_{i}.jpg"
    img = cv2.imread(path)
    if img is not None:
        images.append(img)

stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    cv2.imwrite('result.jpg', result)
    print("Bild erstellt")
else:
    print(f"Fehler: Status-Code {status}")

from __future__ import print_function

import numpy as np
import cv2 as cv

import argparse
import sys

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
parser.add_argument('--mode',
                    type=int, choices=modes, default=cv.Stitcher_PANORAMA,
                    help='Determines configuration of stitcher. The default is `PANORAMA` (%d), '
                         'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
                         'for stitching materials under affine transformation, such as scans.' % modes)
parser.add_argument('--output', default='result.jpg',
                    help='Resulting image. The default is `result.jpg`.')
parser.add_argument('img', nargs='+', help='input images')

__doc__ += '\n' + parser.format_help()


def main():
    args = parser.parse_args()

    # read input images
    imgs = []
    for img_name in args.img:
        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            print("can't read image " + img_name)
            sys.exit(-1)
        imgs.append(img)

    # ![stitching]
    stitcher = cv.Stitcher.create(args.mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    # ![stitching]

    cv.imwrite(args.output, pano)
    print("stitching completed successfully. %s saved!" % args.output)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()