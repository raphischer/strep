# fix corrupted errors relating to imagemagick (for loading motion_blur and snow variants)
1. Make sure to install `imagemagick > 7.x`
2. In `site-packages/tensorflow_datasets/image_classification/corruptions.py`, change the following lines:
    - l.39 rename package: `return 'magick'  # pylint: disable=unreachable`
    - ll.638 - 644 restructure subprocess call: ```subprocess.check_output([
          convert_bin,
          im_input.name,
          '-motion-blur',
          '{}x{}+{}'.format(radius, sigma, angle),
          im_output.name,
      ])```
    - ll.590 - 596 restructure subprocess call: ```
    subprocess.check_output([
          convert_bin,
          im_input.name,
          '-motion-blur',
          '{}x{}+{}'.format(radius, sigma, angle),
          im_output.name,
      ])```