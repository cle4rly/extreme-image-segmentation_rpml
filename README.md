<h1>Research Project MLCV - Extreme Image Segmentation</h1>

This repository includes some scripts to generate 3d grayscale images of smoth curves cutting a cube
  which should be then distinguished by a very basic algorithm for image segmentation (seeded region growing).

generate problems:
- problems.py (construct curves)
- calc_points.py (save as matrix)
- calc_dists.py (calculate curve distances)
- gen_data.py (add noise)

analysing segmentation result:
- analyse_vari.py (variation of information)
- analyse_rand.py (Rand Index)
- analyse_unclass.py (number of unclassified pixel)

visualizing matrices by saving files:
- show_values.py
- show_regions.py 
- show_dists.py
