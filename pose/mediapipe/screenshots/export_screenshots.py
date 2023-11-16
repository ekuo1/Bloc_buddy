import imageio

reader = imageio.get_reader('output_red_horizontal.mp4')

for frame_number, im in enumerate(reader):
    # im is numpy array
    if frame_number % 20 == 0:
        imageio.imwrite(f'frame_{frame_number}.jpg', im)