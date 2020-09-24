import Augmentor as Augmentor

image_path = 'images/'
output_directory = 'output/'

p = Augmentor.Pipeline(
            source_directory=image_path,
            output_directory=output_directory
        )

#p.flip_left_right(probability=0.5)
#p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
#p.zoom_random(probability=1, percentage_area=0.8)
p.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=8)
#p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
#p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
#p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
#p.random_erasing(probability=0.5, rectangle_area=0.2)


p.sample(1)