This repo contains codes to use SAM for automatic and semi-automatic mask generation. The pretrained model used in this code can be downloaded from [here](https://drive.google.com/file/d/1aAgfYAh9hTECz8w5clWfU2mwQOkUlJrn/view?usp=sharing).

"draw_mask.py" is semi-auto masking code where you can draw a rectangular box around the object. Instantaneously runs after drawing the box.
"auto_mask_generator.py" is automated masking code. It takes around 75sec for each image.