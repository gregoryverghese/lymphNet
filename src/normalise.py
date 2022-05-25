import staintools
import datetime

# Set upP
PROJECT_DIR = 'C:/Users/hooll/Dropbox/19 PostGrad/DigitalPathology/gc-stainnorm/ln-gc-classification/'
RESULTS_DIR = PROJECT_DIR+'results/'
#METHOD = 'macenko'
METHOD = 'vahadane'

# Read data
target = staintools.read_image(PROJECT_DIR+"data/my_target_image.png")
to_transform = staintools.read_image(PROJECT_DIR+"data/image2.png")
to_transform1 = staintools.read_image(PROJECT_DIR+"data/image3.png")
to_transform2 = staintools.read_image(PROJECT_DIR+"data/image4.png")

# Plot
images = [target, to_transform, to_transform1, to_transform2]
titles = ["Target"] + ["Original"] *3
staintools.plot_image_list(images, width=4, title_list=titles,
                            save_name=RESULTS_DIR + 'original-images.png', show=0)

# Standardize brightness (optional, can improve the tissue mask calculation)
target = staintools.LuminosityStandardizer.standardize(target)
to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
to_transform1 = staintools.LuminosityStandardizer.standardize(to_transform1)
to_transform2 = staintools.LuminosityStandardizer.standardize(to_transform2)

# Plot
images = [target, to_transform, to_transform1, to_transform2]
titles = ["Target standardized"] + ["Original standardized"]*3
staintools.plot_image_list(images, width=4, title_list=titles,
                            save_name=RESULTS_DIR + 'original-images-standardized.png', show=0)
                            
# Stain normalize
normalizer = staintools.StainNormalizer(method=METHOD)
normalizer.fit(target)
transformed = normalizer.transform(to_transform)
transformed1 = normalizer.transform(to_transform1)
transformed2 = normalizer.transform(to_transform2)

images = [target, transformed,transformed1,transformed2]
titles = ["Target"] + ["Stain normalized"]*3
staintools.plot_image_list(images, width=4, title_list=titles,
                            save_name=RESULTS_DIR + 'stain-normalized-images' + METHOD + '.png', show=0)
