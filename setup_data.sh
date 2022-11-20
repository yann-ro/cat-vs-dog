wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
unzip cats_and_dogs_filtered.zip

mv -v cats_and_dogs_filtered/train/dogs/* cats_and_dogs_filtered/train/
mv -v cats_and_dogs_filtered/train/cats/* cats_and_dogs_filtered/train/
rm -rf cats_and_dogs_filtered/train/cats/
rm -rf cats_and_dogs_filtered/train/dogs/

mv -v cats_and_dogs_filtered/validation/dogs/* cats_and_dogs_filtered/validation/
mv -v cats_and_dogs_filtered/validation/cats/* cats_and_dogs_filtered/validation/
rm -rf cats_and_dogs_filtered/validation/cats/
rm -rf cats_and_dogs_filtered/validation/dogs/

rm cats_and_dogs_filtered.zip

mkdir best_models