for f in "/home/ubuntu/keras/enver/acrec/Frames/*/*/*/*.jpg"
do
     mogrify $f -resize 224x224! $f
done



