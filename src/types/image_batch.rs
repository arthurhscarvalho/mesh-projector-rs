use ndarray::iter::AxisIterMut;
use ndarray::{Array3, Array4, ArrayViewMut2, ArrayViewMut3, Axis};
use rayon::iter::IntoParallelIterator;

/// A batch of images and their associated maps (depth, normal, occupancy).
///
/// This struct stores multiple images and their corresponding depth maps,
/// normal maps, and occupancy maps in 4D/3D arrays for efficient batch processing.
/// - `images`: (N, H, W, C) array of image data.
/// - `depth_maps`: (N, H, W, C) array of depth map data.
/// - `normal_maps`: (N, H, W, C) array of normal map data.
/// - `occupancy_maps`: (N, H, W) array indicating pixel occupancy.
pub struct ImageBatch {
    pub images: Array4<u8>,
    pub depth_maps: Array3<u8>,
    pub normal_maps: Array4<u8>,
    pub occupancy_maps: Array3<u8>,
}

impl ImageBatch {
    pub fn new(num_images: usize, rows: usize, cols: usize, channels: usize) -> Self {
        ImageBatch {
            images: Array4::zeros((num_images, rows, cols, channels)),
            depth_maps: Array3::zeros((num_images, rows, cols)),
            normal_maps: Array4::zeros((num_images, rows, cols, channels)),
            occupancy_maps: Array3::zeros((num_images, rows, cols)),
        }
    }

    pub fn iter_mut(&mut self) -> ImageBatchIterMut<'_> {
        ImageBatchIterMut {
            images: self.images.axis_iter_mut(Axis(0)),
            depth_maps: self.depth_maps.axis_iter_mut(Axis(0)),
            normal_maps: self.normal_maps.axis_iter_mut(Axis(0)),
            occupancy_maps: self.occupancy_maps.axis_iter_mut(Axis(0)),
        }
    }

    pub fn par_iter_mut(&mut self) -> rayon::vec::IntoIter<ImageBatchItemMut<'_>> {
        self.iter_mut().into_par_iter()
    }
}

pub struct ImageBatchIterMut<'a> {
    images: AxisIterMut<'a, u8, ndarray::Dim<[usize; 3]>>,
    depth_maps: AxisIterMut<'a, u8, ndarray::Dim<[usize; 2]>>,
    normal_maps: AxisIterMut<'a, u8, ndarray::Dim<[usize; 3]>>,
    occupancy_maps: AxisIterMut<'a, u8, ndarray::Dim<[usize; 2]>>,
}

pub type ImageBatchItemMut<'a> = (
    ArrayViewMut3<'a, u8>,
    ArrayViewMut2<'a, u8>,
    ArrayViewMut3<'a, u8>,
    ArrayViewMut2<'a, u8>,
);

impl<'a> Iterator for ImageBatchIterMut<'a> {
    type Item = ImageBatchItemMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match (
            self.images.next(),
            self.depth_maps.next(),
            self.normal_maps.next(),
            self.occupancy_maps.next(),
        ) {
            (Some(img), Some(depth), Some(normal), Some(occ)) => Some((img, depth, normal, occ)),
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.images.size_hint()
    }
}

impl<'a> ExactSizeIterator for ImageBatchIterMut<'a> {}

impl<'a> IntoParallelIterator for ImageBatchIterMut<'a> {
    type Item = ImageBatchItemMut<'a>;
    type Iter = rayon::vec::IntoIter<Self::Item>;

    fn into_par_iter(self) -> Self::Iter {
        self.images
            .zip(self.depth_maps)
            .zip(self.normal_maps)
            .zip(self.occupancy_maps)
            .map(|(((img, depth), normal), occ)| (img, depth, normal, occ))
            .collect::<Vec<_>>()
            .into_par_iter()
    }
}
