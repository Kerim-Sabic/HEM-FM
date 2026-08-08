import numpy as np

from hemfm.augmentations import apply_points, augmentation_matrix, physical_distance


def test_augmented_coordinates_preserve_physical_length_with_updated_chain():
    image_to_physical = np.asarray([[0.4, 0, -10], [0, 0.7, 5], [0, 0, 1]], dtype=float)
    points = np.asarray([[20, 30], [80, 90]], dtype=float)
    augment = augmentation_matrix((112, 112), 12, 1.1, (5, -3))
    augmented = apply_points(augment, points)
    augmented_to_physical = image_to_physical @ np.linalg.inv(augment)
    np.testing.assert_allclose(
        physical_distance(augmented, augmented_to_physical),
        physical_distance(points, image_to_physical),
        rtol=0,
        atol=1e-10,
    )

