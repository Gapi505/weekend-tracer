#[cfg(test)]
mod tests {
    use crate::*; // Import everything from the main crate

    #[test]
    fn test_rotation_around_y_axis() {
        let mut transform = Transform::new(
            Vec3::new(0.0, 0.0, 0.0), // Initial position
            Vec3::new(0.0, 45.0, 0.0), // Initial rotation
            Vec3::new(1.0, 1.0, 1.0),  // Scale
        );

        transform.rotate_around_axis(Vec3::up(), 45.0);

        let expected_rotation = Vec3::new(0.0, 90.0, 0.0); // Expected result
        assert!(
            (transform.rotation.x - expected_rotation.x).abs() < 0.001
                && (transform.rotation.y - expected_rotation.y).abs() < 0.001
                && (transform.rotation.z - expected_rotation.z).abs() < 0.001,
            "Expected {:?}, got {:?}",
            expected_rotation,
            transform.rotation
        );
    }

    #[test]
    fn test_rotation_around_x_axis() {
        let mut transform = Transform::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(45.0, 0.0, 0.0), // Initial rotation
            Vec3::new(1.0, 1.0, 1.0),
        );

        transform.rotate_around_axis(Vec3::new(1.0, 0.0, 0.0), 45.0);

        let expected_rotation = Vec3::new(90.0, 0.0, 0.0);
        assert!(
            (transform.rotation.x - expected_rotation.x).abs() < 0.001
                && (transform.rotation.y - expected_rotation.y).abs() < 0.001
                && (transform.rotation.z - expected_rotation.z).abs() < 0.001,
            "Expected {:?}, got {:?}",
            expected_rotation,
            transform.rotation
        );
    }

    #[test]
    fn test_rotation_around_z_axis() {
        let mut transform = Transform::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 45.0), // Initial rotation
            Vec3::new(1.0, 1.0, 1.0),
        );

        transform.rotate_around_axis(Vec3::new(0.0, 0.0, 1.0), 45.0);

        let expected_rotation = Vec3::new(0.0, 0.0, 90.0);
        assert!(
            (transform.rotation.x - expected_rotation.x).abs() < 0.001
                && (transform.rotation.y - expected_rotation.y).abs() < 0.001
                && (transform.rotation.z - expected_rotation.z).abs() < 0.001,
            "Expected {:?}, got {:?}",
            expected_rotation,
            transform.rotation
        );
    }

    #[test]
    fn test_combined_rotation() {
        let mut transform = Transform::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(30.0, 45.0, 60.0), // Initial rotation
            Vec3::new(1.0, 1.0, 1.0),
        );

        transform.rotate_around_axis(Vec3::up(), 45.0);

        let expected_rotation = Vec3::new(30.0, 90.0, 60.0);
        assert!(
            (transform.rotation.x - expected_rotation.x).abs() < 0.001
                && (transform.rotation.y - expected_rotation.y).abs() < 0.001
                && (transform.rotation.z - expected_rotation.z).abs() < 0.001,
            "Expected {:?}, got {:?}",
            expected_rotation,
            transform.rotation
        );
    }
}
