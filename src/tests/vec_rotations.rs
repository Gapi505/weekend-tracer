#[cfg(test)]
mod tests {
    use std::num::FpCategory::Zero;
    use crate::*; // Import everything from the main crate

    macro_rules! assert_vec3_eq {
    ($a:expr, $b:expr, $eps:expr) => {
        assert!(
            ($a.x - $b.x).abs() < $eps &&
            ($a.y - $b.y).abs() < $eps &&
            ($a.z - $b.z).abs() < $eps,
            "Assertion failed: {:?} != {:?} within epsilon {}",
            $a,
            $b,
            $eps
        );
    };
}
    #[test]
    fn transform_lerp(){
        let t1 = Transform::at(vec3!(0.));
        let t2 = Transform::at(vec3!(0., 1., 0.));
        let lerped = t1.lerp(t2, 0.5);
        assert_eq!(lerped.position.y, 0.5)
    }
    #[test]
    fn transform_lerp_at_0(){
        let t1 = Transform::at(vec3!(0.));
        let t2 = Transform::at(vec3!(0., 1., 0.));
        let lerped = t1.lerp(t2, -1.);
        assert_eq!(lerped.position.y, -1.)
    }
    #[test]
    fn vec3_lerp(){
        let v1 = vec3!(0., 0., 0.);
        let v2 = vec3!(1., 0., 0.);
        let lerped = v1.lerp(&v2, 0.5);
        assert_eq!(lerped.x, 0.5)
    }
}
