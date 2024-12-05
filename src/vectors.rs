use std::ops::*;
use num_traits::{Float, Num, Zero, One};

#[derive(Clone, Copy, Debug)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T>
where
    T: Copy + Num,
{
    // General constructor
    pub fn new(x: T, y: T) -> Self {
        Vec2 { x, y }
    }

    pub fn splat(value: T) -> Self {
        Vec2::new(value, value)
    }

    // Methods that work for any numeric type
    pub fn dot(&self, other: Vec2<T>) -> T {
        self.x * other.x + self.y * other.y
    }

    pub fn distance(&self, other: Vec2<T>) -> T {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    pub fn zero() -> Self {
        Self::splat(T::zero())
    }

    pub fn one() -> Self {
        Self::splat(T::one())
    }
}

// Floating-point specific methods
impl<T> Vec2<T>
where
    T: Copy + Float,
{
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    pub fn length_sq(&self) -> T {
        self.x * self.x + self.y * self.y
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len != T::zero() {
            *self / len
        } else {
            *self
        }
    }
}

// Implementing basic operations for all numeric types
impl<T> Add for Vec2<T>
where
    T: Copy + Num,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self::new(self.x + other.x, self.y + other.y)
    }
}
impl<T> AddAssign for Vec2<T>
where
    T: Copy + Num,
{
    fn add_assign(&mut self, rhs: Self) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
    }
}
impl<T> Sub for Vec2<T>
where
    T: Copy + Num,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl<T> Mul<T> for Vec2<T>
where
    T: Copy + Num,
{
    type Output = Self;
    fn mul(self, n: T) -> Self::Output {
        Self::new(self.x * n, self.y * n)
    }
}

impl<T> Div<T> for Vec2<T>
where
    T: Copy + Num,
{
    type Output = Self;
    fn div(self, n: T) -> Self::Output {
        Self::new(self.x / n, self.y / n)
    }
}

impl<T> Neg for Vec2<T>
where
    T: Copy + Num + Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl<T> std::fmt::Display for Vec2<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[macro_export]
macro_rules! vec2 {
    ($x:expr, $y:expr) => {
        Vec2::new($x, $y)
    };
    ($x:expr) => {
        Vec2::splat($x)
    };
}



#[derive(Copy, Clone, Debug)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

// Define the `new` method with minimal bounds (only `Copy`)
impl<T> Vec3<T>
where
    T: Copy
{
    pub fn new(x: T, y: T, z: T) -> Self {
        Vec3 { x, y, z }
    }

    pub fn splat(value: T) -> Self {
        Vec3::new(value, value, value)
    }
}

// General numeric methods (like `dot`, `zero`, `one`) that require `Zero` and `One`
impl<T> Vec3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero + One + Sub<Output = T>,
{
    pub fn dot(&self, other: Vec3<T>) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(&self, other: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn zero() -> Self {
        Self::splat(T::zero())
    }

    pub fn one() -> Self {
        Self::splat(T::one())
    }

    pub fn up() -> Self {Self::new(T::zero(), T::one(), T::zero())}
    pub fn right() -> Self {Self::new(T::one(), T::zero(), T::zero())}

    pub fn forward() -> Self {Self::new(T::zero(), T::zero(), T::one())}
}

// Floating-point specific methods
impl<T> Vec3<T>
where
    T: Copy + Float,
{
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn length_sq(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len != T::zero() {
            *self / len
        } else {
            *self
        }
    }

    pub fn rotate_around_origin(&mut self, rotation: Vec3<T>) -> Self{
        // Convert rotation angles from degrees to radians
        let rot_x = rotation.x.to_radians();
        let rot_y = rotation.y.to_radians();
        let rot_z = rotation.z.to_radians();

        // Store original values for clarity
        let (x, y, z) = (self.x, self.y, self.z);

        // Rotate around X-axis
        let rotated_x = Vec3 {
            x,
            y: y * rot_x.cos() - z * rot_x.sin(),
            z: y * rot_x.sin() + z * rot_x.cos(),
        };

        // Rotate around Y-axis
        let rotated_y = Vec3 {
            x: rotated_x.x * rot_y.cos() + rotated_x.z * rot_y.sin(),
            y: rotated_x.y,
            z: -rotated_x.x * rot_y.sin() + rotated_x.z * rot_y.cos(),
        };

        // Rotate around Z-axis
        let rotated_z = Vec3 {
            x: rotated_y.x * rot_z.cos() - rotated_y.y * rot_z.sin(),
            y: rotated_y.x * rot_z.sin() + rotated_y.y * rot_z.cos(),
            z: rotated_y.z,
        };

        // Update self with the final rotated vector
        rotated_z
    }
    pub fn lerp(&self, other: &Self, t: T) -> Self {
        Self {
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
            z: self.z + t * (other.z - self.z),
        }
    }
}

// Implementing all the operators for Vec3
impl<T> Add for Vec3<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T> AddAssign for Vec3<T>
where
    T: Copy + AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T> Sub for Vec3<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T> SubAssign for Vec3<T>
where
    T: Copy + SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<T> Mul<T> for Vec3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, scalar: T) -> Self::Output {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl<T> Mul for Vec3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
        )
    }
}
impl Mul<Vec3<f32>> for f32 {
    type Output = Vec3<f32>;

    fn mul(self, vec: Vec3<f32>) -> Self::Output {
        Vec3::new(self * vec.x, self * vec.y, self * vec.z)
    }
}

impl<T> MulAssign<T> for Vec3<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, scalar: T) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl<T> Div<T> for Vec3<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;
    fn div(self, scalar: T) -> Self::Output {
        Self::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl<T> DivAssign<T> for Vec3<T>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, scalar: T) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

impl<T> Neg for Vec3<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<T> std::fmt::Display for Vec3<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}


#[macro_export]
macro_rules! vec3 {
    ($x:expr, $y:expr, $z:expr) => {
        Vec3::new($x, $y, $z)
    };
    ($x:expr) => {
        Vec3::splat($x)
    };
}


// Vec4 Definition
#[derive(Copy, Clone, Debug)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

// General implementation with minimal bounds for `new`
impl<T> Vec4<T>
where
    T: Copy,
{
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Vec4 { x, y, z, w }
    }

    pub fn splat(value: T) -> Self {
        Vec4::new(value, value, value, value)
    }
}

// Implementation for numeric types (`dot`, `zero`, `one`)
impl<T> Vec4<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero + One,
{
    pub fn dot(&self, other: Vec4<T>) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn zero() -> Self {
        Self::splat(T::zero())
    }

    pub fn one() -> Self {
        Self::splat(T::one())
    }
}

// Floating-point specific methods
impl<T> Vec4<T>
where
    T: Copy + Float,
{
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len != T::zero() {
            *self / len
        } else {
            *self
        }
    }
}

// Implementing all the operators for Vec4
impl<T> Add for Vec4<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

impl<T> AddAssign for Vec4<T>
where
    T: Copy + AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<T> Sub for Vec4<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

impl<T> SubAssign for Vec4<T>
where
    T: Copy + SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<T> Mul<T> for Vec4<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, scalar: T) -> Self::Output {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)
    }
}

impl<T> MulAssign<T> for Vec4<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, scalar: T) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
    }
}

impl<T> Div<T> for Vec4<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;
    fn div(self, scalar: T) -> Self::Output {
        Self::new(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)
    }
}

impl<T> DivAssign<T> for Vec4<T>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, scalar: T) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
    }
}

impl<T> Neg for Vec4<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl<T> std::fmt::Display for Vec4<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

// Macro for Vec4
#[macro_export]
macro_rules! vec4 {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        Vec4::new($x, $y, $z, $w)
    };
    ($x:expr) => {
        Vec4::splat($x)
    };
}



#[derive(Debug, Copy, Clone)]
pub(crate) struct Transform{
    pub position: Vec3<f32>,
    pub rotation: Vec3<f32>, // Euler angles
    pub direction: Vec3<f32>, // Directional vector. Represents the same value as rotation but different format
    pub scale: Vec3<f32>,
}
impl Transform {
    pub fn new(position: Vec3<f32>, rotation: Vec3<f32>, direction: Vec3<f32>, scale: Vec3<f32>) -> Self {
        Self{
            position,
            rotation,
            direction,
            scale,
        }
    }
    pub fn at(position: Vec3<f32>) -> Self {
        Self{
            position,
            ..Default::default()
        }
    }
    pub fn lerp(mut self, to: Transform, t: f32) -> Self {
        self.position = self.position.lerp(&to.position, t);
        self.rotation = self.rotation.lerp(&to.rotation, t);
        self.scale = self.scale.lerp(&to.scale, t);
        self.update_dir();
        self
    }
    pub fn facing(&mut self, facing: Vec3<f32>) -> &mut Self {
        self.direction = facing;
        self
    }
    /// updates rotation based on direction. they should represent the same value
    pub fn update_rot(&mut self) -> Vec3<f32> {
        // Normalize the direction vector to ensure it's a unit vector
        let dir = self.direction.normalize();

        // Calculate the pitch (rotation around the X-axis)
        let pitch = dir.y.asin();

        // Calculate the yaw (rotation around the Y-axis)
        let yaw = dir.x.atan2(dir.z);

        // Update the rotation vector (keeping the roll the same)
        self.rotation = vec3![yaw, pitch, self.rotation.z];

        // Return the updated rotation
        self.rotation
    }

    pub fn update_dir(&mut self) -> Vec3<f32> {
        let (yaw_sin, yaw_cos) = self.rotation.x.sin_cos();
        let (pitch_sin, pitch_cos) = self.rotation.y.sin_cos();

        self.direction = vec3![
        pitch_cos * yaw_sin,  // X component
        pitch_sin,            // Y component
        pitch_cos * yaw_cos   // Z component
    ];
        self.direction
    }
    /// Returns the right vector based on the current direction
    pub fn right(&self) -> Vec3<f32> {
        // World up vector (usually fixed to Y axis)
        let world_up = vec3![0.0, 1.0, 0.0];
        // Calculate the right vector
        self.direction.cross(world_up).normalize()
    }

    /// Returns the up vector based on the current direction
    pub fn up(&self) -> Vec3<f32> {
        // Calculate the right vector
        let right = self.right();
        // Calculate the up vector
        right.cross(self.direction).normalize()
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self{
            position: vec3![0.,0.,0.],
            rotation: vec3![0.,0.,0.],
            direction: vec3![0.,0.,1.],
            scale: vec3![0.,0.,0.]
        }
    }
}

