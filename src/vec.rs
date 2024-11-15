use std::ops::*;


// Vec2 Definition
#[derive(Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn splat(value: f32) -> Self {
        Vec2::new(value, value)
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn dot(&self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }
    pub fn lerp_assign(&mut self, other: Self, t: f32) {
        self.x += t * (other.x - self.x);
        self.y += t * (other.y - self.y);
    }
    pub fn expand(&self, z: f32) -> Vec3 {
        Vec3::new(self.x, self.y, z)
    }
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len != 0.0 {
            *self / len
        } else {
            *self
        }
    }
    pub fn distance(&self, other: Self) -> f32 {
        (*self - other).length()
    }

    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    pub fn one() -> Self {
        Self::splat(1.0)
    }

}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, n: f32) -> Self {
        Self::new(self.x * n, self.y * n)
    }
}

impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, n: f32) {
        self.x *= n;
        self.y *= n;
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, n: f32) -> Self {
        Self::new(self.x / n, self.y / n)
    }
}

impl DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, n: f32) {
        self.x /= n;
        self.y /= n;
    }
}

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// Macro for Vec2
macro_rules! vec2 {
    ($x:expr, $y:expr) => {
        Vec2::new($x, $y)
    };
    ($x:expr) => {
        Vec2::splat($x)
    };
}

#[derive(Copy, Clone)]
pub struct Vec3{
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Vec3{
    pub fn new(x: f32, y: f32, z: f32) -> Vec3{
        Vec3{x,y,z}
    }
    pub fn splat(x: f32) -> Vec3{
        Self::new(x,x,x)
    }
    pub fn length(&self) -> f32{
        (self.x*self.x + self.y*self.y + self.z*self.z).sqrt()
    }
    pub fn dot(&self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn lerp_assign(&mut self, other: Self, t: f32) {
        self.x += t * (other.x - self.x);
        self.y += t * (other.y - self.y);
        self.z += t * (other.z - self.z);
    }
    pub fn expand(&self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    pub fn cut(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len != 0.0 {
            *self / len
        } else {
            *self
        }
    }
    pub fn distance(&self, other: Self) -> f32 {
        (*self - other).length()
    }
    pub fn hadamard(&self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
    pub fn reflect(&self, normal: Vec3) -> Vec3 {
        *self - normal * 2.0 * self.dot(normal)
    }
    pub fn angle_between(&self, other: Self) -> f32 {
        let dot = self.dot(other);
        let len_product = self.length() * other.length();
        if len_product != 0.0 {
            (dot / len_product).acos()
        } else {
            0.0
        }
    }
    pub fn clamp(&self, min: Self, max: Self) -> Self {
        Self::new(
            self.x.max(min.x).min(max.x),
            self.y.max(min.y).min(max.y),
            self.z.max(min.z).min(max.z),
        )
    }
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    pub fn one() -> Self {
        Self::splat(1.0)
    }

}
impl std::fmt::Display for Vec3{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error>{
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl Add for Vec3{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}
impl AddAssign for Vec3{
    fn add_assign(&mut self, other: Self){
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}
impl Sub for Vec3{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}
impl SubAssign for Vec3{
    fn sub_assign(&mut self, other: Self){
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}
impl Mul<f32> for Vec3{
    type Output = Self;
    fn mul(self, n :f32) -> Self {
        Self::new(self.x * n,self.y * n,self.z * n)
    }
}
impl Mul for Vec3 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, n : Vec3) -> Vec3 {
        Vec3::new(self * n.x,self * n.y,self * n.z)
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, n: f32) {
        self.x *= n;
        self.y *= n;
        self.z *= n;
    }
}

// Implement vector multiplication assignment: Vec3 *= Vec3
impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}
// Implement scalar division: Vec3 / f32
impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, n: f32) -> Self {
        Self::new(self.x / n, self.y / n, self.z / n)
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, n: f32) {
        self.x /= n;
        self.y /= n;
        self.z /= n;
    }
}

impl Div for Vec3 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

impl DivAssign for Vec3 {
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
    }
}
impl Neg for Vec3{
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}


macro_rules! vec3 {
    ($x:expr, $y:expr, $z:expr) => {
        Vec3::new($x, $y, $z)
    };
    ($x:expr) => {
        Vec3::splat($x)
    };
}


// Vec4 Definition
#[derive(Copy, Clone)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 { x, y, z, w }
    }

    pub fn splat(value: f32) -> Self {
        Vec4::new(value, value, value, value)
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn dot(&self, other: Vec4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
    pub fn lerp_assign(&mut self, other: Self, t: f32) {
        self.x += t * (other.x - self.x);
        self.y += t * (other.y - self.y);
        self.z += t * (other.z - self.z);
        self.w += t * (other.w - self.w);
    }
    pub fn cut(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len != 0.0 {
            *self / len
        } else {
            *self
        }
    }
    pub fn distance(&self, other: Self) -> f32 {
        (*self - other).length()
    }
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    pub fn one() -> Self {
        Self::splat(1.0)
    }

}

impl Add for Vec4 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

impl AddAssign for Vec4 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl Sub for Vec4 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

impl SubAssign for Vec4 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    fn mul(self, n: f32) -> Self {
        Self::new(self.x * n, self.y * n, self.z * n, self.w * n)
    }
}

impl MulAssign<f32> for Vec4 {
    fn mul_assign(&mut self, n: f32) {
        self.x *= n;
        self.y *= n;
        self.z *= n;
        self.w *= n;
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;
    fn div(self, n: f32) -> Self {
        Self::new(self.x / n, self.y / n, self.z / n, self.w / n)
    }
}

impl DivAssign<f32> for Vec4 {
    fn div_assign(&mut self, n: f32) {
        self.x /= n;
        self.y /= n;
        self.z /= n;
        self.w /= n;
    }
}

impl Neg for Vec4 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl std::fmt::Display for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

// Macro for Vec4
macro_rules! vec4 {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        Vec4::new($x, $y, $z, $w)
    };
    ($x:expr) => {
        Vec4::splat($x)
    };
}