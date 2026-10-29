use core::array;
use core::slice;
use std::iter::{Flatten, Sum};
use std::mem::MaybeUninit;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Neg;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    content: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Vec<usize>, values: Vec<T>) -> Option<Self> {
        if shape.is_empty() {
            return None;
        };
        let shape_len: usize = shape.iter().product();
        if values.len() != shape_len {
            return None;
        }

        Some(Self {
            shape,
            content: values,
        })
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> bool {
        let actual_len: usize = shape.iter().product();
        if self.content.len() != actual_len {
            return false;
        }

        self.shape = shape;
        true
    }

    pub fn flatten_indices(shape: &[usize], indices: &[usize]) -> usize {
        debug_assert_eq!(shape.len(), indices.len());

        let mut final_index = 0;
        for (dim, i) in shape.iter().zip(indices.iter()) {
            final_index *= *dim;
            final_index += *i;
        }

        final_index
    }

    pub fn shapen_index(shape: &[usize], mut index: usize) -> Vec<usize> {
        let mut final_indices = vec![];
        for dim in shape {
            final_indices.push(index % dim);
            index /= dim;
        }

        final_indices
    }
}

impl<T: Sum<T> + Mul<Output = T> + Clone> Tensor<T> {
    pub fn dot(a: Tensor<T>, b: Tensor<T>) -> Option<Tensor<T>> {
        if a.shape.last() != b.shape.first() {
            return None;
        }

        let mut shape: Vec<usize> = vec![];
        shape.extend_from_slice(&a.shape[..a.shape.len() - 1]);
        shape.extend_from_slice(&b.shape[1..]);
        let len_a: usize = a.shape[..a.shape.len() - 1].iter().product();
        let len_b: usize = b.shape[1..].iter().product();

        let size = b.shape[0];

        let mut content = Vec::with_capacity(shape.iter().product());
        for rel_i_a in 0..len_a {
            for rel_i_b in 0..len_b {
                let i_a = rel_i_a * size;
                let i_b = rel_i_b;

                let value = (0..size)
                    .map(|i| a.content[i_a + i].clone() * b.content[i_b + len_b * i].clone())
                    .sum();

                content.push(value); // rel_i_a * len_b + rel_i_b
            }
        }

        Some(Tensor { shape, content })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Vector<T, const N: usize> {
    content: [T; N],
}

impl<T, const N: usize> IntoIterator for Vector<T, N> {
    type Item = T;
    type IntoIter = array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.content.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Vector<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Vector<T, N> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn new(content: [T; N]) -> Self {
        Self { content }
    }

    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.content.iter()
    }

    pub fn map_ref<F: FnMut(&T) -> T>(&mut self, mut function: F) {
        for i in 0..N {
            self.content[i] = function(&mut self.content[i])
        }
    }

    pub fn map_ref_enumerate<F: FnMut(&T, usize) -> T>(&mut self, mut function: F) {
        for i in 0..N {
            self.content[i] = function(&mut self.content[i], i)
        }
    }

    pub fn map_mut<F: FnMut(&mut T)>(&mut self, mut function: F) {
        for i in 0..N {
            function(&mut self.content[i])
        }
    }

    pub fn map_mut_enumerate<F: FnMut(&mut T, usize)>(&mut self, mut function: F) {
        for i in 0..N {
            function(&mut self.content[i], i)
        }
    }

    pub fn map<U, F: FnMut(T) -> U>(self, function: F) -> Vector<U, N> {
        Vector {
            content: self.content.map(function),
        }
    }

    pub fn map_enumerate<F: FnMut(T, usize) -> T>(self, mut function: F) -> Vector<T, N> {
        let mut content: [MaybeUninit<T>; N] = MaybeUninit::new(self.content).into();

        for (i, item) in content.iter_mut().enumerate() {
            let value = unsafe { item.assume_init_read() };
            item.write(function(value, i));
        }

        let content: MaybeUninit<[T; N]> = content.into();

        Vector {
            content: unsafe { content.assume_init_read() },
        }
    }

    pub fn content(&self) -> &[T; N] {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut [T; N] {
        &mut self.content
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.content.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.content.get_mut(index)
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.content[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.content[index]
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.content.iter_mut()
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(value: [T; N]) -> Self {
        Vector { content: value }
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
    fn from(value: Vector<T, N>) -> Self {
        value.content
    }
}

impl<T: Copy, const N: usize> Vector<T, N> {
    pub fn add_left<Lhs: Add<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left + *val);
    }

    pub fn add_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Add<Rhs, Output = T>,
    {
        self.map_ref(|val| *val + right);
    }

    pub fn sub_left<Lhs: Sub<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left - *val);
    }

    pub fn sub_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Sub<Rhs, Output = T>,
    {
        self.map_ref(|val| *val - right);
    }

    pub fn mul_left<Lhs: Mul<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left * *val);
    }

    pub fn mul_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Mul<Rhs, Output = T>,
    {
        self.map_ref(|val| *val * right);
    }

    pub fn div_left<Lhs: Div<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left / *val);
    }

    pub fn div_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Div<Rhs, Output = T>,
    {
        self.map_ref(|val| *val / right);
    }

    pub fn rem_left<Lhs: Rem<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left % *val);
    }

    pub fn rem_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Rem<Rhs, Output = T>,
    {
        self.map_ref(|val| *val % right);
    }
}

impl<T: Add<Output = T> + Copy, const N: usize> Add for Vector<T, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_ref_enumerate(|val, i| *val + unsafe { rhs[i].assume_init() });

        self
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> Sub for Vector<T, N> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_ref_enumerate(|val, i| *val - unsafe { rhs[i].assume_init() });

        self
    }
}

impl<T: Mul<Output = T> + Copy, const N: usize> Mul for Vector<T, N> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_ref_enumerate(|val, i| *val * unsafe { rhs[i].assume_init() });

        self
    }
}

impl<T: Div<Output = T> + Copy, const N: usize> Div for Vector<T, N> {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_ref_enumerate(|val, i| *val / unsafe { rhs[i].assume_init() });

        self
    }
}

impl<T: Rem<Output = T> + Copy, const N: usize> Rem for Vector<T, N> {
    type Output = Self;

    fn rem(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_ref_enumerate(|val, i| *val % unsafe { rhs[i].assume_init() });

        self
    }
}

impl<T: Neg<Output = T> + Copy, const N: usize> Neg for Vector<T, N> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.map_ref(|val| -*val);

        self
    }
}

impl<T: AddAssign, const N: usize> AddAssign for Vector<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_mut_enumerate(|val, i| *val += unsafe { rhs[i].assume_init_read() });
    }
}

impl<T: SubAssign, const N: usize> SubAssign for Vector<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_mut_enumerate(|val, i| *val -= unsafe { rhs[i].assume_init_read() });
    }
}

impl<T: MulAssign, const N: usize> MulAssign for Vector<T, N> {
    fn mul_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_mut_enumerate(|val, i| *val *= unsafe { rhs[i].assume_init_read() });
    }
}

impl<T: DivAssign, const N: usize> DivAssign for Vector<T, N> {
    fn div_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_mut_enumerate(|val, i| *val /= unsafe { rhs[i].assume_init_read() });
    }
}

impl<T: RemAssign, const N: usize> RemAssign for Vector<T, N> {
    fn rem_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<T>; N] = MaybeUninit::new(rhs.content).into();
        self.map_mut_enumerate(|val, i| *val %= unsafe { rhs[i].assume_init_read() });
    }
}

impl<T: Mul + Clone, const N: usize> Vector<T, N> {
    pub fn multiply(&self, other: &T) -> Vector<T::Output, N> {
        let mut result: Vector<MaybeUninit<T::Output>, N> = Vector {
            content: [const { MaybeUninit::uninit() }; N],
        };

        for i in 0..N {
            result.content[i].write(self.content[i].clone() * other.clone());
        }
        let actual_content: MaybeUninit<[T::Output; N]> = result.content.into();

        Vector {
            content: unsafe { actual_content.assume_init() },
        }
    }

    pub fn multiply_elem(&self, other: &Vector<T, N>) -> Vector<T::Output, N> {
        let mut result: Vector<MaybeUninit<T::Output>, N> = Vector {
            content: [const { MaybeUninit::uninit() }; N],
        };

        for i in 0..N {
            result.content[i].write(self.content[i].clone() * other.content[i].clone());
        }
        let actual_content: MaybeUninit<[T::Output; N]> = result.content.into();

        Vector {
            content: unsafe { actual_content.assume_init() },
        }
    }
}

impl<T: Div + Clone, const N: usize> Vector<T, N> {
    pub fn divide(&self, other: &T) -> Vector<T::Output, N> {
        let mut result: Vector<MaybeUninit<T::Output>, N> = Vector {
            content: [const { MaybeUninit::uninit() }; N],
        };

        for i in 0..N {
            result.content[i].write(self.content[i].clone() / other.clone());
        }
        let actual_content: MaybeUninit<[T::Output; N]> = result.content.into();

        Vector {
            content: unsafe { actual_content.assume_init() },
        }
    }

    pub fn divide_elem(&self, other: &Vector<T, N>) -> Vector<T::Output, N> {
        let mut result: Vector<MaybeUninit<T::Output>, N> = Vector {
            content: [const { MaybeUninit::uninit() }; N],
        };

        for i in 0..N {
            result.content[i].write(self.content[i].clone() / other.content[i].clone());
        }
        let actual_content: MaybeUninit<[T::Output; N]> = result.content.into();

        Vector {
            content: unsafe { actual_content.assume_init() },
        }
    }
}

impl<T: Sum<T> + Mul<Output = T> + Clone, const N: usize> Vector<T, N> {
    pub fn dot_vec(&self, other: &Vector<T, N>) -> T {
        self.content
            .iter()
            .zip(other.content.iter())
            .map(|(left, right)| left.clone() * right.clone())
            .sum()
    }

    pub fn dot_mat<const M: usize>(&self, other: &Matrix<T, N, M>) -> Vector<T, M> {
        let mut result: Vector<MaybeUninit<T>, M> = Vector {
            content: [const { MaybeUninit::uninit() }; M],
        };

        for i in 0..M {
            result.content[i].write(
                (0..N)
                    .map(|j| self.content[j].clone() * other.content[j][i].clone())
                    .sum(),
            );
        }
        let actual_content: MaybeUninit<[T; M]> = result.content.into();

        Vector {
            content: unsafe { actual_content.assume_init() },
        }
    }
}

impl<T: Mul<Output = T> + Clone, const N: usize> Vector<T, N> {
    pub fn outer<const M: usize>(&self, other: &Vector<T, M>) -> Matrix<T, N, M> {
        let mut result: Matrix<MaybeUninit<T>, N, M> = Matrix {
            content: [const { [const { MaybeUninit::uninit() }; M] }; N],
        };

        for j in 0..N {
            for i in 0..M {
                result.content[j][i].write(self.content[j].clone() * other.content[i].clone());
            }
        }
        let mid_content: [MaybeUninit<[T::Output; M]>; N] =
            result.content.map(|row: [MaybeUninit<T>; M]| row.into());
        let actual_content: MaybeUninit<[[T; M]; N]> = mid_content.into();

        Matrix {
            content: unsafe { actual_content.assume_init() },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Matrix<T, const H: usize, const W: usize> {
    content: [[T; W]; H],
}

impl<T, const H: usize, const W: usize> IntoIterator for Matrix<T, H, W> {
    type Item = T;
    type IntoIter = Flatten<array::IntoIter<[T; W], H>>;

    fn into_iter(self) -> Self::IntoIter {
        self.content.into_iter().flatten()
    }
}

impl<'a, T, const H: usize, const W: usize> IntoIterator for &'a Matrix<T, H, W> {
    type Item = &'a T;
    type IntoIter = Flatten<slice::Iter<'a, [T; W]>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const H: usize, const W: usize> IntoIterator for &'a mut Matrix<T, H, W> {
    type Item = &'a mut T;
    type IntoIter = Flatten<slice::IterMut<'a, [T; W]>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const H: usize, const W: usize> From<[[T; W]; H]> for Matrix<T, H, W> {
    fn from(value: [[T; W]; H]) -> Self {
        Matrix { content: value }
    }
}

impl<T, const H: usize, const W: usize> From<Matrix<T, H, W>> for [[T; W]; H] {
    fn from(value: Matrix<T, H, W>) -> Self {
        value.content
    }
}

impl<T, const H: usize, const W: usize> Matrix<T, H, W> {
    pub fn new(content: [[T; W]; H]) -> Self {
        Self { content }
    }

    pub fn iter(&self) -> Flatten<slice::Iter<'_, [T; W]>> {
        self.content.iter().flatten()
    }

    pub fn map_ref<F: FnMut(&T) -> T>(&mut self, mut function: F) {
        for j in 0..H {
            for i in 0..W {
                self.content[j][i] = function(&mut self.content[j][i])
            }
        }
    }

    pub fn map_ref_enumerate<F: FnMut(&T, usize, usize) -> T>(&mut self, mut function: F) {
        for j in 0..H {
            for i in 0..W {
                self.content[j][i] = function(&mut self.content[j][i], j, i)
            }
        }
    }

    pub fn map_mut<F: FnMut(&mut T)>(&mut self, mut function: F) {
        for j in 0..H {
            for i in 0..W {
                function(&mut self.content[j][i])
            }
        }
    }

    pub fn map_mut_enumerate<F: FnMut(&mut T, usize, usize)>(&mut self, mut function: F) {
        for j in 0..H {
            for i in 0..W {
                function(&mut self.content[j][i], j, i)
            }
        }
    }

    pub fn map<U, F: FnMut(T) -> U>(self, mut function: F) -> Matrix<U, H, W> {
        Matrix {
            content: self.content.map(move |row| row.map(&mut function)),
        }
    }

    pub fn map_enumerate<F: FnMut(T, usize, usize) -> T>(self, mut function: F) -> Matrix<T, H, W> {
        let content: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(self.content).into();
        let mut content: [[MaybeUninit<T>; W]; H] = content.map(|row| row.into());

        for (j, row) in content.iter_mut().enumerate() {
            for (i, item) in row.iter_mut().enumerate() {
                let value = unsafe { item.assume_init_read() };
                item.write(function(value, j, i));
            }
        }

        let content: [MaybeUninit<[T; W]>; H] = content.map(|row| row.into());
        let content: MaybeUninit<[[T; W]; H]> = content.into();

        Matrix {
            content: unsafe { content.assume_init_read() },
        }
    }

    pub fn content(&self) -> &[[T; W]; H] {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut [[T; W]; H] {
        &mut self.content
    }

    pub fn get(&self, row_index: usize, col_index: usize) -> Option<&T> {
        self.content.get(row_index)?.get(col_index)
    }

    pub fn get_mut(&mut self, row_index: usize, col_index: usize) -> Option<&mut T> {
        self.content.get_mut(row_index)?.get_mut(col_index)
    }
}

impl<T, const H: usize, const W: usize> Index<(usize, usize)> for Matrix<T, H, W> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.content[index.0][index.1]
    }
}

impl<T, const H: usize, const W: usize> IndexMut<(usize, usize)> for Matrix<T, H, W> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.content[index.0][index.1]
    }
}

impl<T, const H: usize, const W: usize> Matrix<T, H, W> {
    pub fn iter_mut(&mut self) -> Flatten<slice::IterMut<'_, [T; W]>> {
        self.content.iter_mut().flatten()
    }
}

impl<T: Copy, const H: usize, const W: usize> Matrix<T, H, W> {
    pub fn add_left<Lhs: Add<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left + *val);
    }

    pub fn add_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Add<Rhs, Output = T>,
    {
        self.map_ref(|val| *val + right);
    }

    pub fn sub_left<Lhs: Sub<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left - *val);
    }

    pub fn sub_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Sub<Rhs, Output = T>,
    {
        self.map_ref(|val| *val - right);
    }

    pub fn mul_left<Lhs: Mul<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left * *val);
    }

    pub fn mul_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Mul<Rhs, Output = T>,
    {
        self.map_ref(|val| *val * right);
    }

    pub fn div_left<Lhs: Div<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left / *val);
    }

    pub fn div_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Div<Rhs, Output = T>,
    {
        self.map_ref(|val| *val / right);
    }

    pub fn rem_left<Lhs: Rem<T, Output = T> + Copy>(&mut self, left: Lhs) {
        self.map_ref(|val| left % *val);
    }

    pub fn rem_right<Rhs: Copy>(&mut self, right: Rhs)
    where
        T: Rem<Rhs, Output = T>,
    {
        self.map_ref(|val| *val % right);
    }
}

impl<T: Add<Output = T> + Copy, const H: usize, const W: usize> Add for Matrix<T, H, W> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_ref_enumerate(|val, j, i| *val + unsafe { rhs[j][i].assume_init() });

        self
    }
}

impl<T: Sub<Output = T> + Copy, const H: usize, const W: usize> Sub for Matrix<T, H, W> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_ref_enumerate(|val, j, i| *val - unsafe { rhs[j][i].assume_init() });

        self
    }
}

impl<T: Mul<Output = T> + Copy, const H: usize, const W: usize> Mul for Matrix<T, H, W> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_ref_enumerate(|val, j, i| *val * unsafe { rhs[j][i].assume_init() });

        self
    }
}

impl<T: Div<Output = T> + Copy, const H: usize, const W: usize> Div for Matrix<T, H, W> {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_ref_enumerate(|val, j, i| *val / unsafe { rhs[j][i].assume_init() });

        self
    }
}

impl<T: Rem<Output = T> + Copy, const H: usize, const W: usize> Rem for Matrix<T, H, W> {
    type Output = Self;

    fn rem(mut self, rhs: Self) -> Self::Output {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_ref_enumerate(|val, j, i| *val % unsafe { rhs[j][i].assume_init() });

        self
    }
}

impl<T: Neg<Output = T> + Copy, const H: usize, const W: usize> Neg for Matrix<T, H, W> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.map_ref(|val| -*val);

        self
    }
}

impl<T: AddAssign, const H: usize, const W: usize> AddAssign for Matrix<T, H, W> {
    fn add_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_mut_enumerate(|val, j, i| *val += unsafe { rhs[j][i].assume_init_read() });
    }
}

impl<T: SubAssign, const H: usize, const W: usize> SubAssign for Matrix<T, H, W> {
    fn sub_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_mut_enumerate(|val, j, i| *val -= unsafe { rhs[j][i].assume_init_read() });
    }
}

impl<T: MulAssign, const H: usize, const W: usize> MulAssign for Matrix<T, H, W> {
    fn mul_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_mut_enumerate(|val, j, i| *val *= unsafe { rhs[j][i].assume_init_read() });
    }
}

impl<T: DivAssign, const H: usize, const W: usize> DivAssign for Matrix<T, H, W> {
    fn div_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_mut_enumerate(|val, j, i| *val /= unsafe { rhs[j][i].assume_init_read() });
    }
}

impl<T: RemAssign, const H: usize, const W: usize> RemAssign for Matrix<T, H, W> {
    fn rem_assign(&mut self, rhs: Self) {
        let rhs: [MaybeUninit<[T; W]>; H] = MaybeUninit::new(rhs.content).into();
        let rhs: [[MaybeUninit<T>; W]; H] = rhs.map(|row| row.into());
        self.map_mut_enumerate(|val, j, i| *val %= unsafe { rhs[j][i].assume_init_read() });
    }
}

impl<T: Sum<T> + Mul<Output = T> + Clone, const H: usize, const W: usize> Matrix<T, H, W> {
    pub fn dot_vec(&self, other: &Vector<T, W>) -> Vector<T, H> {
        let mut result: Vector<MaybeUninit<T>, H> = Vector {
            content: [const { MaybeUninit::uninit() }; H],
        };

        for j in 0..H {
            result.content[j].write(
                self.content[j]
                    .iter()
                    .zip(other.content.iter())
                    .map(|(left, right)| left.clone() * right.clone())
                    .sum(),
            );
        }
        let actual_content: MaybeUninit<[T; H]> = result.content.into();

        Vector {
            content: unsafe { actual_content.assume_init() },
        }
    }
}

impl<T: Sum<T> + Mul<Output = T> + Clone, const H: usize, const M: usize> Matrix<T, H, M> {
    pub fn dot_mat<const W: usize>(&self, other: &Matrix<T, M, W>) -> Matrix<T, H, W> {
        let mut result: Matrix<MaybeUninit<T>, H, W> = Matrix {
            content: [const { [const { MaybeUninit::uninit() }; W] }; H],
        };

        for j in 0..H {
            for i in 0..W {
                result.content[j][i].write(
                    (0..M)
                        .map(|k| self.content[j][k].clone() * other.content[k][i].clone())
                        .sum(),
                );
            }
        }

        let mid_content: [MaybeUninit<[T::Output; W]>; H] = result.content.map(|row| row.into());
        let actual_content: MaybeUninit<[[T; W]; H]> = mid_content.into();

        Matrix {
            content: unsafe { actual_content.assume_init() },
        }
    }
}
