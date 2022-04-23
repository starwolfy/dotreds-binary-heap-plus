#![allow(clippy::needless_doctest_main)]
#![allow(missing_docs)]

use compare::Compare;
use core::fmt;
use core::mem::{size_of, swap};
use core::ptr;
use std::cmp::Ordering;
use std::iter::FromIterator;
use std::ops::Deref;
use std::ops::DerefMut;
use std::slice;
use std::vec;

pub struct BinaryHeap<T, C = MaxComparator>
where
    C: Compare<T>,
{
    pub data: Vec<T>,
    cmp: C,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct MaxComparator;

impl<T: Ord> Compare<T> for MaxComparator {
    fn compare(&self, a: &T, b: &T) -> Ordering {
        a.cmp(&b)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct MinComparator;

impl<T: Ord> Compare<T> for MinComparator {
    fn compare(&self, a: &T, b: &T) -> Ordering {
        b.cmp(&a)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct FnComparator<F>(pub F);

impl<T, F> Compare<T> for FnComparator<F>
where
    F: Fn(&T, &T) -> Ordering,
{
    fn compare(&self, a: &T, b: &T) -> Ordering {
        self.0(a, b)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct KeyComparator<F>(pub F);

impl<K: Ord, T, F> Compare<T> for KeyComparator<F>
where
    F: Fn(&T) -> K,
{
    fn compare(&self, a: &T, b: &T) -> Ordering {
        self.0(a).cmp(&self.0(b))
    }
}

pub struct PeekMut<'a, T: 'a, C: 'a + Compare<T>> {
    heap: &'a mut BinaryHeap<T, C>,
    sift: bool,
}

impl<'a, T: fmt::Debug, C: Compare<T>> fmt::Debug for PeekMut<'a, T, C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("PeekMut").field(&self.heap.data[0]).finish()
    }
}

impl<'a, T, C: Compare<T>> Drop for PeekMut<'a, T, C> {
    fn drop(&mut self) {
        if self.sift {
            self.heap.sift_down(0);
        }
    }
}

impl<'a, T, C: Compare<T>> Deref for PeekMut<'a, T, C> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.heap.data[0]
    }
}

impl<'a, T, C: Compare<T>> DerefMut for PeekMut<'a, T, C> {
    fn deref_mut(&mut self) -> &mut T {
        self.sift = true;
        &mut self.heap.data[0]
    }
}

impl<'a, T, C: Compare<T>> PeekMut<'a, T, C> {
    pub fn pop(mut this: PeekMut<'a, T, C>) -> T {
        let value = this.heap.pop().unwrap();
        this.sift = false;
        value
    }
}

impl<T: Clone, C: Compare<T> + Clone> Clone for BinaryHeap<T, C> {
    fn clone(&self) -> Self {
        BinaryHeap {
            data: self.data.clone(),
            cmp: self.cmp.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}

impl<T: Ord> Default for BinaryHeap<T> {
    #[inline]
    fn default() -> BinaryHeap<T> {
        BinaryHeap::new()
    }
}

impl<T: fmt::Debug, C: Compare<T>> fmt::Debug for BinaryHeap<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, C: Compare<T> + Default> BinaryHeap<T, C> {
    pub fn from_vec(vec: Vec<T>) -> Self {
        BinaryHeap::from_vec_cmp(vec, C::default())
    }
}

impl<T, C: Compare<T>> BinaryHeap<T, C> {
    pub fn from_vec_cmp(vec: Vec<T>, cmp: C) -> Self {
        unsafe { BinaryHeap::from_vec_cmp_raw(vec, cmp, true) }
    }
    pub unsafe fn from_vec_cmp_raw(vec: Vec<T>, cmp: C, rebuild: bool) -> Self {
        let mut heap = BinaryHeap { data: vec, cmp };
        if rebuild && !heap.data.is_empty() {
            heap.rebuild();
        }
        heap
    }
}

impl<T: Ord> BinaryHeap<T> {
    pub fn new() -> Self {
        BinaryHeap::from_vec(vec![])
    }

    pub fn with_capacity(capacity: usize) -> Self {
        BinaryHeap::from_vec(Vec::with_capacity(capacity))
    }
}

impl<T: Ord> BinaryHeap<T, MinComparator> {
    pub fn new_min() -> Self {
        BinaryHeap::from_vec(vec![])
    }

    pub fn with_capacity_min(capacity: usize) -> Self {
        BinaryHeap::from_vec(Vec::with_capacity(capacity))
    }
}

impl<T, F> BinaryHeap<T, FnComparator<F>>
where
    F: Fn(&T, &T) -> Ordering,
{
    pub fn new_by(f: F) -> Self {
        BinaryHeap::from_vec_cmp(vec![], FnComparator(f))
    }

    pub fn with_capacity_by(capacity: usize, f: F) -> Self {
        BinaryHeap::from_vec_cmp(Vec::with_capacity(capacity), FnComparator(f))
    }
}

impl<T, F, K: Ord> BinaryHeap<T, KeyComparator<F>>
where
    F: Fn(&T) -> K,
{
    pub fn new_by_key(f: F) -> Self {
        BinaryHeap::from_vec_cmp(vec![], KeyComparator(f))
    }

    pub fn with_capacity_by_key(capacity: usize, f: F) -> Self {
        BinaryHeap::from_vec_cmp(Vec::with_capacity(capacity), KeyComparator(f))
    }
}

impl<T, C: Compare<T>> BinaryHeap<T, C> {
    #[inline]
    pub fn replace_cmp(&mut self, cmp: C) {
        unsafe {
            self.replace_cmp_raw(cmp, true);
        }
    }

    pub unsafe fn replace_cmp_raw(&mut self, cmp: C, rebuild: bool) {
        self.cmp = cmp;
        if rebuild && !self.data.is_empty() {
            self.rebuild();
        }
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            iter: self.data.iter(),
        }
    }

    pub fn into_iter_sorted(self) -> IntoIterSorted<T, C> {
        IntoIterSorted { inner: self }
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.get(0)
    }

    pub fn peek_mut(&mut self) -> Option<PeekMut<T, C>> {
        if self.is_empty() {
            None
        } else {
            Some(PeekMut {
                heap: self,
                sift: false,
            })
        }
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop().map(|mut item| {
            if !self.is_empty() {
                swap(&mut item, &mut self.data[0]);
                self.sift_down_to_bottom(0);
            }
            item
        })
    }

    pub fn push(&mut self, item: T) {
        let old_len = self.len();
        self.data.push(item);
        self.sift_up(0, old_len);
    }

    pub fn into_vec(self) -> Vec<T> {
        self.into()
    }

    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut end = self.len();
        while end > 1 {
            end -= 1;
            unsafe {
                let ptr = self.data.as_mut_ptr();
                ptr::swap(ptr, ptr.add(end));
            }
            self.sift_down_range(0, end);
        }
        self.into_vec()
    }

    // The implementations of sift_up and sift_down use unsafe blocks in
    // order to move an element out of the vector (leaving behind a
    // hole), shift along the others and move the removed element back into the
    // vector at the final location of the hole.
    // The `Hole` type is used to represent this, and make sure
    // the hole is filled back at the end of its scope, even on panic.
    // Using a hole reduces the constant factor compared to using swaps,
    // which involves twice as many moves.
    pub fn sift_up(&mut self, start: usize, pos: usize) -> usize {
        unsafe {
            // Take out the value at `pos` and create a hole.
            let mut hole = Hole::new(&mut self.data, pos);

            while hole.pos() > start {
                let parent = (hole.pos() - 1) / 2;
                // if hole.element() <= hole.get(parent) {
                if self.cmp.compare(hole.element(), hole.get(parent)) != Ordering::Greater {
                    break;
                }
                hole.move_to(parent);
            }
            hole.pos()
        }
    }

    /// Take an element at `pos` and move it down the heap,
    /// while its children are larger.
    pub fn sift_down_range(&mut self, pos: usize, end: usize) {
        unsafe {
            let mut hole = Hole::new(&mut self.data, pos);
            let mut child = 2 * pos + 1;
            while child < end - 1 {
                // compare with the greater of the two children
                // if !(hole.get(child) > hole.get(child + 1)) { child += 1 }
                child += (self.cmp.compare(hole.get(child), hole.get(child + 1))
                    != Ordering::Greater) as usize;
                // if we are already in order, stop.
                // if hole.element() >= hole.get(child) {
                if self.cmp.compare(hole.element(), hole.get(child)) != Ordering::Less {
                    return;
                }
                hole.move_to(child);
                child = 2 * hole.pos() + 1;
            }
            if child == end - 1
                && self.cmp.compare(hole.element(), hole.get(child)) == Ordering::Less
            {
                hole.move_to(child);
            }
        }
    }

    pub fn sift_down(&mut self, pos: usize) {
        let len = self.len();
        self.sift_down_range(pos, len);
    }

    /// Take an element at `pos` and move it all the way down the heap,
    /// then sift it up to its position.
    ///
    /// Note: This is faster when the element is known to be large / should
    /// be closer to the bottom.
    pub fn sift_down_to_bottom(&mut self, mut pos: usize) {
        let end = self.len();
        let start = pos;
        unsafe {
            let mut hole = Hole::new(&mut self.data, pos);
            let mut child = 2 * pos + 1;
            while child < end - 1 {
                let right = child + 1;
                // compare with the greater of the two children
                // if !(hole.get(child) > hole.get(right)) { child += 1 }
                child += (self.cmp.compare(hole.get(child), hole.get(right)) != Ordering::Greater)
                    as usize;
                hole.move_to(child);
                child = 2 * hole.pos() + 1;
            }
            if child == end - 1 {
                hole.move_to(child);
            }
            pos = hole.pos;
        }
        self.sift_up(start, pos);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn drain(&mut self) -> Drain<T> {
        Drain {
            iter: self.data.drain(..),
        }
    }

    pub fn clear(&mut self) {
        self.drain();
    }

    fn rebuild(&mut self) {
        let mut n = self.len() / 2;
        while n > 0 {
            n -= 1;
            self.sift_down(n);
        }
    }

    pub fn append(&mut self, other: &mut Self) {
        if self.len() < other.len() {
            swap(self, other);
        }

        if other.is_empty() {
            return;
        }

        #[inline(always)]
        fn log2_fast(x: usize) -> usize {
            8 * size_of::<usize>() - (x.leading_zeros() as usize) - 1
        }

        // `rebuild` takes O(len1 + len2) operations
        // and about 2 * (len1 + len2) comparisons in the worst case
        // while `extend` takes O(len2 * log_2(len1)) operations
        // and about 1 * len2 * log_2(len1) comparisons in the worst case,
        // assuming len1 >= len2.
        #[inline]
        fn better_to_rebuild(len1: usize, len2: usize) -> bool {
            2 * (len1 + len2) < len2 * log2_fast(len1)
        }

        if better_to_rebuild(self.len(), other.len()) {
            self.data.append(&mut other.data);
            self.rebuild();
        } else {
            self.extend(other.drain());
        }
    }
}

/// Hole represents a hole in a slice i.e. an index without valid value
/// (because it was moved from or duplicated).
/// In drop, `Hole` will restore the slice by filling the hole
/// position with the value that was originally removed.
struct Hole<'a, T: 'a> {
    data: &'a mut [T],
    /// `elt` is always `Some` from new until drop.
    elt: Option<T>,
    pos: usize,
}

impl<'a, T> Hole<'a, T> {
    /// Create a new Hole at index `pos`.
    ///
    /// Unsafe because pos must be within the data slice.
    #[inline]
    unsafe fn new(data: &'a mut [T], pos: usize) -> Self {
        debug_assert!(pos < data.len());
        let elt = ptr::read(&data[pos]);
        Hole {
            data,
            elt: Some(elt),
            pos,
        }
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    /// Returns a reference to the element removed.
    #[inline]
    fn element(&self) -> &T {
        self.elt.as_ref().unwrap()
    }

    /// Returns a reference to the element at `index`.
    ///
    /// Unsafe because index must be within the data slice and not equal to pos.
    #[inline]
    unsafe fn get(&self, index: usize) -> &T {
        debug_assert!(index != self.pos);
        debug_assert!(index < self.data.len());
        self.data.get_unchecked(index)
    }

    /// Move hole to new location
    ///
    /// Unsafe because index must be within the data slice and not equal to pos.
    #[inline]
    unsafe fn move_to(&mut self, index: usize) {
        debug_assert!(index != self.pos);
        debug_assert!(index < self.data.len());
        let index_ptr: *const _ = self.data.get_unchecked(index);
        let hole_ptr = self.data.get_unchecked_mut(self.pos);
        ptr::copy_nonoverlapping(index_ptr, hole_ptr, 1);
        self.pos = index;
    }
}

impl<'a, T> Drop for Hole<'a, T> {
    #[inline]
    fn drop(&mut self) {
        // fill the hole again
        unsafe {
            let pos = self.pos;
            ptr::write(self.data.get_unchecked_mut(pos), self.elt.take().unwrap());
        }
    }
}

pub struct Iter<'a, T: 'a> {
    iter: slice::Iter<'a, T>,
}

// #[stable(feature = "collection_debug", since = "1.17.0")]
impl<'a, T: 'a + fmt::Debug> fmt::Debug for Iter<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.iter.as_slice()).finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
// #[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        self.iter.next_back()
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
// impl<'a, T> ExactSizeIterator for Iter<'a, T> {
//     fn is_empty(&self) -> bool {
//         self.iter.is_empty()
//     }
// }

// #[stable(feature = "fused", since = "1.26.0")]
// impl<'a, T> FusedIterator for Iter<'a, T> {}

/// An owning iterator over the elements of a `BinaryHeap`.
///
/// This `struct` is created by the [`into_iter`] method on [`BinaryHeap`][`BinaryHeap`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: struct.BinaryHeap.html#method.into_iter
/// [`BinaryHeap`]: struct.BinaryHeap.html
// #[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct IntoIter<T> {
    iter: vec::IntoIter<T>,
}

// #[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.iter.as_slice())
            .finish()
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back()
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
// impl<T> ExactSizeIterator for IntoIter<T> {
//     fn is_empty(&self) -> bool {
//         self.iter.is_empty()
//     }
// }

// #[stable(feature = "fused", since = "1.26.0")]
// impl<T> FusedIterator for IntoIter<T> {}

// #[unstable(feature = "binary_heap_into_iter_sorted", issue = "59278")]
#[derive(Clone, Debug)]
pub struct IntoIterSorted<T, C: Compare<T>> {
    inner: BinaryHeap<T, C>,
}

// #[unstable(feature = "binary_heap_into_iter_sorted", issue = "59278")]
impl<T, C: Compare<T>> Iterator for IntoIterSorted<T, C> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.pop()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len();
        (exact, Some(exact))
    }
}

/// A draining iterator over the elements of a `BinaryHeap`.
///
/// This `struct` is created by the [`drain`] method on [`BinaryHeap`]. See its
/// documentation for more.
///
/// [`drain`]: struct.BinaryHeap.html#method.drain
/// [`BinaryHeap`]: struct.BinaryHeap.html
// #[stable(feature = "drain", since = "1.6.0")]
// #[derive(Debug)]
pub struct Drain<'a, T: 'a> {
    iter: vec::Drain<'a, T>,
}

// #[stable(feature = "drain", since = "1.6.0")]
impl<'a, T: 'a> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

// #[stable(feature = "drain", since = "1.6.0")]
impl<'a, T: 'a> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back()
    }
}

// #[stable(feature = "drain", since = "1.6.0")]
// impl<'a, T: 'a> ExactSizeIterator for Drain<'a, T> {
//     fn is_empty(&self) -> bool {
//         self.iter.is_empty()
//     }
// }

// #[stable(feature = "fused", since = "1.26.0")]
// impl<'a, T: 'a> FusedIterator for Drain<'a, T> {}

// #[stable(feature = "binary_heap_extras_15", since = "1.5.0")]
impl<T: Ord> From<Vec<T>> for BinaryHeap<T> {
    /// creates a max heap from a vec
    fn from(vec: Vec<T>) -> Self {
        BinaryHeap::from_vec(vec)
    }
}

// #[stable(feature = "binary_heap_extras_15", since = "1.5.0")]
// impl<T, C: Compare<T>> From<BinaryHeap<T, C>> for Vec<T> {
//     fn from(heap: BinaryHeap<T, C>) -> Vec<T> {
//         heap.data
//     }
// }

impl<T, C: Compare<T>> Into<Vec<T>> for BinaryHeap<T, C> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> FromIterator<T> for BinaryHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        BinaryHeap::from(iter.into_iter().collect::<Vec<_>>())
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<T, C: Compare<T>> IntoIterator for BinaryHeap<T, C> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            iter: self.data.into_iter(),
        }
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, C: Compare<T>> IntoIterator for &'a BinaryHeap<T, C> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

// #[stable(feature = "rust1", since = "1.0.0")]
impl<T, C: Compare<T>> Extend<T> for BinaryHeap<T, C> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        // <Self as SpecExtend<I>>::spec_extend(self, iter);
        self.extend_desugared(iter);
    }
}

impl<T, C: Compare<T>> BinaryHeap<T, C> {
    fn extend_desugared<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        self.reserve(lower);

        for elem in iterator {
            self.push(elem);
        }
    }
}

// #[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Copy, C: Compare<T>> Extend<&'a T> for BinaryHeap<T, C> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}
