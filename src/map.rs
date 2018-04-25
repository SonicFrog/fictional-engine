use std::collections::hash_map::DefaultHasher;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::vec::Vec;

pub struct ConcurrentHashMap<K, V>
where
    K: Hash + Eq + Clone + Display,
    V: Display + Clone,
{
    hasher: DefaultHasher,
    bucket_count: usize,
    buckets: Vec<RwLock<Bucket<K, V>>>,
}

struct Bucket<K, V>
where
    K: Hash + PartialEq + Eq + Clone + Display,
    V: Sized + Display + Clone,
{
    values: Vec<(K, V)>,
}

impl<K, V> ConcurrentHashMap<K, V>
where
    K: Hash + PartialEq + Eq + Clone + Display,
    V: Sized + Display + Clone,
{
    pub fn new(bucket_count: usize, bucket_size: usize) -> ConcurrentHashMap<K, V> {
        let mut map = ConcurrentHashMap {
            bucket_count: bucket_count,
            hasher: DefaultHasher::new(),
            buckets: Vec::with_capacity(bucket_count),
        };

        for _i in 0..bucket_count {
            map.buckets.push(RwLock::new(Bucket{
                values: Vec::with_capacity(bucket_size),
            }));
        }

        map
    }

    fn hash(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.bucket_count
    }

    #[inline]
    fn find_bucket_ro(&self, key: &K) -> RwLockReadGuard<Bucket<K, V>> {
        self.buckets[self.hash(key)].read().unwrap()
    }

    #[inline]
    fn find_bucket_rw(&self, key: &K) -> RwLockWriteGuard<Bucket<K, V>>{
        self.buckets[self.hash(key)].write().unwrap()
    }

    pub fn insert(&self, key: &K, value: V) {
        let mut bucket = self.find_bucket_rw(key);

        bucket.values.push((key.clone(), value));
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let bucket = self.find_bucket_ro(key);

        bucket.values
            .iter()
            .find(move |kv|
                  kv.0 == *key
            )
            .map(|kv| kv.1.clone())
    }

    pub fn remove(&self, key: &K) {
        let mut bucket = self.find_bucket_rw(key);

        for i in 0..bucket.values.len() {
            if bucket.values[i].0 == *key {
                bucket.values.remove(i);
                return;
            }
        }
    }
}
