#![feature(test)]
extern crate owning_ref;

use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::ops::Deref;
use std::vec::Vec;

use self::owning_ref::{OwningHandle, OwningRef};

const DEFAULT_TABLE_CAPACITY: usize = 128;
const DEFAULT_BUCKET_CAPACITY: usize = 128;

struct VersionTable<K, V> {
    head: AtomicUsize,
    bucket: Vec<RwLock<Bucket<K, V>>>,
}

impl<K, V> VersionTable<K, V>
    where
    K: PartialEq,
{
    fn with_capacity(cap: usize) -> VersionTable<K, V> {
        VersionTable {
            head: AtomicUsize::new(0),
            bucket: (0..cap).map(|_| RwLock::new(Bucket::Empty)).collect(),
        }
    }

    fn current_head(&self) -> usize {
        self.head.load(Ordering::Relaxed) % self.bucket.len()
    }

    fn next_head(&self) -> usize {
        self.head.fetch_add(1, Ordering::Relaxed) % self.bucket.len()
    }

    fn scan<F>(&self, f: F) -> RwLockReadGuard<Bucket<K, V>>
        where
        F: Fn(&Bucket<K, V>) -> bool,
    {
        let start = self.current_head();

        for i in 0..self.bucket.len() {
            let idx = (i + start) % self.bucket.len();
            let bucket = self.bucket[idx].read().unwrap();

            if f(&bucket) {
                return bucket;
            }
        }

        self.bucket[start].read().unwrap()
    }

    fn rev_scan<F>(&self, f: F) -> RwLockReadGuard<Bucket<K, V>>
        where
        F: Fn(&Bucket<K, V>) -> bool,
    {
        let start = self.current_head();

        for i in (0..start).rev().chain((start + 1..self.bucket.len()).rev()) {
            let bucket = self.bucket[i].read().unwrap();

            if f(&bucket) {
                return bucket;
            }
        }

        self.bucket[start].read().unwrap()
    }

    fn remove(&self, key: K) -> Option<V> {
        let head = self.next_head();
        let mut bucket = self.bucket[head].write().unwrap();

        mem::replace(&mut *bucket, Bucket::Removed(key)).value()
    }

    fn scan_mut<F>(&self, f: F) -> RwLockWriteGuard<Bucket<K, V>>
        where
        F: Fn(&Bucket<K, V>) -> bool,
    {
        let start = self.current_head();

        for i in 0..self.bucket.len() {
            let idx = (start + i) % self.bucket.len();
            if let Ok(guard) = self.bucket[idx].try_write() {
                if f(&guard) {
                    return guard;
                }
            }
        }

        self.bucket[start].write().unwrap()
    }

    fn update(&self, key: K, value: V) -> Option<V> {
        let idx = self.next_head();
        let mut bucket = self.bucket[idx].write().unwrap();

        mem::replace(&mut *bucket, Bucket::Contains(key, value)).value()
    }

    fn newest(&self, key: &K) -> RwLockReadGuard<Bucket<K, V>> {
        self.rev_scan(|x| match *x {
            Bucket::Contains(ref ckey, _) => ckey == key,
            Bucket::Removed(ref ckey) => ckey == key,
            _ => false,
        })
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for VersionTable<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VersionTable {{ len: {} }}", self.bucket.len())
    }
}

#[derive(Debug, PartialEq)]
enum Bucket<K, V> {
    Empty,
    Removed(K),
    Contains(K, V),
}

impl<K, V> Bucket<K, V> {
    fn is_free(&self) -> bool {
        match *self {
            Bucket::Empty | Bucket::Removed(_) => true,
            _ => false,
        }
    }

    fn value(self) -> Option<V> {
        if let Bucket::Contains(_, v) = self {
            Some(v)
        } else {
            None
        }
    }

    fn key(&self) -> Option<&K> {
        match self {
            &Bucket::Contains(ref ckey, _) => Some(ckey),
            &Bucket::Removed(ref ckey) => Some(ckey),
            _ => None,
        }
    }

    fn value_ref(&self) -> Result<&V, ()> {
        if let Bucket::Contains(_, ref val) = *self {
            Ok(val)
        } else {
            Err(())
        }
    }

    fn key_matches(&self, key: &K) -> bool
        where
        K: PartialEq,
    {
        match *self {
            Bucket::Contains(ref ckey, _) => ckey == key,
            Bucket::Removed(ref ckey) => ckey == ckey,
            _ => false,
        }
    }
}

struct Table<K: PartialEq, V>
    where
    K: PartialEq + Hash,
{
    hash_builder: RandomState,
    buckets: Vec<VersionTable<K, V>>,
}

impl<K, V> Table<K, V>
    where
    K: PartialEq + Hash,
{
    fn with_capacity(cap: usize, bucket_size: usize) -> Table<K, V> {
        Table {
            hash_builder: RandomState::new(),
            buckets: (0..cap)
                .map(|_| VersionTable::with_capacity(bucket_size))
                .collect(),
        }
    }

    fn hash(&self, key: &K) -> usize {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.buckets.len()
    }

    #[inline]
    fn scan<F>(&self, key: &K, matches: F) -> RwLockReadGuard<Bucket<K, V>>
        where
        F: Fn(&Bucket<K, V>) -> bool,
    {
        self.buckets[self.hash(key)].scan(matches)
    }

    #[inline]
    fn scan_mut<F>(&self, key: &K, matches: F) -> RwLockWriteGuard<Bucket<K, V>>
        where
        F: Fn(&Bucket<K, V>) -> bool,
    {
        self.buckets[self.hash(key)].scan_mut(matches)
    }

    fn lookup(&self, key: &K) -> RwLockReadGuard<Bucket<K, V>> {
        self.buckets[self.hash(key)].newest(key)
    }

    fn remove(&self, key: K) -> Option<V> {
        self.buckets[self.hash(&key)].remove(key)
    }

    fn lookup_or_free_mut(&self, key: &K) -> RwLockWriteGuard<Bucket<K, V>> {
        self.scan_mut(key, |x| match *x {
            Bucket::Contains(ref ckey, _) => ckey == key,
            Bucket::Removed(ref ckey) => ckey == key,
            Bucket::Empty => true,
        })
    }

    fn lookup_mut(&self, key: &K) -> RwLockWriteGuard<Bucket<K, V>> {
        self.scan_mut(key, |x| {
            if let &Bucket::Contains(ref ckey, _) = x {
                ckey == key
            } else {
                false
            }
        })
    }
}

pub struct ReadGuard<'a, K: 'a + PartialEq + Hash, V: 'a> {
    inner: OwningRef<
            OwningHandle<RwLockReadGuard<'a, Table<K, V>>, RwLockReadGuard<'a, Bucket<K, V>>>,
        V,
        >,
}

impl<'a, K: Hash + PartialEq, V: fmt::Display> fmt::Debug for ReadGuard<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ReadGard: {}", *self.inner)
    }
}

impl<'a, K, V> Deref for ReadGuard<'a, K, V>
    where
    K: PartialEq + Hash,
{
    type Target = V;

    fn deref(&self) -> &V {
        &*self.inner
    }
}

impl<'a, K, V: PartialEq> PartialEq for ReadGuard<'a, K, V>
    where
    K: PartialEq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl<'a, K, V: PartialEq> Eq for ReadGuard<'a, K, V>
    where
    K: PartialEq + Hash,
{
}

pub struct LoanMap<K, V>
    where
    K: PartialEq + Hash,
{
    table: RwLock<Table<K, V>>,
}

impl<K, V> LoanMap<K, V>
    where
    K: PartialEq + Hash,
{
    /// Allocates a new `LoanMap` with the default number of buckets
    pub fn new() -> LoanMap<K, V> {
        LoanMap::with_capacity(DEFAULT_TABLE_CAPACITY, DEFAULT_BUCKET_CAPACITY)
    }

    /// Allocates a new `LoanMap` with `cap` buckets
    /// Take great care to allocate enough buckets since resizing the map is *very* costly
    pub fn with_capacity(cap: usize, bucket_size: usize) -> LoanMap<K, V> {
        LoanMap {
            table: RwLock::new(Table::with_capacity(cap, bucket_size)),
        }
    }

    pub fn get(&self, key: &K) -> Option<ReadGuard<K, V>> {
        if let Ok(inner) = OwningRef::new(OwningHandle::new_with_fn(
            self.table.read().unwrap(),
            |x| unsafe { &*x }.lookup(key),
        )).try_map(|x| x.value_ref())
        {
            Some(ReadGuard { inner: inner })
        } else {
            None
        }
    }

    pub fn upsert<F>(&self, key: K, f: F) -> Option<V>
        where
        F: FnOnce(&V) -> V,
    {
        let guard = self.table.read().unwrap();
        let mut bucket = guard.lookup_mut(&key);
        let new_val = {
            let value = bucket.value_ref().expect("key not found");
            f(value)
        };

        mem::replace(&mut *bucket, Bucket::Contains(key, new_val)).value()
    }

    pub fn put(&self, key: K, val: V) -> Option<V> {
        let lock = self.table.read().unwrap();
        let mut bucket = lock.lookup_or_free_mut(&key);

        mem::replace(&mut *bucket, Bucket::Contains(key, val)).value()
    }

    pub fn remove(&self, key: K) -> Option<V> {
        self.table.read().unwrap().remove(key)
    }
}

#[cfg(test)]
mod tests {
    extern crate crossbeam_utils;
    extern crate rand;
    extern crate test;

    use std::fs::File;
    use std::io::Write;
    use std::time::{Duration, SystemTime};
    use std::iter;
    use std::thread;

    use self::crossbeam_utils::scoped::ScopedJoinHandle;
    use self::crossbeam_utils::scoped;
    use self::rand::distributions::Alphanumeric;
    use self::rand::Rng;
    use self::test::Bencher;

    use super::Bucket;
    use super::LoanMap;
    use super::VersionTable;

    /// Generates a random map with `set_size` elements and keys of size `key_size`
    fn gen_rand_map(set_size: usize, key_size: usize) -> (LoanMap<String, String>, Vec<String>) {
        let map: LoanMap<String, String> = LoanMap::with_capacity(16384, 16);
        let mut rng = rand::thread_rng();

        let keys: Vec<String> = (0..set_size)
            .map(|_| {
                iter::repeat(())
                    .map(|()| rng.sample(Alphanumeric))
                    .take(key_size)
                    .collect::<String>()
            })
            .collect();

        for key in &keys {
            map.put(
                key.clone(),
                iter::repeat(())
                    .map(|()| rng.sample(Alphanumeric))
                    .take(1024)
                    .collect::<String>(),
            );
        }

        (map, keys)
    }

    /// Generates a vector of random strings
    fn gen_rand_strings(size: usize, esize: usize) -> Vec<String> {
        let mut rng = rand::thread_rng();

        (0..size)
            .map(|_| {
                iter::repeat(())
                    .map(|()| rng.sample(Alphanumeric))
                    .take(esize)
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn insert_then_get() {
        let map: LoanMap<String, String> = LoanMap::new();

        let (key, value) = (String::from("i'm a key"), String::from("i'm a value"));

        assert_eq!(map.put(key.clone(), value.clone()), None);

        let val = map.get(&key).expect("no value in map after insert");
        assert_eq!(*val, value);
    }

    #[test]
    fn insert_then_remove() {
        let map: LoanMap<String, String> = LoanMap::new();
        let (key, value) = (String::from("key1"), String::from("value1"));

        assert_eq!(map.put(key.clone(), value.clone()), None);
        map.remove(key.clone());

        assert_eq!(map.get(&key), None);
    }

    #[test]
    fn get_inexistent_value() {
        let map: LoanMap<String, String> = LoanMap::new();

        assert_eq!(map.get(&String::from("some key")), None);
    }

    #[test]
    fn version_table_multiple_removes() {
        let capacity = 16;
        let mut rng = rand::thread_rng();
        let table = VersionTable::with_capacity(capacity);
        let keys: Vec<u32> = (0..capacity)
            .map(|_| rng.gen::<u32>())
            .collect();

        for i in 0..keys.len() {
            assert_eq!(table.update(keys[i], i), None);
        }

        for i in 0..keys.len() {
            table.remove(keys[i]);
        }

        for i in 0..keys.len() {
            let key = keys[i];
            assert_eq!(*table.newest(&key), Bucket::Removed(key));
        }
    }

    #[test]
    fn concurrent_puts() {
        let mut rng = rand::thread_rng();
        let keys: Vec<String> = (0..8192)
            .map(|_| {
                iter::repeat(())
                    .map(|()| rng.sample(Alphanumeric))
                    .take(128)
                    .collect::<String>()
            })
            .collect();
        let map: LoanMap<String, u32> = LoanMap::with_capacity(keys.len() * 2, 16);
        const THREAD_NUM: i64 = 16;
        const NUM_PUT: i64 = 8192 * 2;
        const NUM_GET: i64 = 19 * NUM_PUT;

        for i in &keys {
            map.put(i.clone(), 0);
        }

        scoped::scope(|s| {
            let mut guards: Vec<ScopedJoinHandle<()>> = Vec::new();

            for i in 0..THREAD_NUM {
                let idx = i.clone();
                let mut filename = File::create(format!("{}.csv", idx).to_string()).unwrap();

                if i % 2 == 0 {
                    guards.push(s.spawn(|| {
                        let mut rng = rand::thread_rng();
                        let mut file = filename;
                        let mut results: Vec<u32> = Vec::new();

                        for i in 0..NUM_PUT {
                            let key = keys[i as usize % keys.len()].clone();
                            let mut start;

                            {
                                start = SystemTime::now();
                                map.put(key, rng.gen::<u32>());
                            }

                            results.push(start.elapsed().unwrap().subsec_nanos());
                        }

                        for time in results {
                            write!(file, "PUT;{}\n", time).unwrap();
                        }
                    }));
                } else {
                    guards.push(s.spawn(|| {
                        let mut file = filename;
                        let mut results: Vec<u32> = Vec::with_capacity(NUM_GET as usize);

                        for i in 0..NUM_GET {
                            let key = &keys[i as usize % keys.len()];
                            let start = SystemTime::now();

                            {
                                let _guard = map.get(key).expect("missing key");
                                results.push(start.elapsed().unwrap().subsec_nanos());
                                // Hold the lock for "network" sending
                                thread::sleep(Duration::new(0, 500));
                            }
                        }

                        for time in results {
                            write!(file, "GET;{}\n", time).unwrap();
                        }
                    }));
                }
            }
        });
    }

    #[bench]
    fn bench_huge_values(b: &mut Bencher) {
        let set_size = 512;
        let item_size = 4096;
        let values = gen_rand_strings(set_size, item_size);
        let keys = gen_rand_strings(set_size, item_size / 4);
        let map: LoanMap<String, String> = LoanMap::with_capacity(set_size * 2, 16);
        let mut i = 0;

        keys.iter().zip(values.iter()).for_each(|(k, v)| {
            map.put(k.clone(), v.clone());
        });

        b.iter(move || {
            let idx = i % set_size;
            assert_eq!(*map.get(&keys[idx]).expect("missing key"), values[idx]);
            i += 1;
        })
    }

    #[test]
    fn double_insert_updates_value() {
        let map: LoanMap<String, String> = LoanMap::new();

        let key = String::from("i'm a key");
        let value1 = String::from("i'm the first value");
        let value2 = String::from("i'm the second value");

        map.put(key.clone(), value1);
        map.put(key.clone(), value2.clone());

        let from_map = map.get(&key).expect("map does not insert");

        assert_eq!(*from_map, value2);
    }

    #[test]
    fn delete_value() {
        let map: LoanMap<String, String> = LoanMap::new();
        let key = String::from("key1");
        let value = String::from("value");

        map.put(key.clone(), value);
        map.remove(key.clone());

        assert_eq!(map.get(&key), None);
    }

    #[test]
    fn version_table_delete_value() {
        let table: VersionTable<String, u32> = VersionTable::with_capacity(16);
        let key = String::from("random key");
        let value = 1024;

        assert_eq!(table.update(key.clone(), value.clone()), None);
        table.remove(key.clone());
        assert_eq!(table.newest(&key).value_ref(), Err(()));
    }

    #[test]
    fn version_table_values_dont_interfere() {
        let table: VersionTable<String, String> = VersionTable::with_capacity(128);
        let mut rng = rand::thread_rng();
        let keys: Vec<String> = (0..10)
            .map(|_| {
                iter::repeat(())
                    .map(|()| rng.sample(Alphanumeric))
                    .take(128)
                    .collect::<String>()
            })
            .collect();

        let values: Vec<String> = keys.iter()
            .map(|_| {
                iter::repeat(())
                    .map(|()| rng.sample(Alphanumeric))
                    .take(128)
                    .collect::<String>()
            })
            .collect();

        for i in 0..keys.len() {
            table.update(keys[i].clone(), values[i].clone());
        }

        for i in 0..keys.len() {
            assert_eq!(
                table
                    .newest(&keys[i])
                    .value_ref(),
                Ok(&values[i])
            );
        }
    }

    #[test]
    fn version_table_multi_version_multi_key() {
        let size = 256;
        let table: VersionTable<String, String> = VersionTable::with_capacity(size);
        let keys = gen_rand_strings(size, size);
        let values = gen_rand_strings(2 * size, size);

        // insert first set of values
        for (key, value) in keys.iter().zip(values.iter()).take(size) {
            table.update(key.clone(), value.clone());
        }

        // second round
        for (key, value) in keys.iter().zip(values.iter().skip(size)) {
            table.update(key.clone(), value.clone());
            let opt = table.newest(&key);
            assert_eq!(opt.value_ref(), Ok(value));
        }
    }

    #[test]
    fn version_table_update_single_key() {
        let test_size = 512;
        let table: VersionTable<String, String> = VersionTable::with_capacity(127);
        let mut rng = rand::thread_rng();
        let key = iter::repeat(())
            .map(|_| rng.sample(Alphanumeric))
            .take(128)
            .collect::<String>();
        let values = gen_rand_strings(test_size, 128);

        for value in &values {
            table.update(key.clone(), value.clone());
            assert_eq!(
                table.newest(&key).value_ref(),
                Ok(value)
            );
        }
    }

    #[bench]
    fn bench_access_string_key_string_value(b: &mut Bencher) {
        let mut rng = rand::thread_rng();
        let set_size = 8192;
        let (map, keys) = gen_rand_map(8192 * 2, 128);

        b.iter(|| {
            let key = &keys[rng.gen::<usize>() % set_size];
            assert_eq!(map.get(key).is_some(), true);
        })
    }

    #[bench]
    fn bench_access_u32_key_u32_value(b: &mut Bencher) {
        let mut rng = rand::thread_rng();
        let ssize: usize = 8192;
        let keys: Vec<u32> = (0..ssize).map(|_| rng.gen::<u32>()).collect();
        let map: LoanMap<u32, u32> = LoanMap::with_capacity(2 * ssize, 16);

        for key in &keys {
            map.put(*key, rng.gen::<u32>());
        }

        b.iter(|| {
            let key = &keys[rng.gen::<usize>() % ssize];
            assert_eq!(map.get(key).is_some(), true);
        })
    }
}
