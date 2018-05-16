extern crate abox;

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::ops::Deref;
use std::vec::Vec;

use self::abox::AtomicBox;

const DEFAULT_TABLE_CAPACITY: usize = 128;
const DEFAULT_BUCKET_CAPACITY: usize = 128;

/// A concurrent HashMap that uses optimistic concurrency control
/// (hence the name)
pub struct OptiMap<K, V, S>
    where
    K: PartialEq + Hash,
    S: BuildHasher,
{
    table: AtomicTable<K, V, S>,
}

impl<K, V> OptiMap<K, V, RandomState>
    where K: PartialEq + Hash,
{
    /// Creates a new `OptiMap` with default number of buckets and hasher
    pub fn new() -> Self {
        OptiMap {
            table: AtomicTable::new(),
        }
    }

    /// Creates a new `OptiMap` with `cap` buckets
    pub fn with_capacity(cap: usize) -> Self {
        OptiMap::with_capacity_and_hasher(cap, RandomState::new())
    }
}

impl<K, V, S> OptiMap<K, V, S>
    where K: PartialEq + Hash,
          S: BuildHasher,
{
    /// Create a new `OptiMap` that uses the given hasher for hashing keys
    pub fn with_hasher(hasher: S) -> Self {
        OptiMap {
            table: AtomicTable::with_hasher(hasher),
        }
    }

    /// Creates a new `OptiMap` with `cap` collision bucket and the given Hasher
    pub fn with_capacity_and_hasher(cap: usize, hasher: S) -> Self {
        OptiMap {
            table: AtomicTable::with_capacity_and_hasher(cap, hasher),
        }
    }

    /// Gets the value associated with the given key if it exists
    pub fn get(&self, key: &K) -> Option<ValueHolder<K, V>> {
        self.table.get(key).map(|x| ValueHolder { bucket: Arc::clone(&x) })
    }

    /// Insert a new entry in the `OptiMap`.
    /// Note that the map takes ownership of both the key and the value
    pub fn put(&self, key: K, value: V) {
        self.table.put(key, value)
    }

    pub fn delete(&self, key: &K) {
        self.table.delete(key)
    }
}

struct StatCollector {
    full_search: AtomicUsize,
    tx_fail: AtomicUsize,
}

#[derive(Debug)]
/// A struct used to hold the bucket for as long as the value is needed
pub struct ValueHolder<K, V> {
    bucket: Arc<BucketEntry<K, V>>
}

impl<K, V> Deref for ValueHolder<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.bucket.value_ref()
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for ValueHolder<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

struct AtomicVersionTable<K, V> {
    value: AtomicBox<Vec<Arc<BucketEntry<K, V>>>>,
}

impl<K: PartialEq, V> AtomicVersionTable<K, V> {
    fn new() -> AtomicVersionTable<K, V> {
        AtomicVersionTable {
            value: AtomicBox::new(Vec::new()),
        }
    }

    #[inline]
    fn put(&self, key: K, value: V) {
        let bucket = Arc::new(BucketEntry::new(key, value));

        self.value.replace_with(move |x| {
            let mut y = x.clone();


            for i in 0..y.len() {
                if y[i].key_matches(bucket.key()) {
                    y[i] = bucket.clone();
                    return y;
                }
            }

            y.push(bucket.clone());
            y
        });
    }

    #[inline]
    fn delete(&self, key: &K) {
        self.value.replace_with(move |x| {
            let mut y = x.clone();

            for i in 0..y.len() {
                if y[i].key_matches(key) {
                    // FIXME: should not shift all elements in the vector
                    y.remove(i);
                    break;
                }
            }

            y
        })
    }

    #[inline]
    fn get(&self, key: &K) -> Option<Arc<BucketEntry<K, V>>> {
        self.find(|x| x.key_matches(key))
    }

    #[inline]
    fn find<F>(&self, f: F) -> Option<Arc<BucketEntry<K, V>>>
        where F: Fn(&Arc<BucketEntry<K, V>>) -> bool
    {
        self.value.iter().cloned().find(f)
    }
}

impl<K, V> Clone for AtomicVersionTable<K, V> {
    fn clone(&self) -> Self {
        AtomicVersionTable {
            value: self.value.clone(),
        }
    }
}



struct AtomicTable<K, V, S>
    where
    K: PartialEq + Hash,
    S: BuildHasher,
{
    buckets: Vec<AtomicVersionTable<K, V>>,
    hash_builder: S,
}

const DEFAULT_BUCKET_NUMBER: usize = 128;

impl<K, V> AtomicTable<K, V, RandomState>
    where
    K: PartialEq + Hash,
{

    fn new() -> AtomicTable<K, V, RandomState> {
        AtomicTable::with_capacity(DEFAULT_BUCKET_NUMBER)
    }

    /// Instantiate a new AtomicTable with `cap` buckets
    fn with_capacity(cap: usize) -> AtomicTable<K, V, RandomState> {
        AtomicTable::with_capacity_and_hasher(cap, RandomState::new())
    }
}

impl<K, V, S> AtomicTable<K, V, S>
    where
    K: PartialEq + Hash,
    S: BuildHasher,
{
    fn with_hasher(hasher: S) -> AtomicTable<K, V, S> {
        AtomicTable::with_capacity_and_hasher(DEFAULT_BUCKET_NUMBER, hasher)
    }

    fn with_capacity_and_hasher(cap: usize, hasher: S) -> AtomicTable<K, V, S> {
        AtomicTable {
            buckets: (0..cap).map(|_| AtomicVersionTable::new()).collect(),
            hash_builder: hasher,
        }
    }

    fn hash(&self, key: &K) -> usize {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.buckets.len()
    }

    #[inline]
    fn find_bucket(&self, key: &K) -> AtomicVersionTable<K, V>  {
        self.buckets[self.hash(key)].clone()
    }

    #[inline]
    fn scan<F>(&self, key: &K, matches: F) -> Option<Arc<BucketEntry<K, V>>>
        where F: Fn(&Arc<BucketEntry<K, V>>) -> bool,
    {
        self.find_bucket(key).find(matches)
    }

    fn put(&self, key: K, value: V) {
        self.find_bucket(&key).put(key, value);
    }

    fn delete(&self, key: &K) {
        self.find_bucket(key).delete(key);
    }

    fn get(&self, key: &K) -> Option<Arc<BucketEntry<K, V>>> {
        self.scan(key, |x| x.key_matches(key))
    }
}

#[derive(Debug, PartialEq)]
struct BucketEntry<K, V> {
    key: K,
    value: V,
}

impl<K, V> BucketEntry<K, V> {
    fn new(key: K, value: V) -> BucketEntry<K, V> {
        BucketEntry {
            key,
            value,
        }
    }

    fn value(self) -> V {
        self.value
    }

    fn key(&self) -> &K {
        &self.key
    }

    fn value_ref(&self) -> &V {
        &self.value
    }

    fn key_matches(&self, key: &K) -> bool
        where
        K: PartialEq,
    {
        &self.key == key
    }
}

#[cfg(test)]
pub mod tests {
    extern crate core_affinity;
    extern crate crossbeam_utils;
    extern crate rand;
    extern crate test;

    use std::collections::hash_map::RandomState;
    use std::fs::File;
    use std::io::Write;
    use std::time::SystemTime;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use self::rand::Rng;
    use self::test::Bencher;

    use logmap::tests::gen_rand_strings;

    use super::{AtomicVersionTable, AtomicTable, OptiMap};

    #[test]
    fn atomic_version_table_insert_then_get() {
        let table: AtomicVersionTable<String, u32> = AtomicVersionTable::new();
        let key = String::from("keyt");
        let value = 1023;

        table.put(key.clone(), value.clone());
        let bucket = table.get(&key).expect("failed to insert value");

        assert_eq!(bucket.key_matches(&key), true);
        assert_eq!(bucket.value_ref(), &value);
    }

    #[test]
    fn atomic_table_put_then_get() {
        let table = AtomicTable::with_capacity(128);
        let key = String::from("key1");
        let value = String::from("value1");

        table.put(key.clone(), value.clone());

        let out = table.get(&key).expect("value was not inserted");

        assert_eq!(out.key_matches(&key), true);
        assert_eq!(out.value_ref(), &value);
    }

    #[test]
    fn atomic_table_overwrite_value() {
        let table = AtomicTable::new();
        let key = String::from("key1");
        let (v1, v2) = (String::from("some value"), String::from("another value"));

        table.put(key.clone(), v1.clone());
        table.put(key.clone(), v2.clone());

        let bucket = table.get(&key).expect("value was not inserted");

        assert_eq!(bucket.key_matches(&key), true);
        assert_eq!(bucket.value_ref(), &v2);
    }

    #[test]
    fn atomic_table_remove() {
        let table = AtomicTable::new();
        let key = String::from("key1");
        let value = String::from("v2");

        table.put(key.clone(), value);

        table.delete(&key);

        assert_eq!(table.get(&key), None);
    }

    #[test]
    fn optimap_empty() {
        let map: OptiMap<String, String, RandomState> = OptiMap::new();

        assert_eq!(map.get(&String::from("key")), None);
    }

    #[test]
    fn optimap_put_then_get() {
        let map = OptiMap::new();
        let key = String::from("k");
        let value = String::from("v");

        map.put(key.clone(), value.clone());

        let v = map.get(&key).expect("failed to insert value");

        assert_eq!(*v, value);
    }

    #[bench]
    fn optimap_bench_access_str_key_str_value(b: &mut Bencher) {
        let value_count = 8192;
        let key_size = 128;
        let value_size = 1024;
        let map = OptiMap::with_capacity(2 * value_count);
        let kvs: Vec<(String, String)> = gen_rand_strings(value_count, key_size)
            .iter().cloned()
            .zip(
                gen_rand_strings(value_count, value_size)
                    .iter()
                    .cloned())
            .collect();

        for (key, value) in &kvs {
            map.put(key.clone(), value.clone());
        }

        let mut rng = rand::thread_rng();

        b.iter(move || {
            let idx = rng.gen::<usize>() % &kvs.len();
            let vh = map.get(&kvs[idx].0).expect("value was not inserted");

            assert_eq!(*vh, kvs[idx].1);
        })
    }

    #[bench]
    fn optimap_bench_put_str(b: &mut Bencher) {
        let value_count = 8192;
        let key_size = 128;
        let value_size = 1024;
        let map = OptiMap::with_capacity(2 * value_count);
        let kvs: Vec<(String, String)> = gen_rand_strings(value_count, key_size)
            .iter().cloned()
            .zip(
                gen_rand_strings(value_count, value_size)
                    .iter()
                    .cloned())
            .collect();
        let mut rng = rand::thread_rng();

        b.iter(move || {
            let (key, value) = kvs[rng.gen::<usize>() % kvs.len()].clone();
            map.put(key, value);
        })
    }

    #[bench]
    fn optimap_bench_put_u64(b: &mut Bencher) {
        let value_count = 8192;
        let map = OptiMap::with_capacity(2 * value_count);
        let mut rng = rand::thread_rng();

        let keys: Vec<u64> = (0..value_count)
            .map(|_| rng.gen::<u64>())
            .collect();
        let values: Vec<u64> = (0..value_count)
            .map(|_| rng.gen::<u64>())
            .collect();

        let kvs: Vec<(u64, u64)> = keys.iter().cloned()
            .zip(values.iter().cloned()).collect();

        let mut i = 0;

        b.iter(move || {
            map.put(kvs[i].0, kvs[i].1);
            i = (i + 1) % kvs.len();
        })
    }

    #[test]
    // TODO: improve this by balancing puts and gets across cores
    fn optimap_core_affinity_u128_bench() {
        let value_count = 8192;
        let map = Arc::new(OptiMap::with_capacity(value_count * 4));
        let mut rng = rand::thread_rng();

        // test parameters
        let put_count: usize = 1000000;
        let get_count: usize = 1000000;
        let pg_selector = |x| x % 2 == 0;

        let kvs: Vec<(u64, u64)> = (0..value_count)
            .map(|_| (rng.gen::<u64>(), rng.gen::<u64>()))
            .collect();

        kvs.iter().cloned().for_each(|(k, v)| map.put(k, v));
        let tid = Arc::new(AtomicUsize::new(0));
        let core_ids = core_affinity::get_core_ids().unwrap();

        let handles = core_ids.into_iter()
            .skip(1) // leave one core for OS to do some bookkeeping
            .map(|id| {
                let tid = tid.clone();
                let kvs = kvs.clone();
                let map = Arc::clone(&map);

                thread::spawn(move || {
                    core_affinity::set_for_current(id);
                    let id = (&tid).fetch_add(1, Ordering::Relaxed);
                    let filename = format!("{}.csv", id);
                    let mut file = File::create(filename).unwrap();
                    let mut rng = rand::thread_rng();
                    let mut results = Vec::with_capacity(put_count);

                    if pg_selector(id) {
                        for i in 0..put_count {
                            let idx = rng.gen::<usize>() % kvs.len();
                            let kv = kvs[idx].clone();
                            let start = SystemTime::now();

                            {
                                map.put(kv.0, kv.1);
                            }

                            let elapsed = start.elapsed().unwrap().subsec_nanos();

                            results.push(elapsed);
                        }

                        for result in results {
                            write!(file, "PUT;{}\n", result).unwrap();
                        }
                    } else {
                        for i in 0..get_count {
                            let idx = rng.gen::<usize>() % kvs.len();
                            let kv = kvs[idx].clone();
                            let start = SystemTime::now();
                            let mut elapsed;

                            {
                                let _lock = map.get(&kv.0);

                                elapsed = start.elapsed().unwrap().subsec_nanos();
                            }

                            results.push(elapsed);
                        }

                        for result in results {
                            write!(file, "GET;{}\n", result).unwrap();
                        }
                    }

                })
            }).collect::<Vec<_>>();

        handles.into_iter().for_each(|h| h.join().unwrap());
    }
}
