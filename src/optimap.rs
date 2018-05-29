extern crate abox;

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::ops::Deref;
use std::vec::Vec;

use self::abox::AtomicBox;

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
    value: Arc<AtomicBox<Vec<Arc<BucketEntry<K, V>>>>>,
}

impl<K: PartialEq, V> AtomicVersionTable<K, V> {
    fn new() -> AtomicVersionTable<K, V> {
        AtomicVersionTable {
            value: Arc::new(AtomicBox::new(Vec::new())),
        }
    }

    #[inline]
    fn put(&self, key: K, value: V) {
        let bucket = Arc::new(BucketEntry::new(key, value));

        self.value.replace_with(move |x| {
            let mut y = (*x).clone();

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
            let mut y = Vec::new();

            for i in 0..x.len() {
                if !x[i].key_matches(key) {
                    y.push(Arc::clone(&x[i]));
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
        where F: FnMut(&Arc<BucketEntry<K, V>>) -> bool,
    {
        self.value.get().iter().cloned().find(f)
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
    use std::fmt;
    use std::fs::File;
    use std::hash::Hash;
    use std::io::Write;
    use std::marker::PhantomData;
    use std::time::SystemTime;
    use std::sync::{Arc, Barrier, RwLock};

    use self::crossbeam_utils::scoped::ScopedJoinHandle;
    use self::crossbeam_utils::scoped;
    use self::rand::Rng;
    use self::rand::distributions::{Distribution, Standard};
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

        assert!(out.key_matches(&key));
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

    enum Task<K, V> {
        Put(K, V),
        Get(K),
        Remove(K),
    }

    impl<K, V> fmt::Display for Task<K, V> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let val = match self {
                Task::Put(_, _) => "PUT",
                Task::Get(_) => "GET",
                Task::Remove(_) => "DEL",
            };

            write!(f, "{}", val)
        }
    }

    impl<K: Clone, V: Clone> Clone for Task<K, V> {
        fn clone(&self) -> Task<K, V> {
            match self {
                Task::Put(key, value) => Task::Put(key.clone(), value.clone()),
                Task::Get(ref key) => Task::Get(key.clone()),
                Task::Remove(ref key) => Task::Remove(key.clone()),
            }
        }
    }

    struct BenchRunner<K, V> {
        p_k: PhantomData<K>,
        p_v: PhantomData<V>,
    }

    impl<K, V> BenchRunner<K, V>
        where K: Clone + Hash + PartialEq + Send + Sync,
              V: Clone + Send + Sync,
    {
        #[inline]
        fn execute_timed(map: &OptiMap<K, V, RandomState>, task: Task<K, V>) -> u32 {
            let start = SystemTime::now();

            BenchRunner::execute(map, task);

            start.elapsed().unwrap().subsec_nanos()
        }

        #[inline]
        fn execute(map: &OptiMap<K, V, RandomState>, task: Task<K, V>) {
            match task {
                Task::Put(key, value) => map.put(key, value),
                Task::Get(key) => {map.get(&key);},
                Task::Remove(key) => map.delete(&key),
            };
        }

        fn run_benchmark(name: &str, buckets: usize, workload: Vec<Arc<Vec<Task<K, V>>>>,
                         initial: Vec<Task<K, V>>) {
            let map = Arc::new(OptiMap::with_capacity(buckets));
            let core_ids = core_affinity::get_core_ids().unwrap();
            let results: Arc<RwLock<Vec<Vec<_>>>> = Arc::new(RwLock::new(Vec::new()));

            // build initial map
            initial.iter().cloned().for_each(|t| BenchRunner::execute(&map, t));

            scoped::scope(|ctx| {
                let results = Arc::clone(&results);
                let rd = Arc::clone(&results);
                let barrier = Arc::new(Barrier::new(core_ids.len() - 2));

                let guards: Vec<ScopedJoinHandle<_>> = core_ids.into_iter().zip(workload.iter().cloned())
                    .skip(1)
                    .map(|(id, work)| {
                        core_affinity::set_for_current(id);
                        let results = Arc::clone(&results);
                        let map = Arc::clone(&map);
                        let b = Arc::clone(&barrier);

                        ctx.spawn(move || {
                            b.wait();
                            let times = work
                                .iter()
                                .cloned()
                                .map(|t| (t.clone(), BenchRunner::execute_timed(&*map, t)))
                                .collect();
                            results.write().unwrap().push(times);
                        })
                    }).collect();

                for g in guards {
                    g.join().unwrap();
                }

                let mut file = File::create(name).unwrap();

                rd.read().unwrap().iter()
                    .for_each(|res| {
                        res.iter().for_each(|(t, duration)| {
                            write!(file, "{};{}\n", t, duration).unwrap();
                        })
                    });
            });
        }
    }

    fn gen_workload<K: PartialEq + Clone, V>(keys: Vec<K>, put: usize,
                                             get: usize, del: usize) -> Vec<Arc<Vec<Task<K, V>>>>
        where
        Standard: Distribution<V>,
    {
        let mut rng = rand::thread_rng();
        let mut workloads = core_affinity::get_core_ids().unwrap().into_iter()
            .skip(1) // leave one core for Linux
            .map(|_| {
                let tasks = (0..(put + get + del)).map(|i| {
                    let key = keys[rng.gen::<usize>() % keys.len()].clone();

                    if i < put {
                        Task::Get(key)
                    } else if i < put + get {
                        Task::Put(key, rng.gen::<V>())
                    } else {
                        Task::Remove(key)
                    }
                }).collect::<Vec<_>>();

                Arc::new(tasks)
            }).collect::<Vec<Arc<_>>>();

        for mut workload in &mut workloads {
            let mut slice = Arc::get_mut(workload).unwrap().as_mut_slice();
            rng.shuffle(slice);
        }

        workloads
    }

    #[test]
    fn optimap_read_heavy_benchmark() {
        let key_count = 8192;
        let esize = 1024;
        let keys = gen_rand_strings(key_count, esize);
        let put_count = 500000;
        let get_count = 1000000;
        let del_count = 5000;

        let mut rng = rand::thread_rng();
        let init = keys.iter().cloned().map(|k| Task::Put(k, rng.gen::<u64>())).collect();
        let workloads = gen_workload::<String, u64>(keys, put_count, get_count, del_count);

        BenchRunner::run_benchmark("read_heavy.csv", key_count, workloads, init);
    }

    #[test]
    fn optimap_no_collision_benchmark() {
        let key_size = 128;
        let key_count = 8192;
        let keys = gen_rand_strings(key_count, key_size);

        let put_count = 1000000;
        let get_count = 1000000;
        let del_count = 5000;

        let mut rng = rand::thread_rng();
        let init = keys.iter().cloned().map(|k| Task::Put(k, rng.gen::<u64>())).collect();
        let workloads = gen_workload::<String, u64>(keys, put_count, get_count, del_count);

        BenchRunner::run_benchmark("balanced_no_collision", key_count * 2, workloads, init);
    }

    #[test]
    // TODO: improve this by balancing puts and gets across cores
    fn optimap_collision_benchmark() {
        let mut rng = rand::thread_rng();

        // test parameters
        let key_count = 8192;
        let key_size = 128;
        let bucket_count = key_count / 4;
        let put_count: usize = 1000000;
        let get_count: usize = 1000000;
        let del_count: usize = 50000;

        let keys = gen_rand_strings(key_count, key_size);

        let init = keys.iter()
            .map(|k| Task::Put(k.clone(), rng.gen::<u64>()))
            .collect::<Vec<_>>();

        let workloads = core_affinity::get_core_ids().unwrap().into_iter()
            .skip(1)
            .map(|_| {
                let tasks = (0..(put_count + get_count + del_count)).map(|i| {
                    let key = keys[rng.gen::<usize>() % keys.len()].clone();

                    if i < put_count {
                        Task::Get(key)
                    } else if i < put_count + get_count {
                        Task::Put(key, rng.gen::<u64>())
                    } else {
                        Task::Remove(key)
                    }
                }).collect();

                Arc::new(tasks)
            }).collect::<Vec<Arc<_>>>();

        BenchRunner::run_benchmark("balanced_collision", bucket_count, workloads, init);
    }
}
