//! # Policy Cache — In-Memory Cache with Hit-Rate Analytics
//!
//! **100% isolated** — Zero dependencies on trading or agent systems.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A cached item with expiry and access tracking.
#[derive(Debug, Clone)]
pub struct CachedItem<T: Clone> {
    value: T,
    created_at: Instant,
    ttl: Duration,
    hit_count: u64,
}

impl<T: Clone> CachedItem<T> {
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    fn record_hit(&mut self) {
        self.hit_count += 1;
    }
}

/// Generic policy cache with TTL, hit-rate tracking, and size limits.
pub struct PolicyCache<T: Clone + std::fmt::Debug> {
    cache: HashMap<String, CachedItem<T>>,
    max_size: usize,
    default_ttl: Duration,
    // Analytics
    total_hits: u64,
    total_misses: u64,
}

impl<T: Clone + std::fmt::Debug> PolicyCache<T> {
    pub fn new(max_size: usize, default_ttl_secs: u64) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            default_ttl: Duration::from_secs(default_ttl_secs),
            total_hits: 0,
            total_misses: 0,
        }
    }

    /// Insert a value with the default TTL.
    pub fn insert(&mut self, key: String, value: T) {
        if self.cache.len() >= self.max_size {
            self.evict_oldest();
        }
        self.cache.insert(
            key,
            CachedItem {
                value,
                created_at: Instant::now(),
                ttl: self.default_ttl,
                hit_count: 0,
            },
        );
    }

    /// Insert a value with a custom TTL.
    pub fn insert_with_ttl(&mut self, key: String, value: T, ttl: Duration) {
        if self.cache.len() >= self.max_size {
            self.evict_oldest();
        }
        self.cache.insert(
            key,
            CachedItem {
                value,
                created_at: Instant::now(),
                ttl,
                hit_count: 0,
            },
        );
    }

    /// Get a value by key. Returns None if missing or expired.
    pub fn get(&mut self, key: &str) -> Option<&T> {
        if self.is_expired_inner(key) {
            self.cache.remove(key);
            self.total_misses += 1;
            return None;
        }
        let item = self.cache.get_mut(key)?;
        item.record_hit();
        self.total_hits += 1;
        Some(&item.value)
    }

    fn is_expired_inner(&self, key: &str) -> bool {
        self.cache.get(key).is_some_and(|item| item.is_expired())
    }

    /// Check if a key exists and is not expired.
    pub fn contains(&mut self, key: &str) -> bool {
        if self.is_expired_inner(key) {
            self.cache.remove(key);
            self.total_misses += 1;
            return false;
        }
        match self.cache.get_mut(key) {
            Some(item) => {
                item.record_hit();
                self.total_hits += 1;
                true
            }
            None => {
                self.total_misses += 1;
                false
            }
        }
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) {
        self.cache.remove(key);
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.total_hits = 0;
        self.total_misses = 0;
    }

    /// Cache hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            return 0.0;
        }
        self.total_hits as f64 / total as f64
    }

    /// Current cache size.
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Remove expired items.
    pub fn purge_expired(&mut self) {
        let expired_keys: Vec<String> = self
            .cache
            .iter()
            .filter(|(_, item)| item.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        for key in expired_keys {
            self.cache.remove(&key);
        }
    }

    /// Evict the oldest (least recently created) item.
    fn evict_oldest(&mut self) {
        let oldest_key = self
            .cache
            .iter()
            .min_by_key(|(_, item)| item.created_at)
            .map(|(k, _)| k.clone());
        if let Some(key) = oldest_key {
            self.cache.remove(&key);
        }
    }

    /// Total hits.
    pub fn total_hits(&self) -> u64 {
        self.total_hits
    }

    /// Total misses.
    pub fn total_misses(&self) -> u64 {
        self.total_misses
    }

    /// Total lookups.
    pub fn total_lookups(&self) -> u64 {
        self.total_hits + self.total_misses
    }
}

impl<T: Clone + std::fmt::Debug> Default for PolicyCache<T> {
    fn default() -> Self {
        Self::new(100, 300) // 100 items, 5 min TTL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_operations() {
        let mut cache: PolicyCache<String> = PolicyCache::new(10, 60);

        cache.insert("key1".into(), "value1".into());
        assert_eq!(cache.get("key1"), Some(&"value1".into()));
        assert!(cache.contains("key1"));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache: PolicyCache<String> = PolicyCache::new(10, 60);
        assert_eq!(cache.get("missing"), None);
        assert!(!cache.contains("missing"));
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_expiry() {
        let mut cache: PolicyCache<String> = PolicyCache::new(10, 1); // 1 second TTL
        cache.insert("key".into(), "value".into());
        assert!(cache.contains("key"));
        thread::sleep(Duration::from_secs(1));
        assert!(!cache.contains("key"));
    }

    #[test]
    fn test_eviction() {
        let mut cache: PolicyCache<String> = PolicyCache::new(3, 60); // max 3 items
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        cache.insert("c".into(), "3".into());
        assert_eq!(cache.size(), 3);
        cache.insert("d".into(), "4".into());
        assert_eq!(cache.size(), 3);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache: PolicyCache<String> = PolicyCache::new(10, 60);
        cache.insert("key".into(), "value".into());
        assert!(cache.contains("key")); // hit
        assert!(!cache.contains("missing")); // miss
        assert!(!cache.contains("missing2")); // miss
        let rate = cache.hit_rate();
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }
}
