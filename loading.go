// Copyright IBM Corp. 2014, 2026
// SPDX-License-Identifier: MPL-2.0

package lru

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrClosed is returned by LoadingCache operations after Close has been
// called. A closed cache never starts a new loader.
var ErrClosed = errors.New("loading cache is closed")

// Loader is invoked to load the value of a key that is not currently
// servable from the cache. It is called at most once per concurrent miss:
// all concurrent callers for the same key share a single loader call and
// receive the same result, including a shared error.
//
// The context passed to a loader is never derived from a waiting caller's
// context, so cancelling individual callers does not cancel the shared
// load. A loader may therefore outlive the request that triggered it.
type Loader[K comparable, V any] func(ctx context.Context, key K) (V, error)

// LoadingCacheConfig configures a LoadingCache.
type LoadingCacheConfig[K comparable, V any] struct {
	// Size is the maximum number of entries kept in the underlying LRU.
	// Must be positive.
	Size int

	// Loader fetches values for keys that cannot be served from the
	// cache. Required.
	Loader Loader[K, V]

	// OnEvicted, if non-nil, is invoked once per evicted entry outside
	// the cache's internal lock, mirroring the guarantee of
	// NewWithEvict.
	OnEvicted func(key K, value V)

	// TTL is the absolute lifetime of a cached value. After TTL elapses
	// since a value was written, it must never be served again and a
	// fresh load is required.
	//
	// A zero or negative TTL disables time based expiry; entries then
	// only leave the cache through capacity eviction or explicit
	// removal.
	TTL time.Duration

	// RefreshWindow turns on stale-while-revalidate. An entry becomes
	// stale RefreshWindow before its absolute expiry: stale values are
	// returned immediately while a single background loader refreshes
	// the key. Once the absolute TTL passes the stale value is no
	// longer served and the caller waits for a fresh load.
	//
	// A zero RefreshWindow disables stale-while-revalidate. It must be
	// smaller than TTL; setting it without a positive TTL is invalid.
	RefreshWindow time.Duration
}

// loadingEntry is the value type stored in the underlying Cache.
type loadingEntry[K comparable, V any] struct {
	value    V
	version  uint64
	staleAt  time.Time
	expireAt time.Time
}

// loadFlight tracks a single in-flight loader call shared by all callers
// waiting on a key.
type loadFlight[K comparable, V any] struct {
	done chan struct{}
	// seenVersion is the version the key held when the load started, or
	// 0 when the key was absent. A result may only be committed while
	// the key still holds this version (or stays absent at 0).
	seenVersion uint64
	// gen is the LoadingCache generation the load started in. Results
	// from a generation superseded by Purge are discarded.
	gen   uint64
	value V
	err   error
}

// LoadingCache is a thread-safe fixed size LRU cache view that loads
// missing keys through a Loader. It wraps a Cache, which remains fully
// usable on its own; LoadingCache adds single-flight loading, context
// aware waits, stale-while-revalidate and monotonic versioning on top of
// it.
type LoadingCache[K comparable, V any] struct {
	cache     *Cache[K, *loadingEntry[K, V]]
	loader    Loader[K, V]
	onEvicted func(K, V)
	ttl       time.Duration
	stale     time.Duration
	mu        sync.Mutex
	flights   map[K]*loadFlight[K, V]
	version   uint64
	// gen is bumped by Purge so in-flight loads cannot repopulate a
	// purged cache behind the purger's back.
	gen    uint64
	closed bool
	now    func() time.Time

	// pendingEvict collects entries evicted by the underlying Cache so
	// their user facing OnEvicted callbacks can run outside the lock.
	pendingEvict []evictedPair[K, V]
}

// evictedPair is a snapshot of an entry removed from the underlying cache.
type evictedPair[K comparable, V any] struct {
	key   K
	value V
}

// NewLoadingCache creates a LoadingCache with the given configuration.
func NewLoadingCache[K comparable, V any](cfg LoadingCacheConfig[K, V]) (*LoadingCache[K, V], error) {
	if cfg.Size <= 0 {
		return nil, errors.New("must provide a positive size")
	}
	if cfg.Loader == nil {
		return nil, errors.New("must provide a loader")
	}
	if cfg.TTL <= 0 {
		if cfg.RefreshWindow != 0 {
			return nil, errors.New("RefreshWindow requires a positive TTL")
		}
	} else if cfg.RefreshWindow < 0 || cfg.RefreshWindow >= cfg.TTL {
		return nil, errors.New("RefreshWindow must be zero or smaller than TTL")
	}

	lc := &LoadingCache[K, V]{
		loader:    cfg.Loader,
		onEvicted: cfg.OnEvicted,
		ttl:       cfg.TTL,
		stale:     cfg.RefreshWindow,
		flights:   make(map[K]*loadFlight[K, V]),
		now:       time.Now,
	}

	c, err := NewWithEvict[K, *loadingEntry[K, V]](cfg.Size, func(key K, ent *loadingEntry[K, V]) {
		// Runs under the LoadingCache lock because every underlying
		// cache mutation is performed while holding it; the user
		// callback itself is drained after the lock is released.
		if lc.onEvicted != nil && ent != nil {
			lc.pendingEvict = append(lc.pendingEvict, evictedPair[K, V]{key: key, value: ent.value})
		}
	})
	if err != nil {
		return nil, err
	}
	lc.cache = c
	return lc, nil
}

// Cache returns the underlying Cache. Direct writes through it bypass the
// LoadingCache's version tracking and in-flight coordination; most users
// should only read it.
func (lc *LoadingCache[K, V]) Cache() *Cache[K, *loadingEntry[K, V]] {
	return lc.cache
}

// GetOrLoad returns the value for key. A fresh entry is returned directly;
// a stale entry within its revalidation window is returned immediately
// while a background refresh is started; an absent or absolutely expired
// entry triggers a shared load, whose single result (value or error) is
// returned to every concurrent caller.
//
// A waiting caller whose context is cancelled receives its context error
// without affecting the shared load or any other caller.
func (lc *LoadingCache[K, V]) GetOrLoad(ctx context.Context, key K) (V, error) {
	var zero V

	lc.mu.Lock()
	if lc.closed {
		lc.mu.Unlock()
		return zero, ErrClosed
	}
	now := lc.now()
	if ent, ok := lc.cache.Get(key); ok {
		if !lc.isExpired(ent, now) {
			if !lc.isStale(ent, now) {
				lc.mu.Unlock()
				return ent.value, nil
			}
			// Stale but still usable: serve immediately and refresh
			// in the background unless a load is already running.
			if _, loading := lc.flights[key]; !loading {
				lc.startFlight(key, ent.version)
			}
			lc.mu.Unlock()
			return ent.value, nil
		}
	}
	flight := lc.ensureFlight(key)
	lc.mu.Unlock()

	return lc.waitFlight(ctx, flight)
}

// ensureFlight returns the existing flight for key or starts a new one for
// an absent/expired key. The caller must hold lc.mu.
func (lc *LoadingCache[K, V]) ensureFlight(key K) *loadFlight[K, V] {
	if flight, ok := lc.flights[key]; ok {
		return flight
	}
	// Peek captures the current version without changing recency; the
	// entry may be expired or absent.
	ent, _ := lc.cache.Peek(key)
	var seen uint64
	if ent != nil {
		seen = ent.version
	}
	return lc.startFlight(key, seen)
}

// startFlight registers a new flight for key and spawns the loader. The
// caller must hold lc.mu.
func (lc *LoadingCache[K, V]) startFlight(key K, seenVersion uint64) *loadFlight[K, V] {
	flight := &loadFlight[K, V]{
		done:        make(chan struct{}),
		seenVersion: seenVersion,
		gen:         lc.gen,
	}
	lc.flights[key] = flight
	go lc.runFlight(key, flight)
	return flight
}

// runFlight executes the loader and publishes its result. The loader runs
// with a non-cancellable context so that individual waiters giving up
// cannot abort work other callers still depend on.
func (lc *LoadingCache[K, V]) runFlight(key K, flight *loadFlight[K, V]) {
	value, err := lc.loader(context.Background(), key)

	lc.mu.Lock()
	flight.value = value
	flight.err = err
	// Publish before waking waiters: a receive on done must observe the
	// result fields.
	close(flight.done)
	delete(lc.flights, key)

	if err == nil && !lc.closed && flight.gen == lc.gen {
		lc.commitLocked(key, value, flight.seenVersion)
	}
	evicted := lc.drainEvicted()
	lc.mu.Unlock()
	lc.fireEvicted(evicted)
}

// waitFlight blocks until the shared load finishes or the caller's context
// is done. Cancelling here never cancels the shared load.
func (lc *LoadingCache[K, V]) waitFlight(ctx context.Context, flight *loadFlight[K, V]) (V, error) {
	var zero V
	select {
	case <-flight.done:
		return flight.value, flight.err
	case <-ctx.Done():
		return zero, ctx.Err()
	}
}

// commitLocked writes value for key if no newer version has taken the key
// since the load started, so a slow loader's late result can never
// overwrite a newer value. The caller must hold lc.mu; evictions triggered
// by the write are queued and drained by the caller after unlocking.
func (lc *LoadingCache[K, V]) commitLocked(key K, value V, seenVersion uint64) {
	if current, ok := lc.cache.Peek(key); ok && current.version != seenVersion {
		return
	}
	lc.writeLocked(key, value)
}

// writeLocked stores value under a new monotonically increasing version.
// The caller must hold lc.mu; evictions triggered by the write are queued
// and drained by the caller after unlocking.
func (lc *LoadingCache[K, V]) writeLocked(key K, value V) {
	lc.version++
	now := lc.now()
	ent := &loadingEntry[K, V]{
		value:   value,
		version: lc.version,
	}
	if lc.ttl > 0 {
		ent.staleAt = now.Add(lc.ttl - lc.stale)
		ent.expireAt = now.Add(lc.ttl)
	}
	lc.cache.Add(key, ent)
}

// Set explicitly writes a value for key under a new version. Any in-flight
// loader whose result arrives afterwards is superseded and cannot overwrite
// this value.
func (lc *LoadingCache[K, V]) Set(key K, value V) error {
	lc.mu.Lock()
	if lc.closed {
		lc.mu.Unlock()
		return ErrClosed
	}
	lc.writeLocked(key, value)
	evicted := lc.drainEvicted()
	lc.mu.Unlock()
	lc.fireEvicted(evicted)
	return nil
}

// Remove deletes key from the cache. An in-flight loader for the key still
// completes and may repopulate it, since other callers are waiting on that
// shared result.
func (lc *LoadingCache[K, V]) Remove(key K) error {
	lc.mu.Lock()
	if lc.closed {
		lc.mu.Unlock()
		return ErrClosed
	}
	lc.cache.Remove(key)
	evicted := lc.drainEvicted()
	lc.mu.Unlock()
	lc.fireEvicted(evicted)
	return nil
}

// Purge clears the underlying cache and invalidates any in-flight loader
// results: a load started before Purge cannot repopulate a key afterwards.
// Waiters already blocked on such a load still receive its shared result.
func (lc *LoadingCache[K, V]) Purge() error {
	lc.mu.Lock()
	if lc.closed {
		lc.mu.Unlock()
		return ErrClosed
	}
	lc.gen++
	lc.cache.Purge()
	evicted := lc.drainEvicted()
	lc.mu.Unlock()
	lc.fireEvicted(evicted)
	return nil
}

// Resize changes the capacity of the underlying cache.
func (lc *LoadingCache[K, V]) Resize(size int) (int, error) {
	lc.mu.Lock()
	if lc.closed {
		lc.mu.Unlock()
		return 0, ErrClosed
	}
	evictedCount := lc.cache.Resize(size)
	evicted := lc.drainEvicted()
	lc.mu.Unlock()
	lc.fireEvicted(evicted)
	return evictedCount, nil
}

// Close shuts the cache down. No new loader is started afterwards; callers
// receive ErrClosed. In-flight loaders run to completion but their results
// are discarded, and entries remaining in the cache are evicted with their
// OnEvicted callbacks. Close is idempotent.
func (lc *LoadingCache[K, V]) Close() error {
	lc.mu.Lock()
	if lc.closed {
		lc.mu.Unlock()
		return nil
	}
	lc.closed = true
	lc.cache.Purge()
	evicted := lc.drainEvicted()
	lc.mu.Unlock()
	lc.fireEvicted(evicted)
	return nil
}

// Peek returns the value for key without updating recency or triggering a
// load. Stale values within the revalidation window are still returned;
// absolutely expired entries are reported as absent.
func (lc *LoadingCache[K, V]) Peek(key K) (V, bool) {
	var zero V
	lc.mu.Lock()
	defer lc.mu.Unlock()
	ent, ok := lc.cache.Peek(key)
	if !ok || lc.isExpired(ent, lc.now()) {
		return zero, false
	}
	return ent.value, true
}

// Len returns the number of entries currently stored, including stale and
// not-yet-cleaned expired entries.
func (lc *LoadingCache[K, V]) Len() int {
	lc.mu.Lock()
	defer lc.mu.Unlock()
	return lc.cache.Len()
}

// Cap returns the capacity of the underlying cache.
func (lc *LoadingCache[K, V]) Cap() int {
	return lc.cache.Cap()
}

// Keys returns the keys of entries that are not absolutely expired, from
// oldest to newest, without updating recency.
func (lc *LoadingCache[K, V]) Keys() []K {
	lc.mu.Lock()
	defer lc.mu.Unlock()
	now := lc.now()
	keys := lc.cache.Keys()
	out := make([]K, 0, len(keys))
	for _, key := range keys {
		if ent, ok := lc.cache.Peek(key); ok && !lc.isExpired(ent, now) {
			out = append(out, key)
		}
	}
	return out
}

// isExpired reports whether ent has passed its absolute expiry. Entries
// written with TTL disabled never expire. The caller must hold lc.mu for
// the injected clock.
func (lc *LoadingCache[K, V]) isExpired(ent *loadingEntry[K, V], now time.Time) bool {
	return !ent.expireAt.IsZero() && !now.Before(ent.expireAt)
}

// isStale reports whether ent has entered its revalidation window. The
// caller must hold lc.mu.
func (lc *LoadingCache[K, V]) isStale(ent *loadingEntry[K, V], now time.Time) bool {
	return !ent.staleAt.IsZero() && !now.Before(ent.staleAt)
}

// drainEvicted snapshots and resets the queued eviction callbacks. The
// caller must hold lc.mu.
func (lc *LoadingCache[K, V]) drainEvicted() []evictedPair[K, V] {
	if len(lc.pendingEvict) == 0 {
		return nil
	}
	evicted := lc.pendingEvict
	lc.pendingEvict = nil
	return evicted
}

// fireEvicted runs the user facing eviction callbacks outside the cache's
// internal lock so callbacks may re-enter the cache.
func (lc *LoadingCache[K, V]) fireEvicted(evicted []evictedPair[K, V]) {
	if lc.onEvicted == nil {
		return
	}
	for _, e := range evicted {
		lc.onEvicted(e.key, e.value)
	}
}
