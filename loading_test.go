// Copyright IBM Corp. 2014, 2026
// SPDX-License-Identifier: MPL-2.0

package lru

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// newTestLoadingCache builds a LoadingCache with an injectable clock so
// expiry behavior can be tested deterministically.
func newTestLoadingCache[K comparable, V any](t *testing.T, cfg LoadingCacheConfig[K, V], now func() time.Time) *LoadingCache[K, V] {
	t.Helper()
	lc, err := NewLoadingCache(cfg)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	lc.now = now
	return lc
}

// fakeClock is a manually controlled time source for deterministic expiry
// tests. It is safe to advance while loader goroutines read it.
type fakeClock struct {
	ns atomic.Int64
}

func (f *fakeClock) now() time.Time {
	return time.Unix(0, f.ns.Load())
}

func (f *fakeClock) advance(d time.Duration) {
	f.ns.Add(int64(d))
}

// blockableLoader returns a loader that blocks on release until it is
// closed, alongside counters and the per-call start signal.
func blockableLoader[K comparable](key K, value any, loadErr error) (func(context.Context, K) (any, error), <-chan struct{}, chan<- struct{}, *atomic.Int32) {
	started := make(chan struct{}, 16)
	release := make(chan struct{})
	var calls atomic.Int32
	loader := func(_ context.Context, k K) (any, error) {
		calls.Add(1)
		started <- struct{}{}
		<-release
		if k != key {
			return nil, fmt.Errorf("unexpected key %v", k)
		}
		return value, loadErr
	}
	return loader, started, release, &calls
}

func TestLoadingCacheConfigValidation(t *testing.T) {
	loader := func(_ context.Context, k int) (int, error) { return k, nil }

	if _, err := NewLoadingCache[int, int](LoadingCacheConfig[int, int]{Size: 0, Loader: loader}); err == nil {
		t.Fatal("expected error for non-positive size")
	}
	if _, err := NewLoadingCache[int, int](LoadingCacheConfig[int, int]{Size: 1}); err == nil {
		t.Fatal("expected error for nil loader")
	}
	if _, err := NewLoadingCache(LoadingCacheConfig[int, int]{
		Size: 1, Loader: loader, RefreshWindow: time.Second,
	}); err == nil {
		t.Fatal("expected error for RefreshWindow without TTL")
	}
	if _, err := NewLoadingCache(LoadingCacheConfig[int, int]{
		Size: 1, Loader: loader, TTL: time.Second, RefreshWindow: time.Second,
	}); err == nil {
		t.Fatal("expected error for RefreshWindow >= TTL")
	}
	if _, err := NewLoadingCache(LoadingCacheConfig[int, int]{
		Size: 1, Loader: loader, TTL: time.Second, RefreshWindow: 2 * time.Second,
	}); err == nil {
		t.Fatal("expected error for RefreshWindow > TTL")
	}
}

func TestLoadingCacheSingleFlightMiss(t *testing.T) {
	var calls atomic.Int32
	barrier := make(chan struct{})
	loader := func(_ context.Context, k int) (string, error) {
		calls.Add(1)
		<-barrier
		return fmt.Sprintf("v%d", k), nil
	}
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, string]{
		Size: 4, Loader: loader,
	}, time.Now)

	const n = 25
	var wg sync.WaitGroup
	results := make([]string, n)
	errs := make([]error, n)
	start := make(chan struct{})
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			results[i], errs[i] = lc.GetOrLoad(context.Background(), 1)
		}(i)
	}
	close(start)
	// Give every goroutine time to park on the shared flight.
	time.Sleep(50 * time.Millisecond)
	if got := calls.Load(); got != 1 {
		t.Fatalf("loader called %d times, want 1", got)
	}
	close(barrier)
	wg.Wait()

	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Fatalf("caller %d got error: %v", i, errs[i])
		}
		if results[i] != "v1" {
			t.Fatalf("caller %d got %q, want v1", i, results[i])
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("loader called %d times total, want 1", got)
	}
	// A subsequent call is a cache hit and must not invoke the loader.
	if v, err := lc.GetOrLoad(context.Background(), 1); err != nil || v != "v1" {
		t.Fatalf("cache hit got (%q, %v)", v, err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("loader called %d times after hit, want 1", got)
	}
}

func TestLoadingCacheSharedError(t *testing.T) {
	loadErr := errors.New("backend unavailable")
	barrier := make(chan struct{})
	var calls atomic.Int32
	loader := func(_ context.Context, _ int) (string, error) {
		calls.Add(1)
		<-barrier
		return "", loadErr
	}
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, string]{
		Size: 4, Loader: loader,
	}, time.Now)

	const n = 10
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := lc.GetOrLoad(context.Background(), 7)
			errs <- err
		}()
	}
	close(start)
	// Wait until the single shared load has started before releasing it.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) && calls.Load() < 1 {
		time.Sleep(5 * time.Millisecond)
	}
	close(barrier)
	wg.Wait()
	close(errs)
	for err := range errs {
		if !errors.Is(err, loadErr) {
			t.Errorf("got err %v, want %v", err, loadErr)
		}
	}

	if got := calls.Load(); got != 1 {
		t.Fatalf("failing loader called %d times, want 1", got)
	}
	// A failed load is not cached: the next miss starts a fresh loader.
	if _, err := lc.GetOrLoad(context.Background(), 7); !errors.Is(err, loadErr) {
		t.Fatalf("got err %v, want %v", err, loadErr)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("loader called %d times after failed load, want 2", got)
	}
}

func TestLoadingCacheWaiterCancellationDoesNotCancelLoad(t *testing.T) {
	loader, started, release, calls := blockableLoader[int](1, "v1", nil)
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, any]{
		Size: 4, Loader: loader,
	}, time.Now)

	// First caller starts the shared load.
	firstDone := make(chan struct{})
	go func() {
		_, _ = lc.GetOrLoad(context.Background(), 1)
		close(firstDone)
	}()
	<-started

	// A second caller gives up while the load is still running.
	ctx, cancel := context.WithCancel(context.Background())
	cancelDone := make(chan error, 1)
	go func() {
		_, err := lc.GetOrLoad(ctx, 1)
		cancelDone <- err
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()
	if err := <-cancelDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled waiter got %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("loader calls after cancel: %d, want 1", got)
	}

	// A third caller joins the same flight, then the loader finishes and
	// every surviving caller shares the single result.
	thirdErr := make(chan error, 1)
	thirdVal := make(chan any, 1)
	go func() {
		v, err := lc.GetOrLoad(context.Background(), 1)
		thirdVal <- v
		thirdErr <- err
	}()
	time.Sleep(20 * time.Millisecond)
	close(release)

	select {
	case <-firstDone:
	case <-time.After(time.Second):
		t.Fatal("first caller never returned")
	}
	if err := <-thirdErr; err != nil {
		t.Fatalf("third caller error: %v", err)
	}
	if v := <-thirdVal; v != "v1" {
		t.Fatalf("third caller got %v, want v1", v)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("loader called %d times, want 1", got)
	}
	if v, ok := lc.Peek(1); !ok || v != "v1" {
		t.Fatalf("result not admitted to cache: (%v, %v)", v, ok)
	}
}

func TestLoadingCacheStaleWhileRevalidate(t *testing.T) {
	clock := &fakeClock{}
	clock.ns.Store(time.Unix(1000, 0).UnixNano())
	var calls atomic.Int32
	loader := func(_ context.Context, _ int) (int, error) {
		// Each load yields a distinct value so refreshes are observable.
		return int(calls.Add(1)) * 100, nil
	}
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, int]{
		Size:          4,
		Loader:        loader,
		TTL:           10 * time.Second,
		RefreshWindow: 4 * time.Second, // stale from t+6, hard expire at t+10
	}, clock.now)

	if v, err := lc.GetOrLoad(context.Background(), 1); err != nil || v != 100 {
		t.Fatalf("initial load got (%d, %v)", v, err)
	}
	if calls.Load() != 1 {
		t.Fatalf("loader calls: %d, want 1", calls.Load())
	}

	// Fresh: no load.
	clock.advance(5 * time.Second)
	if v, err := lc.GetOrLoad(context.Background(), 1); err != nil || v != 100 {
		t.Fatalf("fresh read got (%d, %v)", v, err)
	}
	if calls.Load() != 1 {
		t.Fatalf("loader calls while fresh: %d, want 1", calls.Load())
	}

	// Stale window: old value returned immediately, refresh in background.
	clock.advance(2 * time.Second) // t+7
	if v, err := lc.GetOrLoad(context.Background(), 1); err != nil || v != 100 {
		t.Fatalf("stale read got (%d, %v), want stale value 100", v, err)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) && calls.Load() < 2 {
		time.Sleep(5 * time.Millisecond)
	}
	if calls.Load() != 2 {
		t.Fatalf("background refresh not started, calls: %d", calls.Load())
	}
	// Wait for the async refresh to land.
	deadline = time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if v, _ := lc.Peek(1); v == 200 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if v, ok := lc.Peek(1); !ok || v != 200 {
		t.Fatalf("refreshed value not cached, got (%d, %v)", v, ok)
	}
	refreshed, _ := lc.cache.Peek(1)

	// Hard expiry: stale value must never be served; caller waits for load.
	clock.advance(11 * time.Second) // past the refreshed entry's absolute TTL
	if !lc.isExpired(refreshed, clock.now()) {
		t.Fatal("test setup: refreshed entry should be absolutely expired")
	}
	blocker := make(chan struct{})
	lc.loader = func(_ context.Context, _ int) (int, error) {
		<-blocker
		return 300, nil
	}
	done := make(chan int, 1)
	go func() {
		v, _ := lc.GetOrLoad(context.Background(), 1)
		done <- v
	}()
	time.Sleep(30 * time.Millisecond)
	select {
	case v := <-done:
		t.Fatalf("absolutely expired value served before reload completed: %d", v)
	default:
	}
	close(blocker)
	if v := <-done; v != 300 {
		t.Fatalf("post-expiry load got %d, want 300", v)
	}
}

func TestLoadingCacheStaleRefreshIsSingleFlight(t *testing.T) {
	clock := &fakeClock{}
	barrier := make(chan struct{})
	var calls atomic.Int32
	loader := func(_ context.Context, k int) (int, error) {
		n := calls.Add(1)
		// The initial load completes immediately; later loads block so
		// callers can be observed sharing the refresh flight.
		if n > 1 {
			<-barrier
		}
		return k, nil
	}
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, int]{
		Size: 4, Loader: loader, TTL: 10 * time.Second, RefreshWindow: 5 * time.Second,
	}, clock.now)

	if _, err := lc.GetOrLoad(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	clock.advance(8 * time.Second) // stale

	// First stale read kicks off the background refresh.
	if v, _ := lc.GetOrLoad(context.Background(), 1); v != 1 {
		t.Fatalf("stale value: %d", v)
	}
	time.Sleep(20 * time.Millisecond)
	// Further stale reads and an expired read all share the one flight.
	clock.advance(3 * time.Second) // t+11: hard expired
	const waiters = 8
	var wg sync.WaitGroup
	for i := 0; i < waiters; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			v, err := lc.GetOrLoad(context.Background(), 1)
			if err != nil || v != 1 {
				t.Errorf("waiter got (%d, %v)", v, err)
			}
		}()
	}
	time.Sleep(30 * time.Millisecond)
	if got := calls.Load(); got != 2 {
		t.Fatalf("loader calls: %d, want 2 (initial + one refresh)", got)
	}
	close(barrier)
	wg.Wait()
	if got := calls.Load(); got != 2 {
		t.Fatalf("loader calls after release: %d, want 2", got)
	}
}

func TestLoadingCacheMonotonicVersionsRejectLateLoader(t *testing.T) {
	clock := &fakeClock{}
	slowBarrier := make(chan struct{})
	var slowCalls atomic.Int32
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, string]{
		Size: 4,
		Loader: func(_ context.Context, k int) (string, error) {
			slowCalls.Add(1)
			<-slowBarrier
			return "old", nil
		},
	}, clock.now)

	// Start a slow load for key 1.
	slowDone := make(chan struct{})
	go func() {
		_, _ = lc.GetOrLoad(context.Background(), 1)
		close(slowDone)
	}()
	time.Sleep(30 * time.Millisecond)

	// While it runs, explicitly Set a newer value.
	if err := lc.Set(1, "new"); err != nil {
		t.Fatal(err)
	}
	if v, _ := lc.Peek(1); v != "new" {
		t.Fatalf("Set did not take effect, got %q", v)
	}

	// The late loader result must not overwrite the newer value.
	close(slowBarrier)
	<-slowDone
	if v, _ := lc.Peek(1); v != "new" {
		t.Fatalf("late loader overwrote newer value, got %q", v)
	}
	if got := slowCalls.Load(); got != 1 {
		t.Fatalf("loader calls: %d, want 1", got)
	}

	// A fresh load after the key is removed again produces a new version.
	if err := lc.Remove(1); err != nil {
		t.Fatal(err)
	}
	fastBarrier := make(chan struct{})
	lc.loader = func(_ context.Context, _ int) (string, error) {
		<-fastBarrier
		return "fresh", nil
	}
	freshDone := make(chan string, 1)
	go func() {
		v, _ := lc.GetOrLoad(context.Background(), 1)
		freshDone <- v
	}()
	time.Sleep(30 * time.Millisecond)
	close(fastBarrier)
	if v := <-freshDone; v != "fresh" {
		t.Fatalf("fresh load got %q", v)
	}
	if v, _ := lc.Peek(1); v != "fresh" {
		t.Fatalf("cache holds %q, want fresh", v)
	}
}

func TestLoadingCacheVersionsIncreaseAcrossWrites(t *testing.T) {
	clock := &fakeClock{}
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, int]{
		Size: 4, Loader: func(_ context.Context, k int) (int, error) { return k, nil },
	}, clock.now)

	for i := 1; i <= 5; i++ {
		if err := lc.Set(1, i); err != nil {
			t.Fatal(err)
		}
		ent, ok := lc.cache.Peek(1)
		if !ok {
			t.Fatal("entry missing")
		}
		if ent.version != uint64(i) {
			t.Fatalf("write %d stored version %d, want %d", i, ent.version, i)
		}
	}

	// A reload after the key is removed gets a strictly greater version.
	ent, _ := lc.cache.Peek(1)
	before := ent.version
	if err := lc.Remove(1); err != nil {
		t.Fatal(err)
	}
	if _, err := lc.GetOrLoad(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	ent, _ = lc.cache.Peek(1)
	if ent.version <= before {
		t.Fatalf("reload version %d not greater than %d", ent.version, before)
	}
}

func TestLoadingCacheEvictionCallbackOnceAndCapacity(t *testing.T) {
	var mu sync.Mutex
	seen := make(map[int]int)
	lc, err := NewLoadingCache(LoadingCacheConfig[int, int]{
		Size: 2,
		Loader: func(_ context.Context, k int) (int, error) {
			return k * 10, nil
		},
		OnEvicted: func(key, value int) {
			mu.Lock()
			seen[key]++
			mu.Unlock()
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, k := range []int{1, 2, 3} {
		if _, err := lc.GetOrLoad(context.Background(), k); err != nil {
			t.Fatal(err)
		}
	}
	if lc.Len() != 2 {
		t.Fatalf("Len %d, want 2", lc.Len())
	}
	mu.Lock()
	if seen[1] != 1 {
		t.Fatalf("eviction callback for key 1 fired %d times, want 1", seen[1])
	}
	mu.Unlock()

	// Resize down evicts exactly one more entry, each callback once.
	if n, err := lc.Resize(1); err != nil || n != 1 {
		t.Fatalf("Resize got (%d, %v), want (1, nil)", n, err)
	}
	mu.Lock()
	total := 0
	for _, c := range seen {
		total += c
	}
	mu.Unlock()
	if total != 2 {
		t.Fatalf("total eviction callbacks %d, want 2", total)
	}
}

func TestLoadingCacheEvictionDuringInflightLoad(t *testing.T) {
	loader, started, release, _ := blockableLoader[int](1, "v1", nil)
	var mu sync.Mutex
	evicted := make(map[any]int)
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, any]{
		Size: 1,
		Loader: func(ctx context.Context, k int) (any, error) {
			return loader(ctx, k)
		},
		OnEvicted: func(k int, v any) {
			mu.Lock()
			evicted[v]++
			mu.Unlock()
		},
	}, time.Now)

	loadDone := make(chan struct{})
	go func() {
		_, _ = lc.GetOrLoad(context.Background(), 1)
		close(loadDone)
	}()
	<-started

	// While the load runs, fill the single slot with another key.
	if err := lc.Set(2, "other"); err != nil {
		t.Fatal(err)
	}
	close(release)
	<-loadDone

	// The late result is committed (key absent at version 0), evicting
	// key 2; every physical eviction fires the callback exactly once.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		ok := evicted["other"] == 1
		mu.Unlock()
		if ok {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	if evicted["other"] != 1 {
		t.Fatalf("key 2 eviction callback fired %d times, want 1", evicted["other"])
	}
	for v, c := range evicted {
		if c != 1 {
			t.Fatalf("value %v callback fired %d times, want 1", v, c)
		}
	}
}

func TestLoadingCachePurgeInvalidatesInflightAndFiresCallbacksOnce(t *testing.T) {
	loader, started, release, calls := blockableLoader[int](1, "v1", nil)
	var mu sync.Mutex
	evicted := make(map[int]int)
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, any]{
		Size: 4,
		Loader: func(ctx context.Context, k int) (any, error) {
			return loader(ctx, k)
		},
		OnEvicted: func(k int, _ any) {
			mu.Lock()
			evicted[k]++
			mu.Unlock()
		},
	}, time.Now)

	// Seed an entry so Purge has something to evict.
	if err := lc.Set(9, "seed"); err != nil {
		t.Fatal(err)
	}

	go func() { _, _ = lc.GetOrLoad(context.Background(), 1) }()
	<-started

	if err := lc.Purge(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	if evicted[9] != 1 {
		t.Fatalf("purge callback for seed fired %d times, want 1", evicted[9])
	}
	mu.Unlock()

	close(release)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if lc.Len() == 0 && calls.Load() == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if lc.Len() != 0 {
		t.Fatalf("in-flight result repopulated purged cache, Len=%d", lc.Len())
	}
	// Every callback still fired at most once.
	mu.Lock()
	for k, c := range evicted {
		if c != 1 {
			t.Fatalf("key %d callback fired %d times, want 1", k, c)
		}
	}
	mu.Unlock()
}

func TestLoadingCacheResizeDuringInflightLoad(t *testing.T) {
	loader, started, release, _ := blockableLoader[int](3, "v3", nil)
	var mu sync.Mutex
	var callbacks int
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, any]{
		Size: 4,
		Loader: func(ctx context.Context, k int) (any, error) {
			return loader(ctx, k)
		},
		OnEvicted: func(int, any) {
			mu.Lock()
			callbacks++
			mu.Unlock()
		},
	}, time.Now)

	for _, k := range []int{0, 1, 2} {
		if err := lc.Set(k, k); err != nil {
			t.Fatal(err)
		}
	}
	go func() { _, _ = lc.GetOrLoad(context.Background(), 3) }()
	<-started

	// Shrink while the load is in flight.
	if n, err := lc.Resize(2); err != nil || n != 1 {
		t.Fatalf("Resize got (%d, %v), want (1, nil)", n, err)
	}
	close(release)

	// Wait for the in-flight load to commit: key 3 must be present and
	// the cache must hold exactly the resized capacity.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if _, ok := lc.Peek(3); ok && lc.Len() == 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if _, ok := lc.Peek(3); !ok {
		t.Fatal("in-flight result never committed after release")
	}
	if lc.Len() != 2 {
		t.Fatalf("Len after in-flight commit: %d, want 2", lc.Len())
	}
	mu.Lock()
	defer mu.Unlock()
	if callbacks != 2 {
		t.Fatalf("eviction callbacks %d, want 2 (resize + capacity on commit)", callbacks)
	}
}

func TestLoadingCacheCloseStopsNewLoadersAndFiresCallbacksOnce(t *testing.T) {
	loader, started, release, calls := blockableLoader[int](1, "v1", nil)
	var mu sync.Mutex
	evicted := make(map[int]int)
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, any]{
		Size: 4,
		Loader: func(ctx context.Context, k int) (any, error) {
			return loader(ctx, k)
		},
		OnEvicted: func(k int, _ any) {
			mu.Lock()
			evicted[k]++
			mu.Unlock()
		},
	}, time.Now)
	if err := lc.Set(5, "seed"); err != nil {
		t.Fatal(err)
	}

	// In-flight load when Close happens.
	go func() { _, _ = lc.GetOrLoad(context.Background(), 1) }()
	<-started

	if err := lc.Close(); err != nil {
		t.Fatal(err)
	}
	// Idempotent.
	if err := lc.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	// No new loader may start, and all entry points report ErrClosed.
	if _, err := lc.GetOrLoad(context.Background(), 2); !errors.Is(err, ErrClosed) {
		t.Fatalf("GetOrLoad after Close got %v, want ErrClosed", err)
	}
	for _, op := range []func() error{
		func() error { return lc.Set(2, "x") },
		func() error { return lc.Remove(2) },
		func() error { return lc.Purge() },
		func() error { _, err := lc.Resize(8); return err },
	} {
		if err := op(); !errors.Is(err, ErrClosed) {
			t.Fatalf("operation after Close got %v, want ErrClosed", err)
		}
	}
	before := calls.Load()
	time.Sleep(30 * time.Millisecond)
	if calls.Load() != before {
		t.Fatal("new loader started after Close")
	}

	// The in-flight loader completes but its result is discarded.
	close(release)
	time.Sleep(50 * time.Millisecond)
	if lc.Len() != 0 {
		t.Fatalf("in-flight result committed after Close, Len=%d", lc.Len())
	}
	mu.Lock()
	defer mu.Unlock()
	if evicted[5] != 1 {
		t.Fatalf("seed eviction callback fired %d times, want 1", evicted[5])
	}
	for k, c := range evicted {
		if c != 1 {
			t.Fatalf("key %d callback fired %d times, want 1", k, c)
		}
	}
}

func TestLoadingCacheFailedRefreshKeepsStaleValue(t *testing.T) {
	clock := &fakeClock{}
	var calls atomic.Int32
	lc := newTestLoadingCache(t, LoadingCacheConfig[int, int]{
		Size:          4,
		TTL:           10 * time.Second,
		RefreshWindow: 5 * time.Second,
		Loader: func(_ context.Context, _ int) (int, error) {
			n := calls.Add(1)
			if n == 1 {
				return 100, nil // initial fill
			}
			return 0, errors.New("revalidate failed")
		},
	}, clock.now)

	if v, err := lc.GetOrLoad(context.Background(), 1); err != nil || v != 100 {
		t.Fatalf("initial load got (%d, %v)", v, err)
	}
	clock.advance(7 * time.Second) // stale, not hard expired

	// The stale value is still served; the background refresh fails but
	// must not evict or overwrite the existing entry.
	if v, err := lc.GetOrLoad(context.Background(), 1); err != nil || v != 100 {
		t.Fatalf("stale read got (%d, %v), want 100", v, err)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) && calls.Load() < 2 {
		time.Sleep(5 * time.Millisecond)
	}
	time.Sleep(20 * time.Millisecond) // let the failed refresh settle
	if v, ok := lc.Peek(1); !ok || v != 100 {
		t.Fatalf("stale entry lost after failed refresh: (%d, %v)", v, ok)
	}
	if lc.Len() != 1 {
		t.Fatalf("Len %d, want 1", lc.Len())
	}

	// Once hard expired, the stale value is no longer served and the
	// caller waits for a load, receiving its error.
	clock.advance(4 * time.Second) // t+11
	if _, ok := lc.Peek(1); ok {
		t.Fatal("hard expired entry must not be visible to Peek")
	}
	if _, err := lc.GetOrLoad(context.Background(), 1); err == nil {
		t.Fatal("expected loader error after hard expiry")
	}
}

func TestLoadingCacheConcurrentStress(t *testing.T) {
	var calls atomic.Int64
	lc, err := NewLoadingCache(LoadingCacheConfig[int, int]{
		Size: 16,
		Loader: func(_ context.Context, k int) (int, error) {
			calls.Add(1)
			time.Sleep(time.Millisecond)
			return k, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				key := (seed + i) % 32
				v, err := lc.GetOrLoad(context.Background(), key)
				if err != nil {
					t.Errorf("GetOrLoad(%d): %v", key, err)
					return
				}
				if v != key {
					t.Errorf("got %d, want %d", v, key)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	if lc.Len() > 16 {
		t.Fatalf("Len %d exceeds capacity 16", lc.Len())
	}
	if calls.Load() == 0 {
		t.Fatal("loader never ran")
	}
}

// TestLoadingCacheStressResizePurgeAndCancellation mixes concurrent loads,
// resizes, purges and caller cancellation. A key may legitimately be
// evicted, reloaded and evicted again across its lifecycle, so this run
// asserts capacity, progress and race safety rather than a global
// per-key eviction count; the at-most-once-per-eviction guarantee is
// covered by the unique-sentinel tests above.
func TestLoadingCacheStressResizePurgeAndCancellation(t *testing.T) {
	var loadCalls atomic.Int64
	var evictionCalls atomic.Int64
	lc, err := NewLoadingCache(LoadingCacheConfig[int, int]{
		Size: 8,
		Loader: func(_ context.Context, k int) (int, error) {
			loadCalls.Add(1)
			time.Sleep(time.Millisecond)
			return k, nil
		},
		OnEvicted: func(_, _ int) { evictionCalls.Add(1) },
	})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for g := 0; g < 12; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for i := 0; i < 300; i++ {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
				_, _ = lc.GetOrLoad(ctx, (seed*7+i)%40)
				cancel()
				if i%50 == 0 {
					_, _ = lc.Resize(4 + (i % 8))
				}
				if i%77 == 0 {
					_ = lc.Purge()
				}
			}
		}(g)
	}
	wg.Wait()
	if lc.Len() > lc.Cap() {
		t.Fatalf("Len %d exceeds capacity %d", lc.Len(), lc.Cap())
	}
	if loadCalls.Load() == 0 || evictionCalls.Load() == 0 {
		t.Fatal("expected both loads and evictions to occur")
	}
}
