golang-lru
==========

This provides the `lru` package which implements a fixed-size
thread safe LRU cache. It is based on the cache in Groupcache.

Documentation
=============

Full docs are available on [Go Packages](https://pkg.go.dev/github.com/hashicorp/golang-lru/v2)

LRU cache example
=================

```go
package main

import (
	"fmt"
	"github.com/hashicorp/golang-lru/v2"
)

func main() {
	l, _ := lru.New[int, any](128)
	for i := 0; i < 256; i++ {
		l.Add(i, nil)
	}
	if l.Len() != 128 {
		panic(fmt.Sprintf("bad len: %v", l.Len()))
	}
}
```

Loading cache example
=====================

`LoadingCache` is a cache view with an attached loader. Concurrent misses
for the same key collapse into a single loader call whose result is shared
by every caller (including a shared error); waiters may cancel their own
context without cancelling the shared load. It supports stale-while-
revalidate with a hard absolute expiry, assigns each successful write a
monotonic version so late loader results never overwrite newer values, and
guarantees each eviction callback fires at most once. The underlying
`Cache` remains independently usable.

```go
package main

import (
	"context"
	"fmt"
	"time"

	"github.com/hashicorp/golang-lru/v2"
)

func main() {
	lc, err := lru.NewLoadingCache(lru.LoadingCacheConfig[string, string]{
		Size: 128,
		Loader: func(ctx context.Context, key string) (string, error) {
			// Exactly one of these runs per concurrent miss.
			return fetchFromBackend(ctx, key)
		},
		OnEvicted:     func(key, value string) { fmt.Println("evicted", key) },
		TTL:           time.Minute,      // hard absolute expiry
		RefreshWindow: 10 * time.Second, // serve stale while refreshing
	})
	if err != nil {
		panic(err)
	}
	defer lc.Close()

	// Concurrent callers for "user:42" share one loader invocation.
	value, err := lc.GetOrLoad(context.Background(), "user:42")
	if err != nil {
		panic(err)
	}
	fmt.Println(value)
}

func fetchFromBackend(_ context.Context, key string) (string, error) {
	return "value-for-" + key, nil
}
```

Expirable LRU cache example
===========================

```go
package main

import (
	"fmt"
	"time"

	"github.com/hashicorp/golang-lru/v2/expirable"
)

func main() {
	// make cache with 10ms TTL and 5 max keys
	cache := expirable.NewLRU[string, string](5, nil, time.Millisecond*10)


	// set value under key1.
	cache.Add("key1", "val1")

	// get value under key1
	r, ok := cache.Get("key1")

	// check for OK value
	if ok {
		fmt.Printf("value before expiration is found: %v, value: %q\n", ok, r)
	}

	// wait for cache to expire
	time.Sleep(time.Millisecond * 12)

	// get value under key1 after key expiration
	r, ok = cache.Get("key1")
	fmt.Printf("value after expiration is found: %v, value: %q\n", ok, r)

	// set value under key2, would evict old entry because it is already expired.
	cache.Add("key2", "val2")

	fmt.Printf("Cache len: %d\n", cache.Len())
	// Output:
	// value before expiration is found: true, value: "val1"
	// value after expiration is found: false, value: ""
	// Cache len: 1
}
```
