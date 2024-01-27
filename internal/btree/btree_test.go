// Adapted from https://github.com/google/btree/blob/v1.1.2/btree_test.go
// Copyright 2022 Sogang University
// Copyright 2014 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build go1.18
// +build go1.18

package btree

import (
	"flag"
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"sync"
	"testing"
)

// perm returns a random permutation of n items with size in the range [0, n).
func perm(n int) (out []*ItemBase) {
	for _, v := range rand.Perm(n) {
		out = append(out, &ItemBase{v, v})
	}
	return
}

// rang returns an ordered list of items with size in the range [0, n).
func rang(n int) (out []*ItemBase) {
	for i := 0; i < n; i++ {
		out = append(out, &ItemBase{i, i})
	}
	return
}

// all extracts all items from a tree in order as a slice.
func all[T Item](t *BTree[T]) (out []T) {
	t.Ascend(func(a T) bool {
		out = append(out, a)
		return true
	})
	return
}

// rangerev returns a reversed ordered list of items with size in the range [0, n).
func rangrev(n int) (out []*ItemBase) {
	for i := n - 1; 0 <= i; i-- {
		out = append(out, &ItemBase{i, i})
	}
	return
}

// allrev extracts all items from a tree in reverse order as a slice.
func allrev[T Item](t *BTree[T]) (out []T) {
	t.Descend(func(a T) bool {
		out = append(out, a)
		return true
	})
	return
}

var btreeDegree = flag.Int("degree", DefaultTargetNodeSize[*ItemBase](), "B-tree degree")

func TestBTree(t *testing.T) {
	tr := New[*ItemBase](*btreeDegree)
	const treeSize = 10000
	for i := 0; i < 10; i++ {
		if min, ok := tr.Min(); ok || min != nil {
			t.Fatalf("empty min, got %+v", min)
		}
		if max, ok := tr.Max(); ok || max != nil {
			t.Fatalf("empty max, got %+v", max)
		}
		for _, item := range perm(treeSize) {
			if x, ok := tr.ReplaceOrInsert(item); ok || x != nil {
				t.Fatal("insert found item", item)
			}
		}
		for _, item := range perm(treeSize) {
			if !tr.Has(item) {
				t.Fatal("has did not find item", item)
			}
		}
		for _, item := range perm(treeSize) {
			if x, ok := tr.ReplaceOrInsert(item); !ok || x == nil {
				t.Fatal("insert didn't find item", item)
			}
		}
		want := &ItemBase{0, 0}
		if min, ok := tr.Min(); !ok || min.Size() != want.Size() {
			t.Fatalf("min: ok %v want %+v, got %+v", ok, want, min)
		}
		want = &ItemBase{treeSize - 1, treeSize - 1}
		if max, ok := tr.Max(); !ok || max.Size() != want.Size() {
			t.Fatalf("max: ok %v want %+v, got %+v", ok, want, max)
		}
		got := all(tr)
		wantRange := rang(treeSize)
		if !reflect.DeepEqual(got, wantRange) {
			t.Fatalf("mismatch:\n got: %v\nwant: %v", got, wantRange)
		}

		gotrev := allrev(tr)
		wantrev := rangrev(treeSize)
		if !reflect.DeepEqual(gotrev, wantrev) {
			t.Fatalf("mismatch:\n got: %v\nwant: %v", gotrev, wantrev)
		}

		for _, item := range perm(treeSize) {
			if x, ok := tr.Delete(item); !ok || x == nil {
				t.Fatalf("didn't find %v", item)
			}
		}
		if got = all(tr); 0 < len(got) {
			t.Fatalf("some left!: %v", got)
		}
		if got = allrev(tr); 0 < len(got) {
			t.Fatalf("some left!: %v", got)
		}
	}
}

func ExampleBTree() {
	tr := New[*ItemBase](*btreeDegree)
	for i := 0; i < 10; i++ {
		tr.ReplaceOrInsert(&ItemBase{i, i})
	}
	fmt.Println("len:       ", tr.Len())
	v, ok := tr.Get(&ItemBase{3, 3})
	fmt.Println("get3:      ", v, ok)
	v, ok = tr.Get(&ItemBase{100, 100})
	fmt.Println("get100:    ", v, ok)
	v, ok = tr.Delete(&ItemBase{4, 4})
	fmt.Println("del4:      ", v, ok)
	v, ok = tr.Delete(&ItemBase{100, 100})
	fmt.Println("del100:    ", v, ok)
	v, ok = tr.ReplaceOrInsert(&ItemBase{5, 5})
	fmt.Println("replace5:  ", v, ok)
	v, ok = tr.ReplaceOrInsert(&ItemBase{100, 100})
	fmt.Println("replace100:", v, ok)
	v, ok = tr.Min()
	fmt.Println("min:       ", v, ok)
	v, ok = tr.DeleteMin()
	fmt.Println("delmin:    ", v, ok)
	v, ok = tr.Max()
	fmt.Println("max:       ", v, ok)
	v, ok = tr.DeleteMax()
	fmt.Println("delmax:    ", v, ok)
	fmt.Println("len:       ", tr.Len())
	// Output:
	// len:        10
	// get3:       &{3 3} true
	// get100:     <nil> false
	// del4:       &{4 4} true
	// del100:     <nil> false
	// replace5:   &{5 5} true
	// replace100: <nil> false
	// min:        &{0 0} true
	// delmin:     &{0 0} true
	// max:        &{100 100} true
	// delmax:     &{100 100} true
	// len:        8
}

func TestDeleteMin(t *testing.T) {
	tr := New[*ItemBase](3)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	for v, ok := tr.DeleteMin(); ok; v, ok = tr.DeleteMin() {
		got = append(got, v)
	}
	if want := rang(100); !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestDeleteMax(t *testing.T) {
	tr := New[*ItemBase](3)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	for v, ok := tr.DeleteMax(); ok; v, ok = tr.DeleteMax() {
		got = append(got, v)
	}
	if want := rangrev(100); !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestDeleteNearest(t *testing.T) {
	tr := New[*ItemBase](3)
	for _, v := range perm(300) {
		tr.ReplaceOrInsert(v)
	}
	for _, item := range perm(100) {
		if x, ok := tr.DeleteNearest(item); !ok || x.Size() != item.Size() {
			t.Fatalf("didn't find %v", item)
		}
	}
	for _, item := range rang(100) {
		if x, ok := tr.DeleteNearest(item); !ok || x.Size() != item.Size()+100 {
			t.Fatalf("didn't find %v", item)
		}
	}
	for _, item := range rangrev(100) {
		if x, ok := tr.DeleteNearest(item); !ok || x.Size() != 299-item.Size() {
			t.Fatalf("didn't find %v", item)
		}
	}
}

func TestAscendRange(t *testing.T) {
	tr := New[*ItemBase](2)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	tr.AscendRange(&ItemBase{40, 40}, &ItemBase{60, 60}, func(a *ItemBase) bool {
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:60]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.AscendRange(&ItemBase{40, 40}, &ItemBase{60, 60}, func(a *ItemBase) bool {
		if 50 < a.Size() {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:51]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestDescendRange(t *testing.T) {
	tr := New[*ItemBase](2)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	tr.DescendRange(&ItemBase{60, 60}, &ItemBase{40, 40}, func(a *ItemBase) bool {
		got = append(got, a)
		return true
	})
	if want := rangrev(100)[39:59]; !reflect.DeepEqual(got, want) {
		t.Fatalf("descendrange:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.DescendRange(&ItemBase{60, 60}, &ItemBase{40, 40}, func(a *ItemBase) bool {
		if a.Size() < 50 {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rangrev(100)[39:50]; !reflect.DeepEqual(got, want) {
		t.Fatalf("descendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestAscendLessThan(t *testing.T) {
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	tr.AscendLessThan(&ItemBase{60, 60}, func(a *ItemBase) bool {
		got = append(got, a)
		return true
	})
	if want := rang(100)[:60]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.AscendLessThan(&ItemBase{60, 60}, func(a *ItemBase) bool {
		if 50 < a.Size() {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rang(100)[:51]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestDescendLessOrEqual(t *testing.T) {
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	tr.DescendLessOrEqual(&ItemBase{40, 40}, func(a *ItemBase) bool {
		got = append(got, a)
		return true
	})
	if want := rangrev(100)[59:]; !reflect.DeepEqual(got, want) {
		t.Fatalf("descendlessorequal:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.DescendLessOrEqual(&ItemBase{60, 60}, func(a *ItemBase) bool {
		if a.Size() < 50 {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rangrev(100)[39:50]; !reflect.DeepEqual(got, want) {
		t.Fatalf("descendlessorequal:\n got: %v\nwant: %v", got, want)
	}
}

func TestAscendGreaterOrEqual(t *testing.T) {
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	tr.AscendGreaterOrEqual(&ItemBase{40, 40}, func(a *ItemBase) bool {
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.AscendGreaterOrEqual(&ItemBase{40, 40}, func(a *ItemBase) bool {
		if 50 < a.Size() {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:51]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestDescendGreaterThan(t *testing.T) {
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []*ItemBase
	tr.DescendGreaterThan(&ItemBase{40, 40}, func(a *ItemBase) bool {
		got = append(got, a)
		return true
	})
	if want := rangrev(100)[:59]; !reflect.DeepEqual(got, want) {
		t.Fatalf("descendgreaterthan:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.DescendGreaterThan(&ItemBase{40, 40}, func(a *ItemBase) bool {
		if a.Size() < 50 {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rangrev(100)[:50]; !reflect.DeepEqual(got, want) {
		t.Fatalf("descendgreaterthan:\n got: %v\nwant: %v", got, want)
	}
}

const benchmarkTreeSize = 10000

func BenchmarkInsert(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		tr := New[*ItemBase](*btreeDegree)
		for _, item := range insertP {
			tr.ReplaceOrInsert(item)
			i++
			if b.N <= i {
				return
			}
		}
	}
}

func BenchmarkSeek(b *testing.B) {
	b.StopTimer()
	size := 100000
	insertP := perm(size)
	tr := New[*ItemBase](*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		tr.AscendGreaterOrEqual(&ItemBase{i % size, i % size}, func(a *ItemBase) bool { return false })
	}
}

func BenchmarkDeleteInsert(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tr.Delete(insertP[i%benchmarkTreeSize])
		tr.ReplaceOrInsert(insertP[i%benchmarkTreeSize])
	}
}

func BenchmarkDeleteInsertCloneOnce(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	tr = tr.Clone()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tr.Delete(insertP[i%benchmarkTreeSize])
		tr.ReplaceOrInsert(insertP[i%benchmarkTreeSize])
	}
}

func BenchmarkDeleteInsertCloneEachTime(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, item := range insertP {
		tr.ReplaceOrInsert(item)
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tr = tr.Clone()
		tr.Delete(insertP[i%benchmarkTreeSize])
		tr.ReplaceOrInsert(insertP[i%benchmarkTreeSize])
	}
}

func BenchmarkDelete(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	removeP := perm(benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New[*ItemBase](*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr.Delete(item)
			i++
			if b.N <= i {
				return
			}
		}
		if 0 < tr.Len() {
			panic(tr.Len())
		}
	}
}

func BenchmarkGet(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	removeP := perm(benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New[*ItemBase](*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr.Get(item)
			i++
			if b.N <= i {
				return
			}
		}
	}
}

func BenchmarkGetCloneEachTime(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	removeP := perm(benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New[*ItemBase](*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr = tr.Clone()
			tr.Get(item)
			i++
			if b.N <= i {
				return
			}
		}
	}
}

func BenchmarkAscend(b *testing.B) {
	arr := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Less(arr[j])
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := 0
		tr.Ascend(func(item *ItemBase) bool {
			if item.Size() != arr[j].Size() {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j], item)
			}
			j++
			return true
		})
	}
}

func BenchmarkDescend(b *testing.B) {
	arr := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Less(arr[j])
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := len(arr) - 1
		tr.Descend(func(item *ItemBase) bool {
			if item.Size() != arr[j].Size() {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j], item)
			}
			j--
			return true
		})
	}
}

func BenchmarkAscendRange(b *testing.B) {
	arr := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Less(arr[j])
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := 100
		tr.AscendRange(&ItemBase{100, 100}, arr[len(arr)-100], func(item *ItemBase) bool {
			if item.Size() != arr[j].Size() {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j], item)
			}
			j++
			return true
		})
		if j != len(arr)-100 {
			b.Fatalf("expected: %v, got %v", len(arr)-100, j)
		}
	}
}

func BenchmarkDescendRange(b *testing.B) {
	arr := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Less(arr[j])
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := len(arr) - 100
		tr.DescendRange(arr[len(arr)-100], &ItemBase{100, 100}, func(item *ItemBase) bool {
			if item.Size() != arr[j].Size() {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j], item)
			}
			j--
			return true
		})
		if j != 100 {
			b.Fatalf("expected: %v, got %v", len(arr)-100, j)
		}
	}
}

func BenchmarkAscendGreaterOrEqual(b *testing.B) {
	arr := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Less(arr[j])
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := 100
		k := 0
		tr.AscendGreaterOrEqual(&ItemBase{100, 100}, func(item *ItemBase) bool {
			if item.Size() != arr[j].Size() {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j], item)
			}
			j++
			k++
			return true
		})
		if j != len(arr) {
			b.Fatalf("expected: %v, got %v", len(arr), j)
		}
		if k != len(arr)-100 {
			b.Fatalf("expected: %v, got %v", len(arr)-100, k)
		}
	}
}

func BenchmarkDescendLessOrEqual(b *testing.B) {
	arr := perm(benchmarkTreeSize)
	tr := New[*ItemBase](*btreeDegree)
	for _, v := range arr {
		tr.ReplaceOrInsert(v)
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Less(arr[j])
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := len(arr) - 100
		k := len(arr)
		tr.DescendLessOrEqual(arr[len(arr)-100], func(item *ItemBase) bool {
			if item.Size() != arr[j].Size() {
				b.Fatalf("mismatch: expected: %v, got %v", arr[j], item)
			}
			j--
			k--
			return true
		})
		if j != -1 {
			b.Fatalf("expected: %v, got %v", -1, j)
		}
		if k != 99 {
			b.Fatalf("expected: %v, got %v", 99, k)
		}
	}
}

const cloneTestSize = 10000

func cloneTest(t *testing.T, b *BTree[*ItemBase], start int, p []*ItemBase, wg *sync.WaitGroup, trees *[]*BTree[*ItemBase], lock *sync.Mutex) {
	t.Logf("Starting new clone at %v", start)
	lock.Lock()
	*trees = append(*trees, b)
	lock.Unlock()
	for i := start; i < cloneTestSize; i++ {
		b.ReplaceOrInsert(p[i])
		if i%(cloneTestSize/5) == 0 {
			wg.Add(1)
			go cloneTest(t, b.Clone(), i+1, p, wg, trees, lock)
		}
	}
	wg.Done()
}

func TestCloneConcurrentOperations(t *testing.T) {
	b := New[*ItemBase](*btreeDegree)
	trees := []*BTree[*ItemBase]{}
	p := perm(cloneTestSize)
	var wg sync.WaitGroup
	wg.Add(1)
	go cloneTest(t, b, 0, p, &wg, &trees, &sync.Mutex{})
	wg.Wait()
	want := rang(cloneTestSize)
	t.Logf("Starting equality checks on %d trees", len(trees))
	for i, tree := range trees {
		if !reflect.DeepEqual(want, all(tree)) {
			t.Errorf("tree %v mismatch", i)
		}
	}
	t.Log("Removing half from first half")
	toRemove := rang(cloneTestSize)[cloneTestSize/2:]
	for i := 0; i < len(trees)/2; i++ {
		tree := trees[i]
		wg.Add(1)
		go func() {
			for _, item := range toRemove {
				tree.Delete(item)
			}
			wg.Done()
		}()
	}
	wg.Wait()
	t.Log("Checking all values again")
	for i, tree := range trees {
		var wantpart []*ItemBase
		if i < len(trees)/2 {
			wantpart = want[:cloneTestSize/2]
		} else {
			wantpart = want
		}
		if got := all(tree); !reflect.DeepEqual(wantpart, got) {
			t.Errorf("tree %v mismatch, want %v got %v", i, len(want), len(got))
		}
	}
}

func BenchmarkDeleteAndRestore(b *testing.B) {
	items := perm(16392)
	b.ResetTimer()
	b.Run(`CopyBigFreeList`, func(b *testing.B) {
		fl := NewFreeList[*ItemBase](16392)
		tr := NewWithFreeList(*btreeDegree, fl)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dels := make([]*ItemBase, 0, tr.Len())
			tr.Ascend(ItemIterator[*ItemBase](func(b *ItemBase) bool {
				dels = append(dels, b)
				return true
			}))
			for _, del := range dels {
				tr.Delete(del)
			}
			// tr is now empty, we make a new empty copy of it.
			tr = NewWithFreeList(*btreeDegree, fl)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
	b.Run(`Copy`, func(b *testing.B) {
		tr := New[*ItemBase](*btreeDegree)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dels := make([]*ItemBase, 0, tr.Len())
			tr.Ascend(ItemIterator[*ItemBase](func(b *ItemBase) bool {
				dels = append(dels, b)
				return true
			}))
			for _, del := range dels {
				tr.Delete(del)
			}
			// tr is now empty, we make a new empty copy of it.
			tr = New[*ItemBase](*btreeDegree)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
	b.Run(`ClearBigFreelist`, func(b *testing.B) {
		fl := NewFreeList[*ItemBase](16392)
		tr := NewWithFreeList(*btreeDegree, fl)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tr.Clear(true)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
	b.Run(`Clear`, func(b *testing.B) {
		tr := New[*ItemBase](*btreeDegree)
		for _, v := range items {
			tr.ReplaceOrInsert(v)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tr.Clear(true)
			for _, v := range items {
				tr.ReplaceOrInsert(v)
			}
		}
	})
}
