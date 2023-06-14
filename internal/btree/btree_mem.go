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

//go:build ignore
// +build ignore

// This binary compares memory usage between btree and gollrb.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"time"

	"github.com/9rum/chronica/internal/btree"
	"github.com/petar/GoLLRB/llrb"
)

var (
	size   = flag.Int("size", 1000000, "size of the tree to build")
	degree = flag.Int("degree", 0, "degree of btree")
	gollrb = flag.Bool("llrb", false, "use llrb instead of btree")
)

type ItemWrapper struct {
	*btree.ItemBase
}

func NewItemWrapper(index, size int) *ItemWrapper {
	return &ItemWrapper{
		ItemBase: btree.NewItemBase(index, size),
	}
}

func (i ItemWrapper) Less(than llrb.Item) bool {
	if i.Size() == than.(*ItemWrapper).Size() {
		return i.Index() < than.(*ItemWrapper).Index()
	}
	return i.Size() < than.(*ItemWrapper).Size()
}

func main() {
	flag.Parse()
	vals := rand.Perm(*size)
	var (
		t, v  any
		stats runtime.MemStats
	)
	v = vals
	for i := 0; i < 10; i++ {
		runtime.GC()
	}
	fmt.Println("---------- BEFORE ----------")
	runtime.ReadMemStats(&stats)
	fmt.Printf("%+v\n", stats)
	start := time.Now()
	if *gollrb {
		tr := llrb.New()
		for _, v := range vals {
			tr.ReplaceOrInsert(NewItemWrapper(v, v))
		}
		t = tr // keep it around
	} else {
		tr := btree.New[*btree.ItemBase](*degree)
		for _, v := range vals {
			tr.ReplaceOrInsert(btree.NewItemBase(v, v))
		}
		t = tr // keep it around
	}
	fmt.Printf("%v inserts in %v\n", *size, time.Since(start))
	fmt.Println("---------- AFTER ----------")
	runtime.ReadMemStats(&stats)
	fmt.Printf("%+v\n", stats)
	for i := 0; i < 10; i++ {
		runtime.GC()
	}
	fmt.Println("---------- AFTER GC ----------")
	runtime.ReadMemStats(&stats)
	fmt.Printf("%+v\n", stats)
	if t == v {
		fmt.Println("to make sure vals and tree aren't GC'd")
	}
}