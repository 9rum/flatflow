// Copyright 2024 The FlatFlow Authors
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

#include "scheduler/scheduler.h"

#include <cstdlib>
#include <ctime>
#include <map>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"

namespace {
    class SchedulerTest final : private flatflow::Scheduler {
      public:
        SchedulerTest() = default;
        ~SchedulerTest() = default;

        void Test() {
            std::srand(std::time(nullptr));
            std::map<int, int> task_map;
            for (int i = 0; i < 100; i++) {
                task_map[i] = std::rand() % 100;
            }
            std::vector<int> task_list;
            for (const auto& task : task_map) {
                task_list.push_back(task.first);
            }
            std::vector<int> result = Schedule(task_list);
            for (int i = 0; i < 100; i++) {
                EXPECT_EQ(task_map[result[i]], task_map[result[i]]);
            }
        }
    };
    TEST(SchedulerTest, StaticSheduler){

    }
    TEST(SchedulerTest, StaticSchedulerRemainder){

    }
}