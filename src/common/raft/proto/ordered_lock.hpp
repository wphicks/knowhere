/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace raft_proto {

/** A mutex which yields to threads in the order in which they attempt to
 * acquire a lock.
 */
struct ordered_mutex {
  void lock()
  {
    auto scoped_lock = std::unique_lock<std::mutex>{raw_mtx_};
    auto ticket      = next_ticket_++;
    queue_control_.wait(scoped_lock, [ticket, this]() { return ticket == current_ticket_; });
  }

  void unlock()
  {
    auto scoped_lock = std::unique_lock<std::mutex>{raw_mtx_};
    ++current_ticket_;
    queue_control_.notify_all();
  }

 private:
  std::condition_variable queue_control_{};
  std::mutex raw_mtx_{};
  std::size_t next_ticket_{};
  std::size_t current_ticket_{};
};

/** A scoped lock based on `ordered_mutex`, which will be acquired in the order in which
 * threads attempt to acquire the underlying mutex */
struct ordered_lock {
  explicit ordered_lock(ordered_mutex& mtx)
    : mtx_{[&mtx]() {
        mtx.lock();
        return &mtx;
      }()}
  {
  }

  ~ordered_lock() { mtx_->unlock(); }

 private:
  ordered_mutex* mtx_;
};

}  // namespace raft_proto
