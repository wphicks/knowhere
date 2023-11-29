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
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include "common/raft/proto/ordered_lock.hpp"

namespace raft_proto {

/* This struct wraps an object which may be modified from some host threads
 * but used without modification from others. Because multiple users can safely
 * access the object simultaneously so long as it is not being modified, any
 * const access to a threadsafe_wrapper<T> will acquire a lock solely to
 * increment an atomic counter indicating that it is currently accessing the
 * underlying object. It will then decrement that counter once the const call
 * to the underlying object has been completed. Non-const access will
 * acquire a lock on the same underlying mutex but not proceed with the
 * non-const call until the counter reaches 0.
 *
 * A special lock (ordered_lock) ensures that the mutex is acquired in the
 * order that threads attempt to acquire it. This ensures that
 * modifying threads are not indefinitely delayed.
 *
 * Example usage:
 *
 * struct foo() {
 *   foo(int data) : data_{data} {}
 *   auto get_data() const { return data_; }
 *   void set_data(int new_data) { data_ = new_data; }
 *  private:
 *   int data_;
 * };
 *
 * auto f = threadsafe_wrapper<foo>{5};
 * f->set_data(6);
 * f->get_data();  // Safe but inefficient. Returns 6.
 * std::as_const(f)->get_data();  // Safe and efficient. Returns 6.
 * std::as_const(f)->set_data(7);  // Fails to compile.
 */
template <typename T>
struct threadsafe_wrapper {
  template <typename... Args>
  threadsafe_wrapper(Args&&... args) : wrapped{std::make_unique<T>(std::forward<Args>(args)...)}
  {
  }

  struct locked_proxy {
    auto* operator->() { return wrapped_; }

   private:
    locked_proxy(T* wrapped, modifier_lock&& lock) : wrapped_{wrapped}, lock_{std::move(lock)} {}

    T* wrapped_;
    modifier_lock lock_;
    friend struct threadsafe_wrapper;
  };

  auto operator->() {
    return locked_proxy{wrapped.get(), modifier_lock{mtx_}};
  }
  auto operator->() const
  {
    return locked_proxy{wrapped.get(), user_lock{mtx_}};
  }

 private:
  // A class for coordinating access to a resource that may be modified by some
  // threads and used without modification by others.
  class modification_mutex {
    void acquire_for_modifier()
    {
      // Prevent any new users from incrementing work counter
      lock_ = std::make_unique<ordered_lock>(mtx_);
      // Wait until all work in progress is done
      while (currently_using_.load() != 0);
      std::atomic_thread_fence(std::memory_order_acquire);
    }
    void release_from_modifier() { lock_.reset(); }
    void acquire_for_user() const
    {
      auto tmp_lock = ordered_lock{mtx_};
      ++currently_using_;
    }
    void release_from_user() const
    {
      std::atomic_thread_fence(std::memory_order_release);
      --currently_using_;
    }
    mutable ordered_mutex mtx_{};
    mutable std::atomic<int> currently_using_{};
    mutable std::unique_ptr<ordered_lock> lock_{nullptr};
    friend struct modifier_lock;
    friend struct user_lock;
  };

  // A lock acquired to modify the wrapped object.
  struct modifier_lock {
    modifier_lock(modification_mutex& mtx)
      : mtx_{[&mtx]() {
          mtx.acquire_for_modifier();
          return &mtx;
        }()}
    {
    }
    ~modifier_lock() { mtx_->release_from_modifier(); }

   private:
    modification_mutex* mtx_;
  };

  // A lock acquired to use but not modify the wrapped object. We ensure that
  // only const methods can be accessed while protected by this lock.
  struct user_lock {
    user_lock(modification_mutex const& mtx)
      : mtx_{[&mtx]() {
          mtx.acquire_for_user();
          return &mtx;
        }()}
    {
    }
    ~user_lock() { mtx_->release_from_user(); }

   private:
    modification_mutex const* mtx_;
  };
  modification_mutex mtx_;
  std::unique_ptr<T> wrapped;
};

}  // namespace raft_proto
