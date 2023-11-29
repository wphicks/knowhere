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
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <set>
#include <thread>
#include <type_traits>

namespace raft_proto {

/* A wrapper used to ensure that an object is not being used while it is being
 * modified on another host thread or device stream
 */
template <typename T>
struct streamsafe_wrapper {
  using wrapped_type = T;

  template <typename... Args>
  explicit streamsafe_wrapper(raft::resources const& res, Args&&... args)
    : wrapped_{[this, &res, args = std::make_tuple(std::forward<Args>(args)...)]() {
        auto lock = modifier_lock{mtx_, res};
        return std::apply(
          [res](auto&&... args) { return std::make_unique<T>(res, std::forward<Args>(args)...); },
          std::move(args));
      }()}
  {
  }

  template <typename... LambdaTs>
  decltype(auto) apply(raft::resources const& res, LambdaTs&&... lambdas)
  {
    auto lock = modifier_lock{mtx_, res};
    if constexpr (sizeof...(lambdas) == 1) {
      return std::get<0>(std::make_tuple(std::move(lambdas)...))(res, get_wrapped());
    } else {
      return std::make_tuple([&res, this](auto&& f) { return f(res, get_wrapped()); }(lambdas)...);
    }
  }

  template <typename... LambdaTs>
  decltype(auto) apply(raft::resources const& res, LambdaTs&&... lambdas) const
  {
    auto lock = user_lock{mtx_, res};
    if constexpr (sizeof...(lambdas) == 1) {
      return std::get<0>(std::make_tuple(std::move(lambdas)...))(res, get_wrapped());
    } else {
      return std::make_tuple([&res, this](auto&& f) {
        auto res_from_pool = raft::resources(res);
        raft::resource::set_cuda_stream(res_from_pool, raft::resource::get_next_usable_stream(res));
        return f(res_from_pool, get_wrapped());
      }(lambdas)...);
    }
  }

  template <
    typename FirstT,
    typename... LambdaTs,
    std::enable_if_t<
      !std::is_same_v<std::remove_cvref<FirstT>, raft::resources>
    >* = nullptr
  >
  decltype(auto) apply(FirstT first_arg, LambdaTs&&... lambdas) const
  {
    auto lock = user_lock{mtx_};
    if constexpr (sizeof...(lambdas) == 0) {
      return first_arg(get_wrapped());
    } else {
      return std::make_tuple(
        first_arg(get_wrapped()),
        [this](auto&& f) {
          return f(get_wrapped());
        }(lambdas)...
      );
    }
  }

  // Synchronize all device-side work that has occurred on the underlying
  // object, including both modification and use
  auto synchronize() const { mtx_->synchronize(); }
  // Synchronize on any stream owned by res and remove those streams from
  // the sets requiring synchronization
  auto synchronize(raft::resources const& res) const { mtx_->synchronize(res); }
  // Synchronize on the given stream and remove that stream from
  // the sets requiring synchronization
  auto synchronize(raft::resources const& res, rmm::cuda_stream_view stream) const
  {
    mtx_->synchronize(res, stream);
  }

  // Synchronize on any stream owned by res if and only if it is among those
  // requiring synchronization and then remove it from the corresponding set.
  auto synchronize_if_required(raft::resources const& res) const { mtx_->synchronize_if_required(res); }
  // Synchronize on the given stream if and only if it is among those
  // requiring synchronization and then remove it from the corresponding set.
  auto synchronize_if_required(raft::resources const& res, rmm::cuda_stream_view stream) const
  {
    mtx_->synchronize_if_required(res, stream);
  }

 private:
  // A class for coordinating access to a resource that may be modified by some
  // threads/streams and used without modification by others.
  struct modification_mutex {

    void synchronize() const
    {
      // Grab a lock to prevent new users from adding work or new
      // modifiers from modifying
      auto tmp_lock = ordered_lock{mtx_};
      // Synchronize all streams that might be using or modifying the locked
      // object
      synchronize_modifiers();
      synchronize_users();
    }

    void synchronize(raft::resources const& res) const
    {
      // Prevent any new modification/use from launching during synchronization
      auto tmp_lock = ordered_lock{mtx_};
      auto stream   = raft::resource::get_cuda_stream(res).value();
      raft::resource::sync_stream(res);
      user_streams_.erase(stream);
      modifier_streams_.erase(stream);
      if (raft::resource::is_stream_pool_initialized(res)) {
        raft::resource::sync_stream_pool(res);
        for (auto stream_idx = std::size_t{}; stream_idx < raft::resource::get_stream_pool_size(res);
             ++stream_idx) {
          stream = raft::resource::get_stream_from_stream_pool(res, stream_idx).value();
          user_streams_.erase(stream);
          modifier_streams_.erase(stream);
        }
      }
    }

    void synchronize(raft::resources const& res, rmm::cuda_stream_view stream) const
    {
      // Prevent any new modification/use from launching during synchronization
      auto tmp_lock = ordered_lock{mtx_};
      raft::resource::sync_stream(res, stream);
      user_streams_.erase(stream.value());
      modifier_streams_.erase(stream.value());
    };

    // Synchronize the indicated stream only if it was used to modify or access
    // the locked object and has not yet been synchronized
    void synchronize_if_required(raft::resources const& res, rmm::cuda_stream_view stream) const
    {
      // Prevent any new modification/use from launching during synchronization
      auto tmp_lock = ordered_lock{mtx_};
      _synchronize_if_required(res, stream);
    }

    void synchronize_if_required(raft::resources const& res) const
    {
      // Prevent any new modification/use from launching during synchronization
      auto tmp_lock = ordered_lock{mtx_};
      _synchronize_if_required(res, raft::resource::get_cuda_stream(res));
      if (raft::resource::is_stream_pool_initialized(res)) {
        for (auto stream_idx = std::size_t{}; stream_idx < raft::resource::get_stream_pool_size(res);
             ++stream_idx) {
          _synchronize_if_required(res, raft::resource::get_stream_from_stream_pool(res, stream_idx));
        }
      }
    }

   private:
    void _synchronize_if_required(raft::resources const& res, rmm::cuda_stream_view stream) const
    {
      auto users_iter = user_streams_.find(stream.value());
      if (users_iter != std::end(user_streams_)) {
        user_streams_.erase(users_iter);
        raft::resource::sync_stream(res, stream);
      }
      auto modifiers_iter = modifier_streams_.find(stream.value());
      if (modifiers_iter != std::end(modifier_streams_)) {
        modifier_streams_.erase(modifiers_iter);
        raft::resource::sync_stream(res, stream);
      }
    }
    static void synchronize(std::set<cudaStream_t>& stream_set)
    {
      while (stream_set.size() != std::size_t{}) {
        for (auto stream : stream_set) {
          auto status = cudaStreamQuery(stream);
          if (status != cudaErrorNotReady) {
            // Whether this stream has succeeded or errored out, it is no
            // longer a stream we need to keep track of
            if (stream_set.erase(stream) != std::size_t{}) {
              // Throw on any error on the stream if it is still available
              if (status != cudaErrorInvalidResourceHandle) { RAFT_CUDA_TRY(status); }
              break;  // Do not continue to iterate on modified set
            }
          }
        }
      }
    }

    // Synchronize on any stream that might have modified the locked object
    void synchronize_modifiers() const { synchronize(modifier_streams_); }

    // Synchronize on any stream that accessed this object but did not modify
    // it
    void synchronize_users() const { synchronize(user_streams_); }

    void record_modification_streams(raft::resources const& res) const {
      // This can only be called in a context where the underlying
      // ordered_mutex has already been locked, so it is safe to modify this
      // set without taking a lock
      modifier_streams_.insert(raft::resource::get_cuda_stream(res).value());
      if (raft::resource::is_stream_pool_initialized(res)) {
        for (auto stream_idx = std::size_t{}; stream_idx < raft::resource::get_stream_pool_size(res);
             ++stream_idx) {
          modifier_streams_.insert(
            raft::resource::get_stream_from_stream_pool(res, stream_idx).value());
        }
      }
    }

    void acquire_for_modifier() const {
      // Prevent any new users from incrementing work counter and any other
      // modifiers from attempting to launch work of their own
      lock_ = std::make_unique<ordered_lock>(mtx_);
      // Wait until all work in progress is done
      while (currently_using_.load() != 0 && user_streams_.size() != 0) {
        if (currently_using_.load() == 0) {
          synchronize_users();
        } else {
          // Yield to user threads so they have a chance to decrement
          // currently_using_, since that occurs outside of a lock
          std::this_thread::yield();
        }
      };

      // Ensure that no other modifiers are still at work on device
      synchronize_modifiers();
    }

    void acquire_for_modifier(raft::resources const& mod_res) const
    {
      // Ensure that all other users and modifiers are done with their work
      acquire_for_modifier();
      // Keep track of any streams which might be modifying the locked object
      record_modification_streams(mod_res);
      std::atomic_thread_fence(std::memory_order_acquire);
    }

    void release_from_modifier() const
    {
      std::atomic_thread_fence(std::memory_order_release);
      lock_.reset();
    }

    void record_user_streams(raft::resources const& res) const {
      // Any stream on this resource is a potential user of this object. Track
      // them all in order to synchronize before modification.
      user_streams_.insert(raft::resource::get_cuda_stream(res).value());
      if (raft::resource::is_stream_pool_initialized(res)) {
        for (auto stream_idx = std::size_t{}; stream_idx < raft::resource::get_stream_pool_size(res);
             ++stream_idx) {
          user_streams_.insert(raft::resource::get_stream_from_stream_pool(res, stream_idx).value());
        }
      }
    }

    void acquire_for_user() const
    {
      auto tmp_lock = ordered_lock{mtx_};
      // Ensure that all modifying threads have completed their work on device
      synchronize_modifiers();
      ++currently_using_;
    }
    void acquire_for_user(raft::resources const& user_res) const
    {
      auto tmp_lock = ordered_lock{mtx_};
      // Ensure that all modifying threads have completed their work on device
      synchronize_modifiers();
      record_user_streams(user_res);
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
    // The streams which have been used or may have been used to access but
    // not modify the locked object.
    // Q: Why not simply store the raft::resources objects themselves?
    // A: Because those objects may go out of scope before we need to
    // query them in order to guarantee that user streams have completed work before modification
    // streams. The streams may have been destroyed, but we can at least query
    // them and ignore the cudaErrorInvalidResourceHandle that gets
    // returned.
    mutable std::set<cudaStream_t> user_streams_{};
    mutable std::set<cudaStream_t> modifier_streams_{};
    friend struct modifier_lock;
    friend struct user_lock;
  };

  // A lock acquired to use but not modify the wrapped object. We ensure that
  // only const methods can be accessed while protected by this lock.
  struct user_lock {
    // We allow the constructor to be public here, since the struct itself is
    // private within streamsafe_wrapper
    user_lock(std::shared_ptr<modification_mutex> mtx, raft::resources const& res)
      : mtx_{[mtx, &res]() {
          mtx->acquire_for_user(res);
          return mtx;
        }()}
    {
    }
    user_lock(std::shared_ptr<modification_mutex> mtx)
      : mtx_{[mtx, &res]() {
          mtx->acquire_for_user();
          return mtx;
        }()}
    {
    }
    ~user_lock() { mtx_->release_from_user(); }

   private:
    std::shared_ptr<modification_mutex> mtx_;
  };

 public:
  // A lock acquired to modify the wrapped object.
  struct modifier_lock {
    ~modifier_lock() { mtx_->release_from_modifier(); }

   private:
    // This constructor ensures that only the given resources object can modify
    // the wrapped object during its lifetime
    modifier_lock(std::shared_ptr<modification_mutex> mtx,
                  raft::resources const& res)
      : mtx_{[mtx, &res]() {
          mtx->acquire_for_modifier(res);
          return mtx;
        }()}
    {
    }
    // This constructor allows resources objects to be recorded as modifiers
    // after the lock has been created
    modifier_lock(std::shared_ptr<modification_mutex> mtx)
      : mtx_{[mtx, &res]() {
          mtx->acquire_for_modifier();
          return mtx;
        }()}
    {
    }

    void record_modification_streams(raft::resources const& res) {
      mtx_->record_modification_streams(res);
    }

    std::shared_ptr<modification_mutex> mtx_;
    friend struct streamsafe_wrapper;
  };


  [[nodiscard]] auto get_lock() const {
    return std::make_shared<modifier_lock>(mtx_);
  }

  template <typename... LambdaTs>
  decltype(auto) apply(
    std::shared_ptr<modifier_lock> lock,
    raft::resources const& res,
    LambdaTs&&... lambdas
  ) {
    if constexpr (sizeof...(lambdas) == 1) {
      lock->record_modification_streams(res);
      return std::get<0>(std::make_tuple(std::move(lambdas)...))(res, get_wrapped());
    } else {
      return std::make_tuple([lock, &res, this](auto&& f) {
        auto res_from_pool = raft::resources(res);
        lock->record_modification_streams(res_from_pool);
        raft::resource::set_cuda_stream(res_from_pool, raft::resource::get_next_usable_stream(res));
        return f(res_from_pool, get_wrapped());
      }(lambdas)...);
    }
  }

 private:
  auto& get_wrapped() { return *wrapped_; }
  auto const& get_wrapped() const { return *wrapped_; }
  std::shared_ptr<modification_mutex> mtx_ = std::make_shared<modification_mutex>();
  std::unique_ptr<T> wrapped_;
};

}  // namespace raft
