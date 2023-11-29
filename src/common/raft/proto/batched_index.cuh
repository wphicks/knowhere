#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <type_traits>
#include <raft/core/pinned_mdarray.cuh>
#include "common/raft/proto/streamsafe_wrapper.hpp"

namespace raft_proto {

template <typename SecondaryIndexType, typename T, typename IdxT, typename InputIdxT=std::int64_t>
struct batched_index {
  using secondary_index_type = SecondaryIndexType;
  using data_type = std::remove_cvref_t<T>;
  using vector_id_type = IdxT;
  using numeric_index_type = InputIdxT;

  auto static constexpr vector_index_kind = secondary_index_type::vector_index_kind;

  batched_index(
    raft::resources const& res,
    numeric_index_type vector_element_count,
    numeric_index_type max_brute_force_size,
    numeric_index_type initial_capacity = numeric_index_type{1}
  ) : indexed_data_{res, vector_element_count, initial_capacity},
    mod_lock_{indexed_data_.get_lock()} {}

  auto capacity() const {
    return indexed_data_.apply(
      [](auto&& storage) { return storage.capacity(); }
    );
  }

  template <typename mdspan_type>
  std::enable_if_t<
    raft::is_mdspan<mdspan_type>::value &&
    std::is_assignable_v<
      typename storage_mdspan_type::reference_type,
      typename mdspan_type::reference_type,
    >
  > add(raft::resources const& res, mdspan_type new_data) {
    auto start_row = (added_vector_count_ += new_data.extent(0));
    auto end_row = start_row + new_data.extent(0);
    auto shared_lock = mod_lock_.lock();
    if (!shared_lock) {
      shared_lock = indexed_data_.get_lock();
      mod_lock_.store(shared_lock);
    }

    // We now have a valid lock, so we check if the capcity is sufficient for
    // the needs of this thread. If not, we resize the storage until it is.
    while (
      end_row > indexed_data_.apply(
        shared_lock,
        res,
        [](auto const& inner_res, auto&& storage) { return storage.capacity(); }
      )
    ) {
      // Exactly one thread will keep a copy of the old lock. The rest will get
      // null weak_ptrs once any thread requires more capacity than currently
      // allocated.
      auto weak_lock =
        mod_lock_.exchange(std::shared_ptr<modifier_lock_type>{nullptr});
      if (shared_lock = weak_lock.lock()) {
        // If this thread grabbed a non-null lock, it is responsible for the
        // resize. It must now wait until all other threads have finished their
        // modification operations and then grab a new lock for itself.

        // Try to get a new lock. This will not proceed until all other threads
        // are done using the existing lock.
        shared_lock = indexed_data_.get_lock();

        // At this point, no other thread can be using the old lock, so we
        // allocate the new storage we require. The new lock has not yet been
        // stored to mod_lock_, so no other thread can make concurrent
        // modifications until we're done.
        indexed_data_.apply(
          shared_lock,
          res,
          [&added_vector_count_](
            raft::resources const& inner_res,
            auto&& storage
          ) {
            return storage.reserve(inner_res, required_capacity);
          }
        );
        // Synchronize to ensure that allocation is complete before we allow
        // other threads to begin writing to the new allocation
        indexed_data_.synchronize_if_required(res);
        // Store the new lock. Other threads will now be able to load it and
        // use it in their write operations.
        mod_lock_.store(shared_lock);
      } else {
        // If the mod_lock_ seen by this thread was null, it just needs to wait
        // until the new lock is available. Another thread will take care of
        // increasing the capacity as necessary for this thread.
        while (!(shared_lock = mod_lock_.load())) {
          std::this_thread::yield();
        }
      }
    }
    // Because we now have a valid lock, we know that the capacity is
    // sufficient for us to write this thread's data. Furthermore, if the
    // capacity was already large enough, we grabbed the lock _before_ any
    // other thread could trigger a resize, and no other thread can trigger the
    // resize until we're done. Therefore, our data will definitely be copied
    // to a valid location, and if a subsequent allocation occurs, these data
    // will definitely be among those copied to the new destination
    indexed_data_.apply(
      shared_lock,
      res,
      [start_row, &new_data](
        raft::resources const& inner_res,
        auto&& storage
      ) {
        storage.add(inner_res, new_data, start_row);
      }
    );
  }

 private:
  // Making this a private static member function because it should ultimately
  // be replaced by a full submdspan implementation
  template <typename mdspan_type>
  static auto slice(
    mdspan_type orig,
    numeric_index_type start_row,
    numeric_index_type end_row
  ) {
    return mdspan_type{
      orig.data_handle() + start_row * orig.extent(1),
      {end_row - start_row, orig.extent(1)}
    };
  }

  using storage_mdspan_type = raft::pinned_matrix_view<
    data_type, numeric_index_type
  >;

  struct indexed_storage {
    indexed_storage(
      raft::resources const& res,
      numeric_index_type vector_element_count,
      numeric_index_type initial_capacity = numeric_index_type{1}
    ) : data_{raft::make_pinned_matrix<data_type>(
      res, initial_capacity, vector_element_count
    )} {}

    auto reserve(raft::resources const& res, numeric_index_type new_capacity) {
      if (new_capacity > data_.extent(0)) {
        auto new_data = raft::make_pinned_matrix<data_type>(
          res, data_.extent(1), std::max(data_.extent(0) * 2, new_capacity)
        );
        raft::copy(
          res,
          slice(new_data, numeric_index_type{}, data_.extent(0)),
          data_.view()
        );
      }

      return data_.extent(0);
    }

    template <typename mdspan_type>
    std::enable_if_t<
      raft::is_mdspan<mdspan_type>::value &&
      std::is_assignable_v<
        typename storage_mdspan_type::reference_type,
        typename mdspan_type::reference_type,
      >
    > add(raft::resources const& res, mdspan_type new_data, numeric_index_type start_row) {
      raft::copy(res, slice(data_, start_row, start_row + new_data.extent(0)), new_data);
      stored_vector_count_ += new_data.extent(0);
    }

    auto capacity() const noexcept {
      return data_.extent(0);
    }

    void build_secondary_index(raft::resources const& res) {
      secondary_index_ = secondary_index_type::build(
        res,
        index_params_,
        raft::mdbuffer(
          res,
          raft::mbduffer(current_data()),
          raft::memory_type::device
        ).view<raft::memory_type::device>()
      );
      migrated_count_ = secondary_index_.size();
    }

   private:
    auto current_data() const {
      return slice(data_.view(), numeric_index_type{}, stored_vector_count_);
    }
    auto current_data() {
      return slice(data_.view(), numeric_index_type{}, stored_vector_count_);
    }
    raft::pinned_matrix<data_type> data_;
    typename secondary_index_type::index_params_type index_params_;
    std::optional<secondary_index_type> secondary_index_ = std::nullopt;
    std::atomic<numeric_index_type> migrated_count_ = numeric_index_type{};
    std::atomic<numeric_index_type> stored_vector_count_ = numeric_index_type{};
  };
  using modifier_lock_type = typename streamsafe_wrapper<indexed_storage>::modifier_lock;

 private:
  streamsafe_wrapper<indexed_storage> indexed_data_;
  std::atomic<std::weak_ptr<modifier_lock_type>> mod_lock_{nullptr};
  std::atomic<numeric_index_type> added_vector_count_{};
  std::atomic<numeric_index_type> capacity_{};
};

}  // namespace raft_proto
