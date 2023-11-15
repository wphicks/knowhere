/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#undef RAFT_EXPLICIT_INSTANTIATE_ONLY
#include "common/raft/proto/raft_index_kind.hpp"
#include "common/raft/integration/raft_knowhere_index.cuh"
#include "common/raft/proto/filtered_search_instantiation.cuh"

RAFT_FILTERED_SEARCH_EXTERN(ivf_flat,
    raft_knowhere::raft_data_t<raft_proto::raft_index_kind::ivf_flat>,
    raft_knowhere::raft_indexing_t<raft_proto::raft_index_kind::ivf_flat>,
    raft_knowhere::raft_input_indexing_t<raft_proto::raft_index_kind::ivf_flat>,
    raft_knowhere::raft_data_t<raft_proto::raft_index_kind::ivf_flat>,
    raft_knowhere::knowhere_bitset_data_type,
    raft_knowhere::knowhere_bitset_indexing_type
)

namespace raft_knowhere {
template struct raft_knowhere_index<raft_proto::raft_index_kind::ivf_flat>;
}  // namespace raft_knowhere
