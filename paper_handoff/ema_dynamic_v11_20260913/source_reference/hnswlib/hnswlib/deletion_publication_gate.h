#pragma once

#include <mutex>
#if __cplusplus >= 201402L
#include <shared_mutex>
#endif

namespace hnswlib {

#if __cplusplus >= 201402L
using DeletionPublicationMutex = std::shared_timed_mutex;
using DeletionPublicationLock = std::shared_lock<DeletionPublicationMutex>;
#else
using DeletionPublicationMutex = std::mutex;
using DeletionPublicationLock = std::unique_lock<DeletionPublicationMutex>;
#endif

}  // namespace hnswlib
