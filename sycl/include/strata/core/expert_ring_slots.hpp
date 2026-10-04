#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strata::core {
class ExpertRingSlots {
public:
    struct Reservation { size_t slot; bool cached; };

    void reset(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.assign(count, Slot{});
        where_.clear();
        next_ = 0;
    }

    Reservation acquire(int64_t key) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (slots_.empty()) throw std::logic_error("expert ring has no slots");
        for (;;) {
            const auto cached = where_.find(key);
            if (cached != where_.end()) return {cached->second, true};
            for (size_t i = 0; i < slots_.size(); ++i) {
                const size_t index = (next_ + i) % slots_.size();
                Slot& slot = slots_[index];
                if (slot.in_flight) continue;
                const auto old = where_.find(slot.key);
                if (old != where_.end() && old->second == index) where_.erase(old);
                slot = {key, true};
                next_ = (index + 1) % slots_.size();
                return {index, false};
            }
            ready_.wait(lock);
        }
    }

    void finish(size_t index, bool success) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            Slot& slot = slots_[index];
            if (success) where_[slot.key] = index;
            else slot.key = -1;
            slot.in_flight = false;
        }
        ready_.notify_all();
    }

private:
    struct Slot { int64_t key = -1; bool in_flight = false; };
    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<Slot> slots_;
    std::unordered_map<int64_t, size_t> where_;
    size_t next_ = 0;
};
}  // namespace strata::core
