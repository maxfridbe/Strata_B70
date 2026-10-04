#include "strata/core/expert_ring_slots.hpp"

#include <cassert>
#include <chrono>
#include <future>

using strata::core::ExpertRingSlots;

int main() {
    ExpertRingSlots ring;
    ring.reset(2);
    auto first = ring.acquire(7);
    auto duplicate = ring.acquire(7);
    assert(!first.cached && !duplicate.cached && first.slot != duplicate.slot);
    ring.finish(first.slot, true);
    ring.finish(duplicate.slot, true);
    auto replacement = ring.acquire(8);
    assert(replacement.slot == first.slot && !replacement.cached);
    auto cached = ring.acquire(7);
    assert(cached.cached && cached.slot == duplicate.slot);
    ring.finish(replacement.slot, true);

    ring.reset(2);
    first = ring.acquire(7);
    duplicate = ring.acquire(7);
    ring.finish(duplicate.slot, true);
    ring.finish(first.slot, false);
    cached = ring.acquire(7);
    assert(cached.cached && cached.slot == duplicate.slot);

    ring.reset(2);
    first = ring.acquire(1);
    auto second = ring.acquire(2);
    ring.finish(second.slot, true);
    auto skipped = ring.acquire(3);
    assert(skipped.slot == second.slot && !skipped.cached);

    std::promise<void> started;
    auto blocked = std::async(std::launch::async, [&] {
        started.set_value();
        return ring.acquire(4);
    });
    started.get_future().wait();
    assert(blocked.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
    ring.finish(skipped.slot, false);
    assert(blocked.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    auto resumed = blocked.get();
    assert(resumed.slot == skipped.slot && !resumed.cached);
    ring.finish(resumed.slot, true);
    ring.finish(first.slot, true);
    assert(ring.acquire(1).cached);
    assert(ring.acquire(4).cached);
}
