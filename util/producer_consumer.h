#pragma once

#ifndef __PRODUCER_CONSUMER_H__

#include <functional>
#include <thread>
#include <vector>
#include <boost/log/trivial.hpp>
#include "threadsafe_limited_queue.h"
#include "barrier.h"

template<typename T> class ProducerConsumer {
public:
    ProducerConsumer(int capacity, int consumers) : queue_(capacity), blocker_(consumers + 1) {
        consumers_ = consumers;
    }

    void StartProducer(std::function<bool(T&)>& func) {
        producer_thr_ = std::move(std::thread([this, func]() {
            T val;
            bool done = false;
            while (!done) {
                done = !func(val);
                queue_.push(val, done);
            }
        }));
    }

    template<typename V>
    void StartConsumer(std::function<void(T&, V&)>& processor, std::function<void(V&)>& updater) {
        auto worker = [this, &processor, &updater]() {
            T val;
            V output;
            while (queue_.pop(val)) {
                processor(val, output);
                updater(output);
            }

            BlockConsumer();
#ifdef _DEBUG
            BOOST_LOG_TRIVIAL(info) << "Thread done";
#endif // _DEBUG

        };

        consumers_thrs_.resize(consumers_);
        for (int i = 0; i < consumers_; ++i) {
            consumers_thrs_[i] = std::move(std::thread(worker));
        }
    }

    void BlockConsumer() {
        blocker_.Block();
    }

    void JoinConsumer() {
        for (auto& item : consumers_thrs_) {
            if (item.joinable())
                item.join();
        }
    }

    void BlockProducer() {
        if (producer_thr_.joinable())
            producer_thr_.join();
        else
            BOOST_LOG_TRIVIAL(error) << "Producer thread not joinable";
    }


private:
    int consumers_;
    std::thread producer_thr_;
    std::vector<std::thread> consumers_thrs_;
    ThreadSafeLimitedQueue<T> queue_;
    Barrier blocker_;
};

#endif // !__PRODUCER_CONSUMER_H__
