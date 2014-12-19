/*
 ** Copyright 2014 Edward Walker
 **
 ** Description: Interface to thread pool - used mainly for running parallel tasks
 ** @author: Ed Walker
 */
#ifndef _THREAD_WORKERS_H_
#define _THREAD_WORKERS_H_

#include "ThreadPool.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <utility>
#include <atomic>
#include <vector>
#include <iostream>

class ThreadWorkers
{
	public:

		ThreadWorkers() {
			total_num_cpus = std::thread::hardware_concurrency();
			if (total_num_cpus < 1)
				total_num_cpus = 16;
			thrd_pool = new ThreadPool(total_num_cpus);
		}

		virtual ~ThreadWorkers() 
		{
			delete thrd_pool;
		}

		/**
		 * Returns the number of cpus
		 */
		int get_num_cpus() 
		{
			return total_num_cpus;
		}

		/**
		 * Given N iterations, returns the start iteration for thread ID tid.
		 */
		int start(int tid, int N) const
		{
			int C = total_num_cpus;
			if (N >= C) {
				return (tid * (N / C));
			} else {
				return (tid < N ? tid : 0);
			}
		}

		/**
		 * Given N iterations, returns the end iteration for thread ID tid
		 */
    	int end(int tid, int N) const
	    {
			int C = total_num_cpus;
			if (N >= C) {
				return ((tid == C-1) ? N : ((tid + 1) * (N / C)));
			} else {
				return (tid < N ? (tid + 1) : 0);
			}
		}

		/**
		 * Runs function on each thread in the pool.  Returns when all task completes.
		 */
		void run_workers(std::function<void(int)> func)
		{
			std::mutex cv_m;
			std::condition_variable cv;

			int run_count = total_num_cpus;

			for (int i = 0; i < total_num_cpus; ++i) {
				thrd_pool->enqueue([i, func, &cv, &cv_m, &run_count] {
						func(i);
						{
							std::unique_lock<std::mutex> grd(cv_m);
							if (--run_count == 0)
								cv.notify_one (); // wake the main thread
						}
					});
			}

			{
				std::unique_lock<std::mutex> grd(cv_m);
				cv.wait(grd, [&run_count] {return run_count == 0;}); // wait for notify
			}

			return;
		}

		/**
		 * Runs an ensemble of tasks in the thread pool
		 */
		template <typename T>
		void run_workers(std::vector<std::pair<std::function<void(T)>, T>> &work_items)
		{
			std::mutex cv_m;
			std::condition_variable cv;

			std::size_t run_count = work_items.size();

			for (auto & ele : work_items) {

				auto func = std::get<0>(ele);
				auto arg = std::get<1>(ele);

				thrd_pool->enqueue([func, arg, &cv, &cv_m, &run_count] {
						func(arg);
						{
							std::unique_lock<std::mutex> grd(cv_m);
							if (--run_count == 0)
								cv.notify_one (); // wake the main thread
						}
					});
			}

			{
				std::unique_lock<std::mutex> grd(cv_m);
				cv.wait(grd, [&run_count] {return run_count == 0;}); // wait for notify
			}

			return;
		}

		/**
		 * Returns an instance of this class
		 */
		static ThreadWorkers * getInstance() {
			/**
			 * §6.7 [stmt.dcl] p4
			 *
			 * If control enters the declaration concurrently while the variable is being initialized, 
			 * the concurrent execution shall wait for completion of the initialization.
			 */
			static ThreadWorkers * s_instance = new ThreadWorkers();
			return s_instance;
		}

	private:

		/**
		 * Total number of CPUs on this computer
		 */
		int total_num_cpus;

    	ThreadPool *thrd_pool;
};

#endif
