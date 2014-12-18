#include <iostream>
#include "ThreadPool2.h"

using namespace std;

void run_workers(ThreadPool &thrd_pool, std::function<void(int)> func, int num_cpus)
{

	std::mutex cv_m;
	std::condition_variable cv;

	int run_count = num_cpus;

	for (int i = 0; i < num_cpus; ++i) {
		thrd_pool.enqueue([i, func, &cv, &cv_m, &run_count] {
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


int main()
{

	ThreadPool thrd_pool(4);
	mutex prt_lock;

	auto func = [&] (int tid) {
		lock_guard<mutex> grd(prt_lock);
		cout << "Hello from worker " << tid << endl;
	};

	run_workers(thrd_pool, func, 100);
}
