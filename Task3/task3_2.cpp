#include <iostream>
#include <cmath>
#include <queue>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include <random>
#include <fstream>
#include <utility>
#include <functional>
#include <unordered_map>

template<typename T>
T fun_sin(T arg) {
    return std::sin(arg);
}

template<typename T>
T fun_sqrt(T arg) {
    return std::sqrt(arg);
}

template<typename T>
T fun_pow(T x, T y) {
    return std::pow(x, y);
}

template<typename T>
class Task {
public:
    size_t id;
    std::function<T()> task_func;
    std::promise<T> result_promise;
    Task(size_t id, std::function<T()> task_func)
        : id(id), task_func(task_func) {}
    Task(Task&& other) noexcept = default;
    Task& operator=(Task&& other) noexcept = default;
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
};

template<typename T>
class Server {
private:
    std::queue<Task<T>> task_queue;
    std::mutex queue_mutex;
    std::unordered_map<size_t, std::future<T>> result_map;
    std::mutex result_mutex;
    std::atomic<bool> running;
    std::thread worker_thread;
    void process_tasks() {
        while (running) {
            Task<T> task(0, nullptr);
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (!task_queue.empty()) {
                    task = std::move(task_queue.front());
                    task_queue.pop();
                }
            }
            if (task.task_func) {
                T result = task.task_func();
                task.result_promise.set_value(result);
            }
        }
    }
public:
    Server() : running(false) {}
    void start() {
        running = true;
        worker_thread = std::thread(&Server::process_tasks, this);
    }
    void stop() {
        running = false;
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
    size_t add_task(std::function<T()> task_func) {
        static size_t task_id = 0;
        size_t current_id = task_id++;
        Task<T> task(current_id, task_func);
        std::future<T> future = task.result_promise.get_future();
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            result_map[current_id] = std::move(future);
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(std::move(task));
        }
        return current_id;
    }
    T request_result(size_t task_id) {
        std::future<T> future;
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            auto it = result_map.find(task_id);
            if (it != result_map.end()) {
                future = std::move(it->second);
                result_map.erase(it);
            } else {
                throw std::runtime_error("Task ID not found");
            }
        }
        return future.get();
    }
};

template<typename T>
class Client {
private:
    Server<T>& server;
    size_t num_tasks;
    std::function<std::pair<std::function<T()>, std::string>()> task_generator;
public:
    Client(Server<T>& server, size_t num_tasks, std::function<std::pair<std::function<T()>, std::string>()> task_generator)
        : server(server), num_tasks(num_tasks), task_generator(task_generator) {}
    void add_tasks_and_get_results(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Не удалось открыть файл: " << filename << std::endl;
            return;
        }
        for (size_t i = 0; i < num_tasks; ++i) {
            auto [task_func, inputs] = task_generator();
            size_t task_id = server.add_task(task_func);
            T result = server.request_result(task_id);
            file << "task: " << task_id << "| args: " << inputs << "| result: " << result << "\n";
        }
    }
};

int main() {
    const size_t num_tasks = 100;
    Server<double> server;
    server.start();
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    auto sin_task_generator = [&dis, &gen]() {
        double arg = dis(gen);
        std::function<double()> task = [arg]() { return fun_sin(arg); };
        std::string inputs = std::to_string(arg);
        return std::make_pair(task, inputs);
    };
    auto sqrt_task_generator = [&dis, &gen]() {
        double arg = dis(gen);
        std::function<double()> task = [arg]() { return fun_sqrt(arg); };
        std::string inputs = std::to_string(arg);
        return std::make_pair(task, inputs);
    };
    auto pow_task_generator = [&dis, &gen]() {
        double base = dis(gen);
        double exponent = dis(gen);
        std::function<double()> task = [base, exponent]() { return fun_pow(base, exponent); };
        std::string inputs = std::to_string(base) + ", " + std::to_string(exponent);
        return std::make_pair(task, inputs);
    };
    Client<double> client1(server, num_tasks, sin_task_generator);
    std::thread t1(&Client<double>::add_tasks_and_get_results, &client1, "client1.txt");
    Client<double> client2(server, num_tasks, sqrt_task_generator);
    std::thread t2(&Client<double>::add_tasks_and_get_results, &client2, "client2.txt");
    Client<double> client3(server, num_tasks, pow_task_generator);
    std::thread t3(&Client<double>::add_tasks_and_get_results, &client3, "client3.txt");
    t1.join();
    t2.join();
    t3.join();
    server.stop();
    return 0;
}