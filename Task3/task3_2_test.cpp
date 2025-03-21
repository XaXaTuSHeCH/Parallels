#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

double fun_sin(double arg) { return std::sin(arg); }
double fun_sqrt(double arg) { return std::sqrt(arg); }
double fun_pow(double x, double y) { return std::pow(x, y); }

void check_file(const std::string& filename, const std::string& func_type) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Не удалось открыть " << filename << "\n";
        return;
    }
    std::string line;
    int total = 0, passed = 0;
    const double eps = 1e-4;
    while (std::getline(file, line)) {
        total++;
        std::istringstream iss(line);
        std::string temp;
        iss >> temp >> temp;
        std::string args_str;
        std::getline(iss, args_str, '|');
        args_str.erase(0, args_str.find(':') + 2);
        std::vector<double> args;
        std::istringstream args_iss(args_str);
        while (std::getline(args_iss, temp, ',')) {
            temp.erase(0, temp.find_first_not_of(" "));
            try {
                args.push_back(std::stod(temp));
            } catch (...) {
                std::cout << filename << " строка " << total << ": ошибка парсинга аргументов (" << line << ")" << std::endl;
                continue;
            }
        }
        std::string result_str;
        std::getline(iss, result_str);
        result_str.erase(0, result_str.find(':') + 2);
        double result;
        try {
            result = std::stod(result_str);
        } catch (...) {
            std::cout << filename << " строка " << total << ": ошибка парсинга результата (" << line << ")" << std::endl;
            continue;
        }
        double expected;
        bool valid = true;
        double epsilon = eps;
        if (func_type == "sin") {
            if (args.size() != 1) {
                std::cout << filename << " строка " << total << ": ожидался 1 аргумент (" << line << ")" << std::endl;
                valid = false;
            } else {
                expected = fun_sin(args[0]);
            }
        } else if (func_type == "sqrt") {
            if (args.size() != 1) {
                std::cout << filename << " строка " << total << ": ожидался 1 аргумент (" << line << ")" << std::endl;
                valid = false;
            } else {
                expected = fun_sqrt(args[0]);
            }
        } else if (func_type == "pow") {
            if (args.size() != 2) {
                std::cout << filename << " строка " << total << ": ожидались 2 аргумента (" << line << ")" << std::endl;
                valid = false;
            } else {
                expected = fun_pow(args[0], args[1]);
                epsilon = std::max(eps, std::abs(expected) * 1e-5);
            }
        }
        if (!valid) continue;
        double diff = std::abs(result - expected);
        if (diff > epsilon) { std::cout << filename << " строка " << total << ": ошибка (ожидалось " << expected
                    << ", получено " << result << ", разница " << diff << ") (" << line << ")" << std::endl;
        } else { passed++; }
    }
    std::cout << filename << ": " << passed << " из " << total << " тестов прошли успешно" << std::endl;
}

int main() {
    check_file("client1.txt", "sin");
    check_file("client2.txt", "sqrt");
    check_file("client3.txt", "pow");
    return 0;
}